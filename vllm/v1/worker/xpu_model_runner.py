# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from contextlib import contextmanager
from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

_xpu_graph_profiling = False
_graphs_built_under_profiler: "weakref.WeakSet[_ReinstantiableXPUGraph]" = (
    weakref.WeakSet()
)


class _ReinstantiableXPUGraph(torch.xpu.XPUGraph):
    """XPUGraph that keeps its recorded graph so the executable can be rebuilt.

    PTI only traces kernels of executables finalized while it is collecting,
    and those executables replay incorrectly once collection stops. During a
    profiling window each graph is rebuilt on its first replay, and those
    graphs are rebuilt again when the window ends.
    """

    def __init__(self, keep_graph: bool = False) -> None:
        # pybind11 runs the C++ constructor from __init__, not __new__.
        super().__init__(True)

    def capture_end(self) -> None:
        super().capture_end()
        self.instantiate()

    def replay(self) -> None:
        if _xpu_graph_profiling and self not in _graphs_built_under_profiler:
            # Rebuilding all graphs at once under PTI exhausts host memory.
            self.instantiate()
            _graphs_built_under_profiler.add(self)
        super().replay()

    def reset(self) -> None:
        _graphs_built_under_profiler.discard(self)
        super().reset()


def begin_xpu_graph_profiling() -> None:
    """Call after the profiler starts; graphs rebuild lazily on next replay."""
    global _xpu_graph_profiling
    # Earlier replays must finish before their executables are replaced.
    torch.xpu.synchronize()
    _graphs_built_under_profiler.clear()
    _xpu_graph_profiling = True


def end_xpu_graph_profiling() -> int:
    """Call after the profiler stops; returns how many graphs were rebuilt."""
    global _xpu_graph_profiling
    _xpu_graph_profiling = False
    graphs = list(_graphs_built_under_profiler)
    _graphs_built_under_profiler.clear()
    if not graphs:
        return 0
    torch.xpu.synchronize()
    for graph in graphs:
        graph.instantiate()
    torch.xpu.synchronize()
    return len(graphs)


def _keep_xpu_graphs(vllm_config: VllmConfig) -> bool:
    profiler_config = vllm_config.profiler_config
    return profiler_config is not None and profiler_config.profiler == "torch"


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper(_keep_xpu_graphs(vllm_config)):
            super().__init__(vllm_config, device)


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper(_keep_xpu_graphs(vllm_config)):
            super().__init__(vllm_config, device)


class XPUMMEncoderModelRunner(MMEncoderModelRunner):
    """An encoder-only model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper(_keep_xpu_graphs(vllm_config)):
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper(keep_xpu_graphs: bool = False):
    # Replace cuda APIs with xpu APIs. Each callable gets its own functools.partial
    # so it is not the same object as torch.xpu.* (Torch Dynamo _get_handlers()
    # asserts on duplicate registration when cuda aliases xpu directly).
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    torch.cuda.stream = partial(torch.xpu.stream)
    torch.cuda.set_stream = partial(torch.xpu.set_stream)

    # torch.xpu.Event does not accept the ``blocking`` or ``external`` kwargs
    # that torch.cuda.Event supports, so drop them here.
    def _xpu_event(*args, blocking=None, external=None, **kwargs):
        return torch.xpu.Event(*args, **kwargs)

    torch.cuda.Event = _xpu_event
    if supports_xpu_graph():
        torch.cuda.graph = partial(torch.xpu.graph)
        torch.cuda.CUDAGraph = (
            _ReinstantiableXPUGraph if keep_xpu_graphs else torch.xpu.XPUGraph
        )
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
        torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing
    yield

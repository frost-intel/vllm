# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import os

import pytest
import torch
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from tests.utils import create_new_process_for_each_test
from vllm.platforms import current_platform
from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

# XPU-only: needs distinct torch.cuda vs torch.xpu current_stream symbols.
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not hasattr(torch.xpu, "current_stream"),
    reason="torch.xpu.current_stream is required",
)


# Child process: patched torch.cuda must not leak to other tests in the session.
@pytest.mark.forked
def test_torch_cuda_wrapper_allows_dynamo_handler_registration() -> None:
    """Guard against XPU CUDA shim breaking Torch Dynamo during AOT compile.

    Before the fix, ``_torch_cuda_wrapper`` assigned
    ``torch.cuda.current_stream = torch.xpu.current_stream`` (same function object).
    On the first AOT/profile run, Dynamo builds its in-graph handler table and
    registers ``torch.cuda.current_stream`` and ``torch.xpu.current_stream``
    separately; duplicate identity triggers::

        AssertionError: Handler already registered for <function current_stream ...>

    That surfaced as EngineCore failing in ``profile_run`` / ``_get_handlers()``.
    The fix uses distinct shim callables so both can be registered.

    This test replays the post-init state (wrapper applied, patches left on
    ``torch.cuda``) and checks that Dynamo's real ``_get_handlers()`` succeeds.
    """
    # Same entry point as XPUModelRunner.__init__ (patches persist after exit).
    with _torch_cuda_wrapper():
        pass

    # Fresh handler table build, as on first torch.compile / AOT in the worker.
    # Registers torch.cuda.current_stream and torch.xpu.current_stream separately;
    # if they are the same object (pre-fix alias), raises Handler already registered.
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
    TorchInGraphFunctionVariable._get_handlers()


def _sampling_graph_status(worker):
    manager = worker.model_runner.cudagraph_manager
    return {
        size: (id(entry["graph"]), entry["num_replays"])
        for size, entry in manager.sampling_graphs.items()
    }


def _reset_sampling_rng(worker):
    torch.xpu.manual_seed(1234)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires an XPU worker")
@create_new_process_for_each_test(method="spawn")
def test_sampling_graphs_captured_at_startup(
    vllm_runner, monkeypatch, tensor_parallel_size
):
    """Startup captures are reused by sampling RPCs without changing tokens."""
    from vllm import SamplingParams

    torch.xpu.init()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_XPU_USE_SAMPLER_KERNEL", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    prompts = ["The capital of France is", "The three primary colors are"]
    params = [
        SamplingParams(
            temperature=1.0, top_k=8, top_p=0.8, max_tokens=8, ignore_eos=True
        ),
        SamplingParams(
            temperature=1.0, top_k=40, top_p=0.95, max_tokens=8, ignore_eos=True
        ),
    ]
    results = []
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_XPU_USE_SAMPLER_GRAPH", str(int(enabled)))
        with vllm_runner(
            os.getenv("VLLM_TEST_MODEL", "facebook/opt-125m"),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=128,
            max_num_seqs=2,
            async_scheduling=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.5,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": [1, 2],
            },
        ) as runner:
            llm = runner.get_llm()
            before = llm.collective_rpc(_sampling_graph_status)
            if enabled:
                assert all(set(status) == {1, 2} for status in before)
                assert all(
                    count == 0 for status in before for _, count in status.values()
                )
            else:
                assert all(not status for status in before)
            llm.collective_rpc(_reset_sampling_rng)
            waves = []
            for wave_prompts in (prompts, prompts[::-1]):
                outputs = llm.generate(wave_prompts, params)
                waves.extend(output.outputs[0].token_ids for output in outputs)
            results.append(waves)
            after = llm.collective_rpc(_sampling_graph_status)
            if enabled:
                for initial, final in zip(before, after):
                    assert initial.keys() == final.keys()
                    assert all(initial[size][0] == final[size][0] for size in initial)
                    assert final[2][1] > 0
            fallback = llm.generate(
                prompts,
                SamplingParams(
                    temperature=1.0, top_k=8, seed=42, logprobs=1, max_tokens=2
                ),
            )
            results[-1].extend(output.outputs[0].token_ids for output in fallback)
            assert llm.collective_rpc(_sampling_graph_status) == after
    assert results[0] == results[1], (
        f"TP{tensor_parallel_size} sampling graph changed tokens: "
        f"normal={results[0]}, graph={results[1]}"
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm serve`` CLI subcommand."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.cli.serve import (
    _run_headless_replica_shard,
    _run_headless_replica_shards,
    run_headless,
)


class _StopAfterEngineConfig(Exception):
    pass


def test_headless_imports_reasoning_parser_plugin_before_engine_config():
    plugin_path = "/tmp/custom_reasoning_parser.py"
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=plugin_path,
    )
    engine_args = MagicMock()

    with (
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            return_value=engine_args,
        ),
        patch(
            "vllm.reasoning.ReasoningParserManager.import_reasoning_parser"
        ) as import_plugin,
    ):

        def create_engine_config(**kwargs):
            import_plugin.assert_called_once_with(plugin_path)
            raise _StopAfterEngineConfig

        engine_args.create_engine_config.side_effect = create_engine_config

        with pytest.raises(_StopAfterEngineConfig):
            run_headless(args)


def test_headless_replica_shard_sets_dp_ranks_before_executor():
    vllm_config = MagicMock()
    seen = {}
    executor = MagicMock()

    def make_executor(config, monitor_workers):
        parallel_config = config.parallel_config
        seen["ranks"] = (
            parallel_config.data_parallel_rank,
            parallel_config.data_parallel_index,
            parallel_config.data_parallel_rank_local,
        )
        assert monitor_workers is False
        return executor

    with patch(
        "vllm.entrypoints.cli.serve.MultiprocExecutor", side_effect=make_executor
    ):
        _run_headless_replica_shard(vllm_config, dp_rank=2, local_dp_rank=2)

    assert seen["ranks"] == (2, 2, 2)
    executor.start_worker_monitor.assert_called_once_with(inline=True)


def test_headless_starts_one_shard_per_local_replica():
    """A non-leader node of a multi-replica node group runs its shard of
    every local replica concurrently and stops them all when one exits."""
    vllm_config = MagicMock()
    vllm_config.parallel_config.data_parallel_rank = 0
    procs = []

    class FakeProc:
        def __init__(self, target, args, name):
            self.target, self.args, self.name = target, args, name
            self.sentinel = len(procs)
            self.started = self.terminated = False
            procs.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self, timeout=None):
            pass

        def kill(self):
            pass

    with (
        patch(
            "vllm.entrypoints.cli.serve.get_mp_context",
            return_value=MagicMock(Process=FakeProc),
        ),
        patch(
            "vllm.entrypoints.cli.serve.multiprocessing.connection.wait",
            return_value=[0],
        ),
    ):
        _run_headless_replica_shards(vllm_config, local_count=3)

    assert [proc.args[1:] for proc in procs] == [(0, 0), (1, 1), (2, 2)]
    assert all(proc.target is _run_headless_replica_shard for proc in procs)
    assert all(proc.started and proc.terminated for proc in procs)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config.parallel import ParallelConfig, external_launcher_dp_rank
from vllm.distributed.parallel_state import rank_layout_tensor
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

# (external_dp, dp, pp, pcp, tp)
SHAPES = [
    (1, 1, 1, 1, 1),
    (1, 2, 2, 1, 1),
    (1, 3, 2, 1, 4),
    (1, 2, 3, 1, 8),
    (1, 4, 2, 2, 2),
    (1, 2, 4, 2, 1),
    (2, 2, 2, 1, 2),
    (1, 1, 4, 1, 2),
    (1, 4, 1, 1, 2),
]
LAYOUTS = ["default", "pp_outer"]


def _groups(layout: torch.Tensor, dp: int, pp: int, pcp: int, tp: int):
    """Groups built the same way as in `initialize_model_parallel`."""

    def to_sets(t: torch.Tensor) -> list[list[int]]:
        return [g.tolist() for g in t.unbind(0)]

    return {
        "tp": to_sets(layout.reshape(-1, tp)),
        "pp": to_sets(layout.transpose(2, 4).reshape(-1, pp)),
        "dp": to_sets(layout.transpose(1, 4).reshape(-1, dp)),
        "ep": to_sets(layout.transpose(1, 2).reshape(-1, dp * pcp * tp)),
    }


def _is_contiguous_block(ranks: list[int]) -> bool:
    return sorted(ranks) == list(range(min(ranks), min(ranks) + len(ranks)))


@pytest.mark.parametrize("rank_layout", LAYOUTS)
@pytest.mark.parametrize(("edp", "dp", "pp", "pcp", "tp"), SHAPES)
def test_layout_is_permutation_with_consistent_dp_rank(
    rank_layout, edp, dp, pp, pcp, tp
):
    world_size = edp * dp * pp * pcp * tp
    layout = rank_layout_tensor(world_size, dp, pp, pcp, tp, rank_layout)
    assert layout.shape == (edp, dp, pp, pcp, tp)
    assert sorted(layout.flatten().tolist()) == list(range(world_size))
    if edp == 1:  # the external-launcher DP rank covers in-model DP only
        for dp_rank in range(dp):
            for rank in layout[:, dp_rank].flatten().tolist():
                assert (
                    external_launcher_dp_rank(
                        rank, world_size, dp, tp, pcp, rank_layout
                    )
                    == dp_rank
                )


@pytest.mark.parametrize("rank_layout", LAYOUTS)
@pytest.mark.parametrize(("edp", "dp", "pp", "pcp", "tp"), SHAPES)
def test_group_composition(rank_layout, edp, dp, pp, pcp, tp):
    world_size = edp * dp * pp * pcp * tp
    layout = rank_layout_tensor(world_size, dp, pp, pcp, tp, rank_layout)
    groups = _groups(layout, dp, pp, pcp, tp)
    for kind, size in (("tp", tp), ("pp", pp), ("dp", dp), ("ep", dp * pcp * tp)):
        # Every kind partitions all ranks into groups of the expected size.
        assert all(len(g) == size for g in groups[kind]), kind
        assert sorted(r for g in groups[kind] for r in g) == list(range(world_size))
    # TP groups are consecutive ranks in both layouts (adjacent local devices).
    assert all(_is_contiguous_block(g) for g in groups["tp"])

    if rank_layout == "pp_outer":
        # Each PP stage's EP group (DP x PCP x TP) is one contiguous rank block,
        # so a launcher packing a multiple of that many ranks per node keeps it
        # on one node; only PP traffic crosses nodes.
        assert all(_is_contiguous_block(g) for g in groups["ep"])
    else:
        # Each DP replica (PP x PCP x TP ranks) is one contiguous rank block.
        replicas = layout.reshape(edp * dp, -1)
        assert all(_is_contiguous_block(r.tolist()) for r in replicas.unbind(0))


@pytest.mark.parametrize(("edp", "dp", "pp", "pcp", "tp"), SHAPES)
def test_pp_outer_matches_default_without_dp_or_pp(edp, dp, pp, pcp, tp):
    if dp > 1 and pp > 1:
        pytest.skip("layouts differ only when both DP and PP are > 1")
    world_size = edp * dp * pp * pcp * tp
    assert torch.equal(
        rank_layout_tensor(world_size, dp, pp, pcp, tp, "pp_outer"),
        rank_layout_tensor(world_size, dp, pp, pcp, tp, "default"),
    )


@pytest.mark.parametrize(("edp", "dp", "pp", "pcp", "tp"), SHAPES)
def test_pp_outer_ep_groups_span_fewer_nodes(edp, dp, pp, pcp, tp):
    # With one PP stage of every replica per node (ranks_per_node = DP*PCP*TP),
    # pp_outer keeps every EP group on one node; default spreads an EP group
    # over PP-many nodes whenever DP and PP are both > 1.
    world_size = edp * dp * pp * pcp * tp
    ranks_per_node = dp * pcp * tp

    def max_nodes_per_ep_group(rank_layout):
        layout = rank_layout_tensor(world_size, dp, pp, pcp, tp, rank_layout)
        return max(
            len({r // ranks_per_node for r in g})
            for g in _groups(layout, dp, pp, pcp, tp)["ep"]
        )

    assert max_nodes_per_ep_group("pp_outer") == 1
    if dp > 1 and pp > 1:
        assert max_nodes_per_ep_group("default") > 1


def test_rank_layout_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert parser.parse_args([]).rank_layout == "default"
    args = parser.parse_args(["--rank-layout", "pp_outer"])
    assert EngineArgs.from_cli_args(args).rank_layout == "pp_outer"


@pytest.mark.parametrize("backend", ["mp", "uni", "ray"])
def test_pp_outer_requires_external_launcher(backend):
    with pytest.raises(ValueError, match="pp_outer"):
        ParallelConfig(rank_layout="pp_outer", distributed_executor_backend=backend)


def test_pp_outer_accepted_with_external_launcher():
    config = ParallelConfig(
        rank_layout="pp_outer", distributed_executor_backend="external_launcher"
    )
    assert config.rank_layout == "pp_outer"

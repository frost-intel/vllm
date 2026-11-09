#!/bin/bash
# Per-node half of bench_feature_ladder.pbs, started once per node by mpiexec.
#   Node 0 runs bench_metrics.py (the mp leader); node 1 hosts the headless workers.
# Arguments come from files written by the ladder script, one argument per line:
#   $ARM_DIR/engine_args.txt  shared engine arguments (both nodes)
#   $ARM_DIR/bench_args.txt   bench_metrics.py-only arguments (node 0)
set -uo pipefail

mapfile -t ENGINE_ARGS < "$ARM_DIR/engine_args.txt"
mapfile -t BENCH_ARGS < "$ARM_DIR/bench_args.txt"
LEADER_DONE=$ARM_DIR/leader.done

if [[ $PALS_RANKID == 0 ]]; then
  python "$BENCH_SCRIPT" "${ENGINE_ARGS[@]}" --node-rank 0 "${BENCH_ARGS[@]}"
  rc=$?
  touch "$LEADER_DONE"
  exit $rc
fi

vllm serve "${ENGINE_ARGS[@]}" --node-rank 1 --headless &
headless=$!
# The headless workers keep running after the leader exits, so stop them once it is done.
while kill -0 $headless 2>/dev/null && [[ ! -e $LEADER_DONE ]]; do
  sleep 5
done
kill -TERM $headless 2>/dev/null
sleep 20
kill -KILL $headless 2>/dev/null
wait $headless 2>/dev/null
exit 0

# SPDX-License-Identifier: Apache-2.0
"""Offline concurrency-sweep benchmark for DeepSeek-V3 (XPU).

Loads the model ONCE, then for each concurrency C sweeps:
  - TTFT phase : generate C prompts with max_tokens=1   -> e2e_ttft (prefill+1tok)
  - Full phase : generate C prompts with max_tokens=OSL -> e2e_full, output tokens

Derived metrics (two-phase, robust on XPU offline path):
  TTFT_s        = e2e_ttft                       (wall time to first token, batch)
  TPOT_s        = (e2e_full - e2e_ttft)/(OSL-1)  (per-output-token decode time)
  throughput    = total_output_tokens / e2e_full (tokens/s, whole batch)

Also opportunistically reports per-request RequestOutput.metrics (real TTFT/TPOT
mean+median) when vLLM populates them.

Emits machine-parseable RESULT lines (grep 'RESULT' to build the table).

With --data-parallel-size > 1 (or --engine async) it uses AsyncLLM instead of LLM:
with the mp executor, LLM rejects DP > 1, while AsyncLLM load-balances requests over
the DP replicas (the same engine path as `vllm serve` with one API server). The
measurement is unchanged. --engine llm/async forces one, to A/B them at DP=1.
"""
import asyncio
import statistics
import time

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="deepseek-ai/DeepSeek-V3")
    g = parser.add_argument_group("benchmark")
    g.add_argument("--concurrencies", type=str, default="1,16,32,48",
                   help="comma-separated batch sizes to sweep")
    g.add_argument("--input-len", type=int, default=1000)
    g.add_argument("--output-len", type=int, default=1000)
    g.add_argument("--warmup", type=int, default=1,
                   help="warmup generate() calls before timing (JIT/compile)")
    # The XPU sampler graph only replays for non-greedy requests with top-k/top-p set.
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=-1)
    g.add_argument("--engine", choices=["auto", "llm", "async"], default="auto",
                   help="auto: AsyncLLM when --data-parallel-size > 1 (LLM rejects "
                        "DP > 1 with the mp executor), else LLM. llm/async force one, "
                        "e.g. to A/B the two at DP=1.")
    return parser


def make_prompts(n, input_len):
    # Approximate input length by repetition (matches existing harness style).
    if input_len <= 50:
        base = "Hello, my name is"
    else:
        base = "The quick brown fox jumps over the lazy dog. " * (input_len // 10)
    return [base] * n


def real_metrics(outputs):
    """Pull true TTFT/TPOT from RequestOutput.metrics if vLLM populated them."""
    ttfts, tpots = [], []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if not m:
            continue
        at = getattr(m, "arrival_time", None)
        ft = getattr(m, "first_token_time", None)
        lt = getattr(m, "last_token_time", None)
        ntok = len(o.outputs[0].token_ids)
        if at is not None and ft is not None:
            ttfts.append(ft - at)
        if ft is not None and lt is not None and ntok > 1:
            tpots.append((lt - ft) / (ntok - 1))
    return ttfts, tpots


class AsyncEngine:
    """AsyncLLM with LLM.generate()'s interface: submits all prompts at once and
    returns the final RequestOutput of each, in order."""

    def __init__(self, args: dict):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def create():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**args))

        self._engine = self._loop.run_until_complete(create())
        self._num_requests = 0

    def generate(self, prompts, sampling_params):
        first = self._num_requests
        self._num_requests += len(prompts)

        async def one(i, prompt):
            final = None
            async for out in self._engine.generate(prompt, sampling_params, f"req-{i}"):
                final = out
            return final

        async def run():
            return await asyncio.gather(
                *(one(first + i, p) for i, p in enumerate(prompts)))

        return self._loop.run_until_complete(run())

    def shutdown(self):
        self._engine.shutdown()


def timed_generate(llm, prompts, max_tokens, sampling):
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True, **sampling)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    return outs, dt


def main(args: dict):
    concurrencies = [int(x) for x in args.pop("concurrencies").split(",") if x]
    input_len = args.pop("input_len")
    output_len = args.pop("output_len")
    warmup = args.pop("warmup")
    sampling = {k: args.pop(k) for k in ("temperature", "top_p", "top_k")}

    engine = args.pop("engine")
    use_async = engine == "async" or (
        engine == "auto" and args.get("data_parallel_size", 1) > 1)
    llm = AsyncEngine(args) if use_async else LLM(**args)
    print(f"engine: {'AsyncLLM' if use_async else 'LLM'}", flush=True)

    # warmup (covers Triton JIT / first-pass kernel compile so it's not in timings)
    if warmup > 0:
        wp = make_prompts(max(concurrencies), input_len)
        for _ in range(warmup):
            timed_generate(llm, wp, 4, sampling)

    print(f"\n{'='*70}")
    print(f"SWEEP input_len={input_len} output_len={output_len} "
          f"concurrencies={concurrencies} sampling={sampling}")
    print(f"{'='*70}")

    for C in concurrencies:
        prompts = make_prompts(C, input_len)

        # Phase A: TTFT (prefill + 1 token), batch of C
        _, e2e_ttft = timed_generate(llm, prompts, 1, sampling)

        # Phase B: full decode
        outs, e2e_full = timed_generate(llm, prompts, output_len, sampling)
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)

        tpot = (e2e_full - e2e_ttft) / max(1, (output_len - 1))
        thr = total_out / e2e_full if e2e_full > 0 else 0.0

        # real per-request metrics if available
        rttft, rtpot = real_metrics(outs)
        real_str = ""
        if rttft:
            real_str += (f" real_TTFT_ms(mean/med)="
                         f"{1000*statistics.mean(rttft):.1f}/"
                         f"{1000*statistics.median(rttft):.1f}")
        if rtpot:
            real_str += (f" real_TPOT_ms(mean/med)="
                         f"{1000*statistics.mean(rtpot):.1f}/"
                         f"{1000*statistics.median(rtpot):.1f}")

        # Leading newline: LLM.generate's progress bar ends without one, and parsers
        # match lines starting with "RESULT".
        print(
            f"\nRESULT concurrency={C} isl={input_len} osl={output_len} "
            f"TTFT_s={e2e_ttft:.3f} TPOT_ms={1000*tpot:.2f} "
            f"throughput_tok_s={thr:.2f} total_out_tokens={total_out} "
            f"e2e_full_s={e2e_full:.2f}{real_str}",
            flush=True,
        )

    print(f"{'='*70}\nSWEEP COMPLETE\n{'='*70}", flush=True)
    if use_async:
        llm.shutdown()


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    main(args)


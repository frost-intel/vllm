# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

## XPU Sampling Graphs

Set `VLLM_XPU_USE_SAMPLER_GRAPH=1` with the V2 model runner and model FULL
graphs enabled to capture separate logits-and-sampling graphs during startup
dummy execution. `sample_tokens` RPCs replay these graphs for eligible decode
batches.

This opt-in path requires the fused XPU sampler and an XPU-kernels build that
accepts a device-resident shared `[seed, offset]` tensor. Capture sizes come from
`cudagraph_capture_sizes`, limited to `max_num_seqs`. Only exact batch sizes
match; other batches use normal sampling. Sampling graph memory is included in
startup graph-memory profiling and released with the model graph manager.

Currently supported: TP>=1, PP>=1, DP1, non-speculative one-token decode, positive
temperature, and top-k/top-p sampling without explicit per-request seeds.
Grammar, logprobs, logits processors, LoRA, sampling masks, batch sharding,
context parallelism, and microbatching fall back to normal sampling.

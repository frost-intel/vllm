# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

logger = init_logger(__name__)


class XPUMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank + qk_rope_head_dim, e.g. 512 + 64 for DeepSeek-V3.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Pin to block_size 16 so the paged-decode kernel dispatches through
        # the kv_tile=_16 policy (ReduceK==1 -> no epilogue SLM reduction
        # buffer). This removes the Intel Xe SLM cap on packed-Q heads and
        # lets high-head-count MLA (e.g. DeepSeek-V3 at low tensor-parallel
        # size) run without the head_size_qk>512 rejection.
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size == 16

    @staticmethod
    def get_impl_cls() -> type["XPUMLAImpl"]:
        return XPUMLAImpl

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MLACommonMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class XPUMLAImpl(MLACommonImpl[MLACommonMetadata]):
    # The XPU FA2 paged-decode kernel returns the natural-log softmax LSE,
    # which the DCP combine kernels consume when decode context parallelism
    # is enabled.
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "XPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for XPUMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "XPUMLAImpl does not support FP8 KV cache yet"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        # q comes in as (ql_nope, q_pe); concatenate to [B, N, kv_lora + rope].
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        decode = attn_metadata.decode
        num_tokens = q.shape[0]
        num_seqs = decode.seq_lens.shape[0]

        # This decode path only supports one query token per sequence.
        # Speculative/MTP decode (query_len > 1) is not yet implemented.
        if num_tokens != num_seqs:
            raise NotImplementedError(
                "XPUMLAImpl.forward_mqa only supports single-token decode "
                f"(got {num_tokens} query tokens for {num_seqs} sequences). "
                "Multi-token (speculative) decode is not implemented yet."
            )

        # Reshape the paged KV latent cache to the 4D layout the kernel wants:
        # [num_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim].
        k_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        # V is a non-contiguous narrow view over the first kv_lora_rank
        # channels of the same buffer; the kernel honors per-tensor strides.
        v_cache = k_cache[..., : self.kv_lora_rank]

        # One query token per sequence => cu_seqlens_q = [0, 1, 2, ..., B].
        cu_seqlens_q = torch.arange(
            num_seqs + 1, dtype=torch.int32, device=q.device
        )
        seqused_k = decode.seq_lens.to(dtype=torch.int32)

        out, lse = flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=attn_metadata.max_seq_len,
            seqused_k=seqused_k,
            block_table=decode.block_table,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
            fa_version=2,
        )

        return out, lse

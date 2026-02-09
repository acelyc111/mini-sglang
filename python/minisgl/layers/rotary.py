from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        if torch.cuda.is_available():
            try:
                from flashinfer import apply_rope_with_cos_sin_cache_inplace  # type: ignore

                self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace
            except ImportError:
                # FlashInfer not available, use fallback
                self.apply_rope_with_cos_sin_cache_inplace = self._apply_rope_fallback
        else:
            self.apply_rope_with_cos_sin_cache_inplace = self._apply_rope_fallback

    def _apply_rope_fallback(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = False,
    ):
        # Fallback implementation of RoPE
        # query: [num_tokens, num_heads, head_size]
        # cos_sin_cache: [max_pos, 2 * head_size]
        # positions: [num_tokens]

        # Extract cos and sin
        rot_dim = cos_sin_cache.shape[-1] // 2

        # Ensure positions tensor has correct shape and type for indexing
        positions = positions.long()

        # Handle potential index out of bounds by clamping
        max_pos = cos_sin_cache.shape[0] - 1
        positions = torch.clamp(positions, 0, max_pos)

        cos = cos_sin_cache[positions, :rot_dim]  # [num_tokens, rot_dim]
        sin = cos_sin_cache[positions, rot_dim:]  # [num_tokens, rot_dim]

        cos = cos.unsqueeze(1)  # [num_tokens, 1, rot_dim]
        sin = sin.unsqueeze(1)  # [num_tokens, 1, rot_dim]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Apply to query and key
        # Assuming head_size == rot_dim for now (common case)
        # If head_size > rot_dim, we only rotate the first rot_dim elements

        q_embed = query[..., :rot_dim]
        k_embed = key[..., :rot_dim]

        # Handle case where rot_dim equals head_size to avoid dimension issues
        if rot_dim == query.shape[-1]:
            # No pass-through dimensions, just update directly
            q_embed = (q_embed * cos) + (rotate_half(q_embed) * sin)
            k_embed = (k_embed * cos) + (rotate_half(k_embed) * sin)
            query.copy_(q_embed)
            key.copy_(k_embed)
            return query, key
        else:
            q_pass = query[..., rot_dim:]
            k_pass = key[..., rot_dim:]

        q_embed = (q_embed * cos) + (rotate_half(q_embed) * sin)
        k_embed = (k_embed * cos) + (rotate_half(k_embed) * sin)

        if q_embed.dim() == 2:
            q_embed = q_embed.unsqueeze(1)
            k_embed = k_embed.unsqueeze(1)
        if q_pass.dim() == 2:
            q_pass = q_pass.unsqueeze(1)
            k_pass = k_pass.unsqueeze(1)

        if q_embed.shape[:-1] == q_pass.shape[:-1]:
            query.copy_(torch.cat((q_embed, q_pass), dim=-1))
            key.copy_(torch.cat((k_embed, k_pass), dim=-1))
        else:
            query_result = torch.zeros_like(query)
            key_result = torch.zeros_like(key)
            query_result[..., : q_embed.shape[-1]] = q_embed
            key_result[..., : k_embed.shape[-1]] = k_embed
            query.copy_(query_result)
            key.copy_(key_result)

        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        # Handle case where _apply_rope_fallback returns tuple
        if result is not None and isinstance(result, tuple):
            return result
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]

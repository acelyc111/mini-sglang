from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """
    Rotary Positional Embedding (RoPE).

    This class implements RoPE, a method for injecting positional information into
    token embeddings. It precomputes cosine and sine values for efficiency and
    uses FlashInfer for optimized application during the forward pass.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        """
        Initialize the RotaryEmbedding layer.

        Args:
            head_size: The size of the attention heads.
            rotary_dim: The dimension used for rotary embeddings. Must be equal to head_size.
            max_position_embeddings: The maximum sequence length supported.
            base: The base value for frequency calculation.
            post_process: Optional function to transform inverse frequencies (e.g., for RoPE scaling).
        """
        super().__init__()
        self.head_size = head_size
        # Currently, the implementation assumes rotary_dim equals head_size
        assert rotary_dim == head_size

        # Calculate inverse frequencies: 1 / (base ** (2i / d))
        # This creates a geometric sequence of frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Apply optional post-processing to inverse frequencies (e.g., for Llama 3 RoPE scaling)
        if post_process is not None:
            inv_freq = post_process(inv_freq)

        # Generate position indices [0, 1, ..., max_position_embeddings - 1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # Compute the angles (frequencies) by taking the outer product of positions and inverse frequencies
        # shape: (max_position_embeddings, rotary_dim // 2)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # Compute cosine and sine of the angles
        cos = freqs.cos()
        sin = freqs.sin()

        # Concatenate cosine and sine values to create the cache
        # The cache structure is expected by FlashInfer
        # shape: (max_position_embeddings, rotary_dim)
        # buffer, so don't load/save to state_dict
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)

        # FlashInfer currently supports specific head sizes
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to query and key tensors.

        Args:
            positions: Tensor containing position indices for each token.
            query: Query tensor to be modified in-place.
            key: Key tensor to be modified in-place.

        Returns:
            A tuple containing the modified query and key tensors.
        """
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
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

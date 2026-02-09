from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from minisgl.platforms import Platform

    # Validate input data
    assert all(isinstance(x, (int, float)) for x in data), (
        f"Invalid data types: {[type(x) for x in data]}"
    )

    # pin_memory is only supported on CUDA. On MPS, create on CPU first, then copy to MPS.
    if Platform.is_cuda():
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    elif Platform.is_mps():
        try:
            return torch.tensor(data, dtype=dtype, device=device)
        except Exception:
            cpu_tensor = torch.tensor(data, dtype=dtype, device="cpu")
            return cpu_tensor.to(device)
    else:
        return torch.tensor(data, dtype=dtype).to(device)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from minisgl.platforms import Platform

    # CPU/MPS fallback for non-CUDA platforms
    if not Platform.is_cuda():
        # Simple greedy or temperature sampling for CPU/MPS
        if temperatures is None or temperatures.item() == 0:
            return torch.argmax(logits, dim=-1)

        # Apply temperature scaling
        logits = logits / temperatures.unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)

        # Top-k sampling
        if top_k is not None:
            if isinstance(top_k, torch.Tensor):
                top_k_val = top_k.item() if top_k.numel() == 1 else top_k
            else:
                top_k_val = top_k

            if not isinstance(top_k_val, torch.Tensor):
                top_k_val = torch.tensor([top_k_val], dtype=torch.long, device=logits.device)

            k_val = int(top_k_val[0].item()) if top_k_val.numel() == 1 else int(top_k_val.item())
            values, indices = torch.topk(probs, k=k_val, dim=-1)
            probs = torch.zeros_like(probs).scatter_(-1, indices, values)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Top-p (nucleus) sampling
        if top_p is not None:
            if isinstance(top_p, torch.Tensor):
                top_p_val = top_p.item() if top_p.numel() == 1 else top_p
            else:
                top_p_val = top_p

            if not isinstance(top_p_val, torch.Tensor):
                top_p_val = torch.tensor([top_p_val], device=logits.device)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            top_p_threshold = top_p_val[0].item() if top_p_val.numel() == 1 else top_p_val.item()
            mask = cumulative_probs > top_p_threshold
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample from final probabilities
        probs_flat = probs.view(-1, probs.size(-1))
        samples = torch.multinomial(probs_flat, num_samples=1)
        return samples.squeeze(-1)

    # CUDA path using flashinfer
    try:
        import flashinfer.sampling as sampling  # type: ignore
    except ImportError:
        # Fallback to CPU/MPS implementation if flashinfer is not available
        return sample_impl(logits.float(), temperatures, top_k, top_p)

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.temperatures is None:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

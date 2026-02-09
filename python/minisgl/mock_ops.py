import torch
import torch.nn.functional as F


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    x_normed = x_normed.to(input_dtype)

    if weight is not None:
        x_normed = x_normed * weight

    if out is not None:
        out.copy_(x_normed)
        return out
    return x_normed


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    if residual is None:
        x.copy_(rmsnorm(x, weight, eps))
        return

    residual.add_(x)
    x.copy_(rmsnorm(residual, weight, eps))


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    indices = indices.to(torch.long)
    k_cache[indices] = k
    v_cache[indices] = v


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    import torch.nn.functional as F

    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    if vocab_range is not None:
        start, length = vocab_range
        indices = indices - start
        indices_mask = (indices < 0) | (indices >= length)
        # Avoid out of bounds access by clamping or masking
        valid_indices = indices.clone()
        valid_indices[indices_mask] = 0

        result = F.embedding(valid_indices, weights)
        result[indices_mask] = 0
        output.copy_(result)
    else:
        result = F.embedding(indices, weights)
        output.copy_(result)

    return output


def fast_compare_key(
    key: torch.Tensor,
    value: torch.Tensor,
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
):
    pass


def test_tensor(
    input: torch.Tensor,
    output: torch.Tensor,
):
    pass

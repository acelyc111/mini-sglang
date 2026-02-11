from typing import Tuple

import torch

from minisgl.utils.logger import init_logger
from .base import BaseOP

logger = init_logger(__name__)


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info_rank0(f"[RMSNormFused] Input x shape: {x.shape}")
        if residual is None:
            logger.info_rank0("[RMSNormFused] Residual is None")
            out = self.rmsnorm(x, self.weight, self.eps)
            logger.info_rank0(f"[RMSNormFused] Output out shape: {out.shape}")
            return out, x
        logger.info_rank0(f"[RMSNormFused] Input residual shape: {residual.shape}")
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        logger.info_rank0(
            f"[RMSNormFused] Post-fused_add_rmsnorm x shape: {x.shape}, residual shape: {residual.shape}"
        )
        return x, residual

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch

    # Check if CUDA is available before importing nvtx
    if torch.cuda.is_available():
        try:
            import torch.cuda.nvtx as nvtx

            use_nvtx = True
        except (ImportError, RuntimeError):
            use_nvtx = False
    else:
        use_nvtx = False

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if use_nvtx:
                display_name = name
                if layer_id_field and hasattr(self, layer_id_field):
                    display_name = name.format(getattr(self, layer_id_field))
                with nvtx.range(display_name):
                    return fn(self, *args, **kwargs)
            else:
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator

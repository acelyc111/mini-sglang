"""MiniSGL package root.

Keeping an explicit __init__.py ensures IDEs treat `minisgl` as a regular
Python package so imports like `from minisgl.distributed import DistributedInfo`
resolve correctly.
"""

from .distributed import (  # noqa: F401
    DistributedInfo,
    DistributedCommunicator,
    destroy_distributed,
    enable_pynccl_distributed,
    get_tp_info,
    set_tp_info,
    try_get_tp_info,
)

__all__ = [
    "DistributedInfo",
    "DistributedCommunicator",
    "destroy_distributed",
    "enable_pynccl_distributed",
    "get_tp_info",
    "set_tp_info",
    "try_get_tp_info",
]
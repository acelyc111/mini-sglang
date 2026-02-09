from minisgl.platforms import Platform

if Platform.is_cuda():
    from .index import indexing
    from .pynccl import PyNCCLCommunicator, init_pynccl
    from .radix import fast_compare_key
    from .store import store_cache
    from .tensor import test_tensor
else:
    from minisgl.mock_ops import fast_compare_key, indexing, store_cache, test_tensor

    class PyNCCLCommunicator:
        def __init__(self, *args, **kwargs):
            pass

        def all_reduce(self, *args, **kwargs):
            pass

        def all_gather(self, *args, **kwargs):
            pass

    def init_pynccl(*args, **kwargs):
        pass


__all__ = [
    "indexing",
    "fast_compare_key",
    "store_cache",
    "test_tensor",
    "init_pynccl",
    "PyNCCLCommunicator",
]

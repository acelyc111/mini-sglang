import sys
import torch


def is_macos():
    return sys.platform == "darwin"


def get_device_name():
    if is_macos():
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_cuda_available():
    return torch.cuda.is_available()



# Platform abstraction layer
class Platform:
    @staticmethod
    def is_cuda():
        return get_device_name() == "cuda"

    @staticmethod
    def is_mps():
        return get_device_name() == "mps"

    @staticmethod
    def get_device_name():
        return get_device_name()

    @staticmethod
    def set_device(device):
        if Platform.is_cuda():
            torch.cuda.set_device(device)
        # MPS doesn't support set_device in the same way, usually just use device object

    @staticmethod
    def synchronize():
        if Platform.is_cuda():
            torch.cuda.synchronize()
        elif Platform.is_mps():
            torch.mps.synchronize()

    @staticmethod
    def empty_cache():
        if Platform.is_cuda():
            torch.cuda.empty_cache()
        elif Platform.is_mps():
            torch.mps.empty_cache()

    @staticmethod
    def Stream(device=None):
        if Platform.is_cuda():
            return torch.cuda.Stream(device)
        return MockStream(device)

    @staticmethod
    def Event(enable_timing=False):
        if Platform.is_cuda():
            return torch.cuda.Event(enable_timing=enable_timing)
        return MockEvent(enable_timing=enable_timing)

    @staticmethod
    def reset_peak_memory_stats(device=None):
        if Platform.is_cuda():
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def current_stream(device=None):
        if Platform.is_cuda():
            return torch.cuda.current_stream(device)
        return MockStream(device)

    @staticmethod
    def set_stream(stream):
        if Platform.is_cuda():
            torch.cuda.set_stream(stream)

    @staticmethod
    def stream_context(stream):
        if Platform.is_cuda():
            return torch.cuda.stream(stream)
        return MockStream(stream.device)


class MockStream:
    def __init__(self, device=None, priority=0, **kwargs):
        self.device = device

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __eq__(self, other):
        return isinstance(other, MockStream)


class MockEvent:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def elapsed_time(self, end_event):
        return 0.0

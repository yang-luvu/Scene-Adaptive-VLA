import os
import threading

import torch


class NvmlPowerSampler:
    """Sample GPU power during one model forward pass."""

    def __init__(self, interval_ms: float):
        if interval_ms <= 0.0:
            raise ValueError(f"Power sample interval must be > 0 ms, got {interval_ms}")

        import pynvml

        self.interval_ms = float(interval_ms)
        self._interval_s = self.interval_ms / 1000.0
        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        try:
            self._handle = self._resolve_handle()
        except Exception:
            self._pynvml.nvmlShutdown()
            raise
        self._samples_w = []
        self._sample_error = None
        self._stop_event = None
        self._thread = None

    def _resolve_handle(self):
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            first_device = visible_devices.split(",")[0].strip()
            if first_device.startswith("GPU-"):
                return self._pynvml.nvmlDeviceGetHandleByUUID(first_device)
            return self._pynvml.nvmlDeviceGetHandleByIndex(int(first_device))
        return self._pynvml.nvmlDeviceGetHandleByIndex(int(torch.cuda.current_device()))

    def _append_sample(self):
        power_w = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        self._samples_w.append(power_w)

    def _sample_loop(self):
        try:
            while not self._stop_event.wait(self._interval_s):
                self._append_sample()
        except Exception as exc:
            self._sample_error = exc
            self._stop_event.set()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("NVML power sampler is already running")

        self._samples_w = []
        self._sample_error = None
        self._stop_event = threading.Event()
        # Boundary samples cover forwards shorter than the polling interval.
        self._append_sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop_event is None or self._thread is None:
            raise RuntimeError("NVML power sampler was not started")

        self._stop_event.set()
        self._thread.join()
        if self._sample_error is not None:
            error = self._sample_error
            self._stop_event = None
            self._thread = None
            raise error
        self._append_sample()
        self._stop_event = None
        self._thread = None

        return {
            "mean_w": sum(self._samples_w) / len(self._samples_w),
            "min_w": min(self._samples_w),
            "max_w": max(self._samples_w),
            "sample_count": len(self._samples_w),
        }

    def close(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._stop_event = None
            self._thread = None
        self._pynvml.nvmlShutdown()

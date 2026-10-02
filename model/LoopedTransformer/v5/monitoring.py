import numpy as np

from .config import WINDOW_EVENTS


class SimpleDriftMonitor:
    def __init__(self, reference_X, threshold=3.0, eps=1e-6):
        reference = np.asarray(reference_X, float)
        self.mean = reference.mean(axis=0)
        self.std = reference.std(axis=0) + eps
        self.threshold = threshold

    def update(self, x):
        normalized = (np.asarray(x, float) - self.mean) / self.std
        score = float(np.sqrt(np.mean(normalized * normalized)))
        return {"score": score, "warning": bool(score > self.threshold)}


class OnlineState:
    def __init__(self, max_events=WINDOW_EVENTS):
        self.max_events = max_events
        self.job_id = None
        self.buffer = []

    def update(self, event):
        if self.job_id is None or event["job_id"] != self.job_id:
            self.job_id = event["job_id"]
            self.buffer = []
        self.buffer.append(event)
        self.buffer = self.buffer[-self.max_events:]
        return list(self.buffer)
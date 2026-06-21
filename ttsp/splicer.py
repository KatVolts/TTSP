import random
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class AudioSplicer:
    def __init__(
        self,
        sample_rate: int = 22050,
        gap_mode: str = "zero",
        gap_ms: float = 0.0,
        trim_silence: bool = False,
        silence_threshold_db: float = -40.0,
        random_gap_range: tuple = (0.0, 100.0),
    ):
        self.sample_rate = sample_rate
        self.gap_mode = gap_mode
        self.gap_ms = gap_ms
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db
        self.random_gap_range = random_gap_range

    def _make_silence(self, duration_ms: float, sample_rate: int) -> np.ndarray:
        samples = int(duration_ms / 1000.0 * sample_rate)
        return np.zeros(samples, dtype=np.float32)

    def _get_gap_duration(self) -> float:
        if self.gap_mode == "zero":
            return 0.0
        elif self.gap_mode == "fixed":
            return self.gap_ms
        elif self.gap_mode == "random":
            return random.uniform(*self.random_gap_range)
        else:
            logger.warning(f"Unknown gap_mode '{self.gap_mode}', defaulting to zero")
            return 0.0

    def splice(self, clips: List[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.array([], dtype=np.float32)

        processed = []
        for clip in clips:
            if self.trim_silence:
                clip = self._trim_silence(clip)
            processed.append(clip)

        if len(processed) == 1:
            return processed[0]

        parts = []
        for i, clip in enumerate(processed):
            parts.append(clip)
            if i < len(processed) - 1:
                gap_ms = self._get_gap_duration()
                if gap_ms > 0:
                    parts.append(self._make_silence(gap_ms, self.sample_rate))

        return np.concatenate(parts)

    @staticmethod
    def _trim_silence(data: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        threshold = 10 ** (threshold_db / 20.0)
        mask = np.abs(data) > threshold

        if not np.any(mask):
            return data

        indices = np.where(mask)[0]
        start = indices[0]
        end = indices[-1]

        margin = max(1, int(0.02 * len(data)))
        start = max(0, start - margin)
        end = min(len(data), end + margin)

        return data[start:end]

import re
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_PHONEMES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
}


def _is_phoneme(stem: str) -> bool:
    base = re.sub(r'\d+$', '', stem)
    return base.upper() in VALID_PHONEMES


class SampleLibrary:
    def __init__(self, samples_dir: str):
        self.samples_dir = Path(samples_dir)
        self.words: Dict[str, Dict[str, List[str]]] = {}
        self.phonemes: Dict[str, Dict[str, List[str]]] = {}
        self._cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.scan()

    def scan(self):
        self.words = {}
        self.phonemes = {}
        self._cache = {}

        if not self.samples_dir.exists():
            logger.warning(f"Samples directory not found: {self.samples_dir}")
            return

        for voice_dir in sorted(self.samples_dir.iterdir()):
            if not voice_dir.is_dir():
                continue

            voice_name = voice_dir.name
            self.words[voice_name] = {}
            self.phonemes[voice_name] = {}

            for wav_file in voice_dir.glob("*.wav"):
                stem = wav_file.stem

                if _is_phoneme(stem):
                    ph_key = re.sub(r'\d+$', '', stem).upper()
                    if ph_key not in self.phonemes[voice_name]:
                        self.phonemes[voice_name][ph_key] = []
                    self.phonemes[voice_name][ph_key].append(str(wav_file))
                else:
                    key = stem.lower()
                    if key not in self.words[voice_name]:
                        self.words[voice_name][key] = []
                    self.words[voice_name][key].append(str(wav_file))

        logger.info(f"Loaded {len(self.words)} voice(s)")
        for vname in self.words:
            nw = len(self.words[vname])
            np_ = len(self.phonemes.get(vname, {}))
            logger.info(f"  {vname}: {nw} word(s), {np_} phoneme(s)")

    def load_clip(self, path: str) -> Tuple[np.ndarray, int]:
        if path in self._cache:
            return self._cache[path]

        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)

        self._cache[path] = (data, sr)
        return data, sr

    @staticmethod
    def trim_silence(data: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
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

    def get_word_clips(self, word: str, voice: str) -> Optional[List[str]]:
        key = word.lower().strip(".,!?;:'\"()[]{}")
        if voice in self.words and key in self.words[voice]:
            return self.words[voice][key]
        return None

    def get_phoneme_clips(self, phoneme: str, voice: str) -> Optional[List[str]]:
        if voice not in self.phonemes:
            return None

        full = phoneme.upper()
        base = full.rstrip("012")

        if full in self.phonemes[voice] and self.phonemes[voice][full]:
            return self.phonemes[voice][full]
        if base in self.phonemes[voice] and self.phonemes[voice][base]:
            return self.phonemes[voice][base]

        return None

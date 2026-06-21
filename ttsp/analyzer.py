import re
import random
import logging
from typing import List, Optional

import numpy as np
from g2p_en import G2p

from .library import SampleLibrary

logger = logging.getLogger(__name__)


class TextAnalyzer:
    def __init__(self, library: SampleLibrary, default_voice: str):
        self.library = library
        self.default_voice = default_voice
        self.g2p = G2p()

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s']", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.split()

    def resolve_word(self, word: str, voice: Optional[str] = None) -> List[tuple]:
        target_voice = voice or self.default_voice
        clips = self.library.get_word_clips(word, target_voice)

        if clips:
            chosen = random.choice(clips)
            data, sr = self.library.load_clip(chosen)
            return [(data, sr)]

        return self._resolve_phonemes(word, target_voice)

    def _resolve_phonemes(self, word: str, voice: str) -> List[tuple]:
        try:
            phonemes = self.g2p(word)
        except Exception as e:
            logger.warning(f"Failed to phonemize '{word}': {e}")
            return []

        phoneme_tokens = [p for p in phonemes if p.strip() and p.strip() != "'"]
        clips = []

        for ph in phoneme_tokens:
            ph = ph.strip()
            if not ph:
                continue

            clip_paths = self.library.get_phoneme_clips(ph, voice)
            if clip_paths:
                chosen = random.choice(clip_paths)
                data, sr = self.library.load_clip(chosen)
                clips.append((data, sr))
            else:
                logger.warning(f"Missing phoneme '{ph}' for voice '{voice}' (word: '{word}')")

        return clips

    def analyze(self, text: str, voice: Optional[str] = None) -> List[tuple]:
        words = self.tokenize(text)
        all_clips = []

        for word in words:
            clips = self.resolve_word(word, voice)
            all_clips.extend(clips)

        return all_clips

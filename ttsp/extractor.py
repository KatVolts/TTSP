import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class SampleExtractor:
    """Extract word clips from a recording using Whisper transcription."""

    def __init__(self, output_dir: str, voice_name: str = "voice1"):
        self.output_dir = Path(output_dir) / voice_name
        self.voice_name = voice_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_whisper(
        self,
        audio_path: str,
        whisper_model: str = "base",
    ) -> List[Dict]:
        """Use Whisper for word-level alignment.

        Returns list of segments: {word, start, end}
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Whisper not installed. Run: pip install openai-whisper"
            )

        audio_path = Path(audio_path)
        logger.info(f"Loading Whisper model '{whisper_model}'...")
        model = whisper.load_model(whisper_model)

        logger.info(f"Transcribing {audio_path}...")
        result = model.transcribe(str(audio_path), verbose=False, word_timestamps=True)

        segments = []
        for seg in result.get("segments", []):
            if "words" not in seg:
                continue
            for w in seg["words"]:
                word = w.get("word", "").strip()
                start = w.get("start", 0)
                end = w.get("end", 0)
                if word:
                    segments.append({
                        "word": word,
                        "start": start,
                        "end": end,
                    })

        logger.info(f"Whisper found {len(segments)} word segments")
        return segments

    def extract(
        self,
        audio_path: str,
        segments: List[Dict],
        apply_silence_trim: bool = True,
        trim_threshold_db: float = -40.0,
        min_rms: float = 0.02,
    ) -> Dict[str, int]:
        """Extract word WAVs from the recording.

        Returns dict with extraction stats.
        """
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        word_counts: Dict[str, int] = {}
        found_words = set()

        for seg in segments:
            word = seg["word"].strip().lower()
            word = re.sub(r"[^\w']", "", word)
            if not word:
                continue

            w_start = int(seg["start"] * sr)
            w_end = int(seg["end"] * sr)
            if w_end > w_start and w_end <= len(audio):
                word_clip = audio[w_start:w_end]
                if apply_silence_trim:
                    word_clip = self._trim_silence(word_clip, trim_threshold_db)

                if len(word_clip) > 0:
                    rms = float(np.sqrt(np.mean(word_clip ** 2)))
                    if rms < min_rms:
                        continue

                    found_words.add(word)
                    if word in word_counts:
                        variant = word_counts[word] + 1
                        filename = f"{word}{variant}.wav"
                    else:
                        variant = 0
                        filename = f"{word}.wav"
                        word_counts[word] = 0

                    out_path = self.output_dir / filename
                    sf.write(out_path, word_clip, sr, subtype="PCM_16")
                    word_counts[word] = variant

        stats = {
            "words": len(word_counts),
            "unique_words": len(found_words),
        }

        logger.info(f"Extraction complete: {stats}")
        return stats

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


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="TTSP - Extract word samples from a recording")
    parser.add_argument("audio", help="Path to recording WAV file")
    parser.add_argument("-o", "--output-dir", default="samples", help="Output samples directory")
    parser.add_argument("-v", "--voice", default="voice1", help="Voice folder name")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    parser.add_argument("--trim-threshold", type=float, default=-40.0, help="Silence threshold in dB")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    extractor = SampleExtractor(args.output_dir, args.voice)

    try:
        segments = extractor.run_whisper(args.audio, args.whisper_model)
    except ImportError as e:
        print(f"ERROR: {e}")
        return

    stats = extractor.extract(
        args.audio,
        segments,
        apply_silence_trim=not args.no_trim,
        trim_threshold_db=args.trim_threshold,
    )

    print(f"\nExtraction complete!")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Total clips: {stats['words']}")


if __name__ == "__main__":
    main()

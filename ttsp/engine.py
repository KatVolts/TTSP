import argparse
import logging
from typing import Optional

import numpy as np
import soundfile as sf

from .library import SampleLibrary
from .analyzer import TextAnalyzer
from .splicer import AudioSplicer

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TTSP:
    def __init__(
        self,
        samples_dir: str,
        voice: str = "voice1",
        sample_rate: int = 22050,
        gap_mode: str = "zero",
        gap_ms: float = 0.0,
        trim_silence: bool = False,
        silence_threshold_db: float = -40.0,
        random_gap_range: tuple = (0.0, 100.0),
    ):
        self.library = SampleLibrary(samples_dir)
        self.analyzer = TextAnalyzer(self.library, voice)
        self.splicer = AudioSplicer(
            sample_rate=sample_rate,
            gap_mode=gap_mode,
            gap_ms=gap_ms,
            trim_silence=trim_silence,
            silence_threshold_db=silence_threshold_db,
            random_gap_range=random_gap_range,
        )
        self.voice = voice
        self.sample_rate = sample_rate

    def synthesize_text(self, text: str, voice: Optional[str] = None) -> tuple:
        clips = self.analyzer.analyze(text, voice)
        if not clips:
            return np.array([], dtype=np.float32), self.sample_rate
        
        audio = self.splicer.splice([c[0] for c in clips])
        sr = clips[0][1]
        return audio, sr

    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        audio, sr = self.synthesize_text(text, voice)
        sf.write(output_path, audio, sr, subtype="PCM_16")
        logger.info(f"Wrote {len(audio)} samples ({len(audio) / sr:.2f}s) to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="TTSP - Text to Speech Poop")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file")
    parser.add_argument("-s", "--samples-dir", default="samples", help="Samples directory")
    parser.add_argument("-v", "--voice", default="voice1", help="Voice folder name")
    parser.add_argument(
        "-g", "--gap-mode",
        default="zero",
        choices=["zero", "fixed", "random"],
        help="Gap mode between clips"
    )
    parser.add_argument("--gap-ms", type=float, default=0.0, help="Gap duration in ms (fixed mode)")
    parser.add_argument("--random-gap-min", type=float, default=0.0, help="Min random gap in ms")
    parser.add_argument("--random-gap-max", type=float, default=100.0, help="Max random gap in ms")
    parser.add_argument("--trim-silence", action="store_true", help="Trim silence from clips")
    parser.add_argument("--silence-threshold", type=float, default=-40.0, help="Silence threshold in dB")
    parser.add_argument("--sr", type=int, default=22050, help="Output sample rate")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    engine = TTSP(
        samples_dir=args.samples_dir,
        voice=args.voice,
        sample_rate=args.sr,
        gap_mode=args.gap_mode,
        gap_ms=args.gap_ms,
        trim_silence=args.trim_silence,
        silence_threshold_db=args.silence_threshold,
        random_gap_range=(args.random_gap_min, args.random_gap_max),
    )

    engine.synthesize(args.text, args.output)


if __name__ == "__main__":
    main()

import json
import logging
import subprocess
import shutil
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# 100 most common English words as a reading script (natural sentences)
WORD_SCRIPT = """
The dog is a good boy. He is very big and strong.
I can see that it is time to go home now.
You and I are going to the park with my dog.
This is my first time at the beach with so many people.
We have been there before, but it was a different time.
She is going to call her friend to tell them the news.
He can find the book in the house, but where is it?
They are going to make a picture of the world now.
I think that it is time to get up and look at the sky.
How much water is in the glass? It is only half full.
The two boys are going to the store to get some candy.
She is going to use her new pen to write a letter.
We are going to see the old man who lives in the house.
What is the time? It is time to go to school now.
The world is a big place with many different people.
I know that it is time to go home, but I want to stay.
They are going to the park to play with their dog.
She is going to call him to tell him the news.
He is going to make a picture of the world for his friend.
I can see that it is time to go home now.
One of the boys will not come because he is sick.
She can do it all by herself, but she wants help.
They are going to take a walk in the park when it is done.
How much is the water? It is only a little, but it is good.
"""

# Derive target words from script (keeps them in sync)
_COMMON = set()
for _line in WORD_SCRIPT.strip().split("\n"):
    for _w in _line.split():
        _c = _w.strip(".,!?;:'\"()[]{}").lower()
        if _c:
            _COMMON.add(_c)
del _line, _w, _c
COMMON_WORDS = frozenset(_COMMON)


class SampleExtractor:
    """Extract word clips from a single recording using MFA alignment."""

    def __init__(self, output_dir: str, voice_name: str = "voice1"):
        self.output_dir = Path(output_dir) / voice_name
        self.voice_name = voice_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_script(self, script_path: str) -> str:
        """Write the word-coverage script to a text file for reading."""
        path = Path(script_path)
        path.write_text(WORD_SCRIPT.strip())
        logger.info(f"Script written to {path}")
        return str(path)

    def run_mfa(
        self,
        audio_path: str,
        script_path: Optional[str] = None,
        mfa_model: str = "english_us_arpa",
    ) -> List[Dict]:
        """Run MFA align and parse the resulting JSON.

        Uses the full WAV as a single utterance with the entire transcript
        joined into one line.

        If script_path is None, uses the built-in WORD_SCRIPT.

        Returns list of segments: {word, start, end}
        """
        audio_path = Path(audio_path)

        script_text = WORD_SCRIPT.strip()
        if script_path:
            script_text = Path(script_path).read_text().strip()

        sentences = [line.strip() for line in script_text.split("\n") if line.strip()]
        transcript = " ".join(sentences)

        # Find MFA executable
        mfa_exe = "mfa"
        conda_mfa = Path(r"C:\Users\Admin\miniconda3\Scripts\mfa.exe")
        conda_lib_bin = Path(r"C:\Users\Admin\miniconda3\Library\bin")
        if conda_mfa.exists():
            mfa_exe = str(conda_mfa)
        env = os.environ.copy()
        if conda_lib_bin.exists():
            env["PATH"] = str(conda_lib_bin) + os.pathsep + env.get("PATH", "")
        home = Path.home()
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)

        # Create temp directory for MFA
        tmpdir = Path.home() / ".ttsp_mfa_tmp"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir()
        corpus_dir = tmpdir / "corpus"
        corpus_dir.mkdir()
        wav_out = corpus_dir / "utt0.wav"
        shutil.copy2(audio_path, wav_out)

        # Transcript: single .txt file alongside WAV
        txt_file = corpus_dir / "utt0.txt"
        txt_file.write_text(transcript + "\n")

        # Generate dictionary with proper phonemes using g2p_en
        dict_path = tmpdir / "dictionary.txt"
        all_words = set()
        for sentence in sentences:
            for word in sentence.split():
                clean = word.strip(".,!?;:'\"()[]{}").lower()
                if clean:
                    all_words.add(clean)

        from g2p_en import G2p
        g2p = G2p()
        dict_lines = []
        for word in sorted(all_words):
            try:
                phones = g2p(word)
                phone_str = " ".join(p.strip().upper() for p in phones if p.strip() and p.strip() != "'")
                if phone_str:
                    dict_lines.append(f"{word} {phone_str}")
            except Exception:
                dict_lines.append(f"{word} AH")
        dict_path.write_text("\n".join(dict_lines) + "\n")

        # Run MFA
        out_dir = tmpdir / "output"
        model_path = mfa_model
        mfa_model_zip = Path.home() / "Documents/MFA/pretrained_models/acoustic" / f"{mfa_model}.zip"
        if mfa_model_zip.exists():
            model_path = str(mfa_model_zip)
        cmd = [
            mfa_exe, "align",
            str(corpus_dir),
            str(dict_path),
            model_path,
            str(out_dir),
            "-j", "1",
            "--beam", "100",
            "--retry_beam", "400",
            "--output_format", "json",
            "--overwrite",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.error(f"MFA failed: {result.stderr}")
            raise RuntimeError(f"MFA alignment failed: {result.stderr}")

        # Parse output JSON
        json_files = sorted(out_dir.glob("*.json"))
        json_files = [f for f in json_files if f.suffix == ".json"]
        if not json_files:
            raise FileNotFoundError(f"MFA output not found in {out_dir}")

        all_segments = []
        for json_file in json_files:
            segments = self._parse_mfa_textgrid_json(json_file)
            all_segments.extend(segments)

        return all_segments

    def _parse_mfa_textgrid_json(self, json_path: Path) -> List[Dict]:
        """Parse MFA 3.x TextGrid JSON (per-utterance format), word tier only."""
        with open(json_path) as f:
            data = json.load(f)

        tiers = data.get("tiers", {})
        word_entries = tiers.get("words", {}).get("entries", [])

        segments = []
        for w in word_entries:
            w_start, w_end, w_text = w
            segments.append({
                "word": w_text,
                "start": w_start,
                "end": w_end,
            })

        return segments

    def run_whisper(
        self,
        audio_path: str,
        whisper_model: str = "base",
    ) -> List[Dict]:
        """Use Whisper for word-level alignment.

        Whisper gives word timestamps. Works on long recordings without
        needing a dictionary or transcript file.

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
            word = re.sub(r"[^\w']", "", word)  # Strip punctuation for filename
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

        missing = COMMON_WORDS - found_words
        stats = {
            "words": len(word_counts),
            "unique_words": len(found_words),
            "coverage": f"{len(found_words)}/{len(COMMON_WORDS)}",
            "missing_words": sorted(missing),
        }

        logger.info(f"Extraction complete: {stats}")
        if missing:
            logger.warning(
                f"Missing words ({len(missing)}): {', '.join(sorted(missing)[:20])}"
            )

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
    parser.add_argument("audio", nargs="?", help="Path to recording WAV file")
    parser.add_argument("-o", "--output-dir", default="samples", help="Output samples directory")
    parser.add_argument("-v", "--voice", default="voice1", help="Voice folder name")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    parser.add_argument("--trim-threshold", type=float, default=-40.0, help="Silence threshold in dB")
    parser.add_argument("--generate-script", action="store_true", help="Generate reading script and exit")
    parser.add_argument("--script-out", default="reading_script.txt", help="Output path for reading script")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    extractor = SampleExtractor(args.output_dir, args.voice)

    if args.generate_script:
        extractor.generate_script(args.script_out)
        print("Script generated. Read it aloud, record, then run without --generate-script.")
        return

    if not args.audio:
        parser.error("audio argument is required (or use --generate-script)")

    # Use Whisper for word-level alignment
    try:
        segments = extractor.run_whisper(args.audio, args.whisper_model)
    except ImportError as e:
        print(f"ERROR: {e}")
        return

    # Extract clips
    stats = extractor.extract(
        args.audio,
        segments,
        apply_silence_trim=not args.no_trim,
        trim_threshold_db=args.trim_threshold,
    )

    print(f"\nExtraction complete!")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Total clips: {stats['words']}")
    print(f"  Coverage: {stats['coverage']}")
    if stats["missing_words"]:
        print(f"  Missing: {', '.join(stats['missing_words'][:20])}")


if __name__ == "__main__":
    main()

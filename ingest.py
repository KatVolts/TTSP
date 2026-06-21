import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ttsp.extractor import SampleExtractor

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <recording.wav> [--voice kat]")
        print()
        print("Extracts words from a recording and adds them to the voice library.")
        print("Phoneme clips are skipped — only word clips are extracted.")
        return

    audio = sys.argv[1]
    voice = "kat"
    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        if idx + 1 < len(sys.argv):
            voice = sys.argv[idx + 1]

    if not Path(audio).exists():
        print(f"ERROR: File not found: {audio}")
        return

    extractor = SampleExtractor("samples", voice)

    print(f"Transcribing {audio}...")
    segments = extractor.run_whisper(audio, "base")
    print(f"Found {len(segments)} word segments.")

    print("Extracting clips...")
    stats = extractor.extract(audio, segments)

    print(f"\nDone!")
    print(f"  New clips: {stats['words']}")
    print(f"  Unique words: {stats['unique_words']}")
    if stats["missing_words"]:
        print(f"  Still missing: {', '.join(stats['missing_words'][:15])}")


if __name__ == "__main__":
    main()

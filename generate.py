import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ttsp import TTSP

def get_output_path(base="output.wav"):
    p = Path(base)
    if not p.exists():
        return p
    i = 1
    while Path(f"{p.stem}{i}.wav").exists():
        i += 1
    return Path(f"{p.stem}{i}.wav")

def main():
    engine = TTSP("samples", voice="kat", gap_mode="zero")

    while True:
        try:
            text = input("Text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not text:
            continue

        out = get_output_path("output.wav")
        engine.synthesize(text, str(out))
        print(f"  -> {out}")

if __name__ == "__main__":
    main()

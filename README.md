# TTSP - Text to Speech Poop

A Python-based TTS engine that splices recorded word and phoneme clips together (YP / "poop" style). Drop in your own voice samples and get synthesized speech from any text.

## How It Works

1. **Record** yourself speaking naturally
2. **Extract** word clips using Whisper transcription
3. **Clip** phonemes manually for fallback coverage
4. **Generate** speech - words are matched exactly, missing words fall back to phoneme splicing via `g2p_en`

```
"hello"  → exact word match → hello.wav
"zzzzz"  → no match → g2p_en → Z AH1 T S B ER Z → phoneme clips spliced together
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Whisper needs a CUDA-capable GPU for reasonable speed (CPU works but is slow).

## Usage

### Extract Words from a Recording

```bash
# Quick way
python ingest.py recording.wav

# With custom voice name
python ingest.py recording.wav --voice kat

# Or via the extractor module directly
python -m ttsp.extractor recording.wav -o samples -v kat
```

Speak naturally - Whisper transcribes whatever you say and extracts word clips. No script needed.

### Clip Phonemes (Manual)

For words not in your library, TTSP falls back to phoneme splicing. Clip individual phonemes from your recordings and drop them into `samples/<voice>/`:

```
samples/kat/AH0.wav
samples/kat/AH1.wav
samples/kat/AH2.wav
samples/kat/EH.wav
samples/kat/Z.wav
...
```

See `phoneme_guide.txt` for the full ARPAbet phoneme list with example words.

### Generate Speech

```bash
# Interactive mode
python generate.py

# CLI mode
python -m ttsp.engine "hello world" -o output.wav -v kat
```

### Programmatic API

```python
from ttsp import TTSP

engine = TTSP("samples", voice="kat", gap_mode="zero")
audio, sr = engine.synthesize_text("the quick brown fox")

# Write to file
engine.synthesize("the quick brown fox", "output.wav")
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `voice` | `kat` | Voice folder name under `samples/` |
| `gap_mode` | `zero` | Gap between clips: `zero`, `fixed`, `random` |
| `gap_ms` | `0.0` | Gap duration in ms (fixed mode) |
| `trim_silence` | `False` | Trim silence from each clip before splicing |
| `silence_threshold_db` | `-40.0` | Silence detection threshold |

## Project Structure

```
ttsp/
├── engine.py        # TTSP class, CLI entry point
├── library.py       # Loads word and phoneme clips from samples/
├── analyzer.py      # Text → word match or phoneme fallback (g2p_en)
├── splicer.py       # Concatenates clips with configurable gaps
├── extractor.py     # Whisper transcription → word clip extraction
├── generate.py      # Interactive CLI
└── ingest.py        # Quick word extraction from new recordings
```

## Tips

- **More recordings = better coverage.** Run `python ingest.py` on any new recording to add words.
- **Phoneme quality matters.** Clip tightly, trim silence, include multiple stress variants (`AH0`, `AH1`, `AH2`).
- **Gap mode `zero`** gives the tightest speech. Use `fixed` or `random` if clips sound too choppy.

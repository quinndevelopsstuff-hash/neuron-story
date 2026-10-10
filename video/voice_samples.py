"""
Voice audition: speak the same three beats (1.2, 3.7, 6.8) with several voices, one file per
voice in video-out/voice-samples/, so a narrator can be chosen before the full audio build.

  python3 video/voice_samples.py                         # the three shortlisted Kokoro voices
  python3 video/voice_samples.py --voices af_heart bm_george
  python3 video/voice_samples.py --tts piper             # the Piper fallback, for comparison
"""

import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_audio import ROOT, OUT, read_story  # noqa: E402
from tts import KOKORO_VOICES, add_tts_args, make_engine, prepare_clip  # noqa: E402

BEATS = ["1.2", "3.7", "6.8"]
SR = 48000
GAP = 0.8  # seconds of silence between the beats


def main():
    ap = argparse.ArgumentParser()
    add_tts_args(ap)
    ap.add_argument("--voices", nargs="*", help="voices to audition (default: the Kokoro shortlist)")
    args = ap.parse_args()
    voices = args.voices or (list(KOKORO_VOICES) if args.tts == "kokoro" else [args.voice])

    beats = {b["id"]: b["text"] for c in read_story(os.path.join(ROOT, "STORY.md")) for b in c["beats"]}
    out_dir = os.path.join(OUT, "voice-samples")
    os.makedirs(out_dir, exist_ok=True)
    gap = np.zeros(int(GAP * SR), np.float32)

    for voice in voices:
        engine = make_engine(args, voice=voice)
        parts, seconds = [], []
        for bid in BEATS:
            x = prepare_clip(*engine.synth(beats[bid]), SR)
            parts += [x, gap]
            seconds.append(len(x) / SR)
        audio = np.clip(np.concatenate(parts), -1, 1)
        path = os.path.join(out_dir, f"{engine.name}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((audio * 32767).astype("<i2").tobytes())
        words = sum(len(beats[b].split()) for b in BEATS)
        print(f"{os.path.relpath(path, ROOT)}: " + ", ".join(f"{b} {s:.1f}s" for b, s in zip(BEATS, seconds))
              + f"  (~{words / sum(seconds) * 60:.0f} words/min)")


if __name__ == "__main__":
    main()

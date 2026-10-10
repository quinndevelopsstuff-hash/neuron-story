"""
Voice audition: speak the same three beats (1.2, 3.7, 6.8) with several voices, one file per
voice in video-out/voice-samples/, so a narrator can be chosen before the full audio build.

  python3 video/voice_samples.py                         # the three shortlisted Kokoro voices
  python3 video/voice_samples.py --voices af_heart bm_george
  python3 video/voice_samples.py --tts piper             # the Piper fallback, for comparison
  python3 video/voice_samples.py --script video          # the video script's wording
  python3 video/voice_samples.py --script video --voices bf_emma --beats 1.1-1.5
                                         # specific beats (IDs or ranges), one file per beat
                                         # plus one file with all of them
"""

import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_audio import OUT, SCRIPTS, load_script  # noqa: E402
from tts import KOKORO_VOICES, add_tts_args, make_engine, prepare_clip  # noqa: E402

BEATS = ["1.2", "3.7", "6.8"]
SR = 48000
GAP = 0.8  # seconds of silence between the beats


def main():
    ap = argparse.ArgumentParser()
    add_tts_args(ap)
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="site", help="site (STORY.md) or video")
    ap.add_argument("--voices", nargs="*", help="voices to audition (default: the Kokoro shortlist)")
    ap.add_argument("--beats", nargs="*", help="beat IDs or ranges, e.g. 1.1-1.5 3.7 (default: 1.2 3.7 6.8)")
    args = ap.parse_args()
    voices = args.voices or (list(KOKORO_VOICES) if args.tts == "kokoro" else [args.voice])

    beats = {b["id"]: b["text"] for c in load_script(args.script) for b in c["beats"]}
    chosen = expand_beats(args.beats, list(beats)) if args.beats else BEATS
    custom = bool(args.beats)
    out_dir = os.path.join(OUT, "voice-samples")
    os.makedirs(out_dir, exist_ok=True)
    gap = np.zeros(int(GAP * SR), np.float32)

    for voice in voices:
        engine = make_engine(args, voice=voice)
        parts, seconds = [], []
        stem = f"{engine.name}-{args.script}" if custom else engine.name
        for bid in chosen:
            x = prepare_clip(*engine.synth(beats[bid]), SR)
            parts += [x, gap]
            seconds.append(len(x) / SR)
            if custom:
                write(os.path.join(out_dir, f"{stem}-{bid}.wav"), x)
        name = f"{stem}-{chosen[0]}-{chosen[-1]}.wav" if custom else f"{stem}.wav"
        path = os.path.join(out_dir, name)
        write(path, np.concatenate(parts))
        words = sum(len(beats[b].split()) for b in chosen)
        print(f"{os.path.relpath(path, os.path.dirname(OUT))}: " + ", ".join(f"{b} {s:.1f}s" for b, s in zip(chosen, seconds))
              + f"  (~{words / sum(seconds) * 60:.0f} words/min)")
        if custom:
            print(f"  plus one file per beat: {stem}-<beat>.wav")


def write(path, audio):
    audio = np.clip(audio, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype("<i2").tobytes())


def expand_beats(tokens, order):
    """'1.1-1.5 3.7' -> ['1.1', '1.2', '1.3', '1.4', '1.5', '3.7'] (ranges follow script order)."""
    out = []
    for tok in tokens:
        a, _, b = tok.partition("-") if tok.count("-") == 1 and tok not in order else (tok, "", "")
        for bid in (a, b or a):
            if bid not in order:
                sys.exit(f"Unknown beat ID: {bid}")
        i, j = order.index(a), order.index(b or a)
        out += order[min(i, j): max(i, j) + 1]
    return out


if __name__ == "__main__":
    main()

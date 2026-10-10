"""
Stage 1 of the video: narration, timeline, subtitles, soundtrack and the final audio mix.

  1. Reads the narration (not the [stage directions]) from STORY.md, beat by beat.
  2. Speaks each beat with the chosen TTS engine (video/tts.py: Kokoro by default, Piper as a
     fallback) -> one clip per beat, cached per engine/voice/text.
  3. Builds the video timeline FROM the narration: every beat lasts exactly as long as its
     clip plus padding (0.2 s before + 0.2 s after = ~0.4 s between beats); chapter cards
     add ~1.5 s between chapters; a title hold at the start and the end card at the end.
  4. Writes narration.wav (48 kHz), subtitles .srt and timeline.json (read by render.mjs).
  5. Generates the original soundtrack (music.py) to the same timeline.
  6. Mixes: music ducked ~12 dB under the voice (ffmpeg sidechaincompress), loudness
     normalised to -16 LUFS (two-pass loudnorm), 48 kHz stereo -> mix.wav.

Usage (from the repo root):
  pip install -r video/requirements.txt
  python3 video/build_audio.py                        # everything, Kokoro (default voice)
  python3 video/build_audio.py --voice am_michael     # another Kokoro voice
  python3 video/build_audio.py --tts piper            # Piper fallback (LibriTTS-high, speaker 228)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts import add_tts_args, make_engine, prepare_clip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "video-out")
AUDIO = os.path.join(OUT, "audio")
CLIPS = os.path.join(AUDIO, "clips")
SR = 48000

# Timing (seconds)
TITLE_HOLD = 3.5           # title on screen before anything moves (it also fades in from black)
INTRO_SCROLL = 2.5         # title fades out
BEAT_LEAD = 0.2            # silence before each beat's voice
BEAT_TAIL = 0.2            # silence after (lead + tail = ~0.4 s between beats)
CHAPTER_CARD = 1.5         # extra pause between chapters, while the chapter card shows
# Extra silence before a beat's voice, for musical moments (the verdict swell at 6.5).
EXTRA_LEAD = {"6.5": 2.0}
END_HOLD = 9.0             # end card
FADE_IN = 1.5              # from black at the very start
FADE_OUT = 2.0             # to black at the very end


# ------------------------------------------------------------------ story

def read_story(path):
    """Chapters and beats from STORY.md, narration only (stage directions removed)."""
    chapters, beat = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        m = re.match(r"^## Chapter (\d+): (.+)", line)
        if m:
            chapters.append({"number": int(m.group(1)), "title": m.group(2), "beats": []})
            beat = None
            continue
        m = re.match(r"^### (\S+)", line)
        if m:
            beat = {"id": m.group(1), "text": ""}
            chapters[-1]["beats"].append(beat)
            continue
        if beat is None or not line or line == "---":
            continue
        if line.startswith("[") and line.endswith("]"):
            continue  # stage direction
        beat["text"] = (beat["text"] + " " + line).strip()
    for c in chapters:
        for b in c["beats"]:
            b["text"] = b["text"].replace("*", "")  # markdown emphasis
    return chapters


def write_wav(path, x, channels=1):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(path) as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return x.reshape(-1, ch) if ch > 1 else x


# ------------------------------------------------------------------ timeline

def build_timeline(chapters, durations):
    """Segments in the same order and with the same ids as the site's Timeline (src/timeline.js)."""
    segs, t = [], TITLE_HOLD
    segs.append({"id": "intro", "type": "intro", "start": t, "dur": INTRO_SCROLL})
    t += INTRO_SCROLL
    for c in chapters:
        segs.append({"id": f"ch{c['number']}", "type": "chapter", "start": t, "dur": CHAPTER_CARD,
                     "title": f"Chapter {c['number']}: {c['title']}"})
        t += CHAPTER_CARD
        for b in c["beats"]:
            lead = BEAT_LEAD + EXTRA_LEAD.get(b["id"], 0)
            d = lead + durations[b["id"]] + BEAT_TAIL
            segs.append({"id": b["id"], "type": "beat", "start": t, "dur": d,
                         "voiceStart": t + lead, "voiceEnd": t + lead + durations[b["id"]],
                         "text": b["text"]})
            t += d
    segs.append({"id": "end", "type": "end", "start": t, "dur": END_HOLD})
    t += END_HOLD
    return {"fps": 30, "duration": t, "fadeIn": FADE_IN, "fadeOut": FADE_OUT, "segments": segs}


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, timeline):
    lines, n = [], 0
    for s in timeline["segments"]:
        if s["type"] != "beat":
            continue
        n += 1
        lines += [str(n), f"{srt_time(s['voiceStart'])} --> {srt_time(s['voiceEnd'] + BEAT_TAIL)}", s["text"], ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))


# ------------------------------------------------------------------ mix

def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def mix(narration_wav, music_wav, out_wav):
    """Duck the music under the voice, then two-pass loudness normalisation to -16 LUFS."""
    duck = ("[0:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[voice][key];"
            "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[music];"
            # Voice is normalised to about -20 dBFS RMS. Threshold -37 dBFS, ratio 6:1 and a fast
            # attack give roughly 12 dB of gain reduction on the music while someone is speaking
            # (measured by duck_report below).
            "[music][key]sidechaincompress=threshold=0.014:ratio=6:attack=20:release=600:knee=2:makeup=1[ducked];"
            "[voice][ducked]amix=inputs=2:duration=longest:normalize=0[mixed]")
    measure = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", narration_wav, "-i", music_wav, "-filter_complex",
         duck + ";[mixed]loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True).stderr
    stats = json.loads(measure[measure.rindex("{"): measure.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
          f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
          f":offset={stats['target_offset']}:linear=true")
    ffmpeg("-i", narration_wav, "-i", music_wav, "-filter_complex", duck + f";[mixed]{ln},aresample=48000[out]",
           "-map", "[out]", "-ar", "48000", "-ac", "2", out_wav)
    # Also keep the ducked music on its own, so the amount of ducking can be checked.
    ffmpeg("-i", narration_wav, "-i", music_wav, "-filter_complex", duck.replace("[voice][ducked]amix=inputs=2:duration=longest:normalize=0[mixed]", "[voice]anullsink"),
           "-map", "[ducked]", os.path.join(AUDIO, "music-ducked.wav"))


def duck_report(timeline):
    """Music level while the voice speaks vs. in the gaps (dB)."""
    m = read_wav(os.path.join(AUDIO, "music-ducked.wav")).mean(axis=1)
    raw = read_wav(os.path.join(AUDIO, "music.wav")).mean(axis=1)
    speak_mask = np.zeros(len(m), bool)
    for s in timeline["segments"]:
        if s["type"] == "beat":
            speak_mask[int((s["voiceStart"] + 0.5) * SR): int((s["voiceEnd"] - 0.2) * SR)] = True
    def ratio(mask):
        return 10 * np.log10(np.mean(m[:len(mask)][mask] ** 2) / np.mean(raw[:len(mask)][mask] ** 2))
    return ratio(speak_mask), ratio(~speak_mask)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    add_tts_args(ap)
    ap.add_argument("--mix-only", action="store_true", help="reuse narration/music, redo the mix")
    args = ap.parse_args()
    os.makedirs(CLIPS, exist_ok=True)

    chapters = read_story(os.path.join(ROOT, "STORY.md"))
    beats = [b for c in chapters for b in c["beats"]]
    print(f"{len(chapters)} chapters, {len(beats)} beats")

    # 1-2. One cached clip per beat (re-synthesised only if the text or voice changes).
    engine = None
    durations = {}
    voice_id = f"{args.tts}|{args.voice}|{args.speaker}|{args.speed}"
    for i, b in enumerate(beats):
        key = hashlib.sha1(f"{voice_id}|{b['text']}".encode()).hexdigest()[:10]
        path = os.path.join(CLIPS, f"{b['id']}-{key}.wav")
        if not os.path.exists(path):
            if engine is None:
                engine = make_engine(args)
                print(f"voice: {engine.name}")
            write_wav(path, prepare_clip(*engine.synth(b["text"]), SR))
            print(f"  spoke {b['id']:>5} ({i + 1}/{len(beats)})")
        b["clip"] = path
        durations[b["id"]] = len(read_wav(path)) / SR

    # 3-4. Timeline from the narration, narration track, subtitles.
    timeline = build_timeline(chapters, durations)
    total = timeline["duration"]
    narration = np.zeros(int(np.ceil(total * SR)) + SR, np.float32)
    seg_by_id = {s["id"]: s for s in timeline["segments"]}
    for b in beats:
        x = read_wav(b["clip"])
        i = int(round(seg_by_id[b["id"]]["voiceStart"] * SR))
        narration[i: i + len(x)] += x
    narration = narration[: int(np.ceil(total * SR))]
    write_wav(os.path.join(AUDIO, "narration.wav"), narration)
    json.dump(timeline, open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    write_srt(os.path.join(OUT, "neuron-story.srt"), timeline)
    print(f"timeline: {total:.1f} s ({total / 60:.1f} min); narration {sum(durations.values()) / 60:.1f} min")

    # 5. Soundtrack.
    if not (args.mix_only and os.path.exists(os.path.join(AUDIO, "music.wav"))):
        from music import render_music
        music = render_music(timeline, SR)
        write_wav(os.path.join(AUDIO, "music.wav"), music.reshape(-1), channels=2)
        print("music: done")

    # 6. Mix.
    mix(os.path.join(AUDIO, "narration.wav"), os.path.join(AUDIO, "music.wav"), os.path.join(AUDIO, "mix.wav"))
    during, gaps = duck_report(timeline)
    print(f"mix: done (music {during:+.1f} dB while speaking, {gaps:+.1f} dB in gaps)")


if __name__ == "__main__":
    main()

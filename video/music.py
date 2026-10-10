"""
Original ambient soundtrack, generated procedurally with numpy (no samples, no downloads).

Layers
  - Pad: chords from wavetable voices (two detuned voices per note) that morph between a
    dark and a bright waveform on slow LFOs, so the timbre keeps evolving.
  - Sub-bass drone: a soft sine (+ a little 2nd harmonic) on the section's root, gliding
    between roots.
  - Bells: sparse FM bell tones on chord tones, at seeded-random times.
  - Shared synthetic reverb (decaying stereo noise impulse response).

Mood follows the story: calm and mysterious (ch 1-2), curious and brighter (3-5), a build
into a swell and release at the verdict (6.5), tense and uncertain while the untrained
network flails (ch 7, with beating detune and a slow heartbeat pulse), rising through the
training time-lapse, and resolving warm at the end. Chords change every 10-16 s with
crossfades and re-voicings, so nothing loops obviously.
"""

import numpy as np
from scipy.signal import butter, oaconvolve, sosfilt

TABLE = 2048


def _midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def _table(harmonics, tilt):
    """One cycle of a band-limited wave: sum of harmonics with 1/k^tilt amplitudes."""
    ph = np.arange(TABLE) / TABLE
    w = sum(np.sin(2 * np.pi * k * ph) / k ** tilt for k in range(1, harmonics + 1))
    return (w / np.abs(w).max()).astype(np.float32)


DARK = _table(4, 2.2)
BRIGHT = _table(14, 1.15)

# Note names -> MIDI, octave 3 = middle register for pads.
N = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def chord(root, *intervals):
    return (N[root], intervals)


# Sections keyed by the timeline segment they start at.
# bright: timbre 0..1, bells: bell rate per second, tension: 0..1 (detune beating, heartbeat).
SECTIONS = [
    ("intro", dict(chords=[chord("D", 0, 7, 14, 15), chord("Bb", 0, 7, 11, 18)], bright=0.25, bells=0.07, tension=0.0, gain=0.9)),
    ("ch2", dict(chords=[chord("A", 0, 7, 10, 14), chord("F", 0, 7, 11, 16)], bright=0.32, bells=0.09, tension=0.0, gain=0.95)),
    ("ch3", dict(chords=[chord("C", 0, 7, 11, 14), chord("G", 0, 4, 9, 14), chord("A", 0, 7, 10, 15)], bright=0.55, bells=0.17, tension=0.0, gain=1.0)),
    ("ch4", dict(chords=[chord("F", 0, 7, 11, 14), chord("C", 0, 4, 11, 16), chord("D", 0, 7, 10, 14)], bright=0.6, bells=0.2, tension=0.0, gain=1.0)),
    ("ch5", dict(chords=[chord("Eb", 0, 7, 11, 14), chord("Bb", 0, 7, 14, 16), chord("G", 0, 7, 10, 15)], bright=0.6, bells=0.16, tension=0.05, gain=1.0)),
    ("ch6", dict(chords=[chord("Ab", 0, 7, 11, 16), chord("Bb", 0, 5, 7, 14)], bright=0.5, bells=0.12, tension=0.05, gain=1.0)),
    ("6.5", dict(chords=[chord("Eb", 0, 7, 12, 16, 19)], bright=0.9, bells=0.35, tension=0.0, gain=1.25)),
    ("6.6", dict(chords=[chord("C", 0, 7, 10, 14), chord("Ab", 0, 7, 11, 16)], bright=0.45, bells=0.1, tension=0.0, gain=1.0)),
    ("ch7", dict(chords=[chord("C", 0, 3, 7, 8), chord("D", 0, 3, 6, 9), chord("Ab", 0, 4, 7, 13)], bright=0.3, bells=0.05, tension=1.0, gain=0.95)),
    ("7.12", dict(chords=[chord("F", 0, 7, 10, 15), chord("Ab", 0, 7, 14, 16), chord("Bb", 0, 5, 7, 14)], bright=0.55, bells=0.15, tension=0.35, gain=1.05)),
    ("7.15", dict(chords=[chord("Eb", 0, 7, 11, 14, 16), chord("Ab", 0, 7, 11, 14)], bright=0.55, bells=0.1, tension=0.0, gain=1.0)),
    ("end", dict(chords=[chord("Eb", 0, 7, 11, 14, 19)], bright=0.45, bells=0.06, tension=0.0, gain=0.95)),
]


def _lowpass(x, sr, hz):
    return sosfilt(butter(2, hz, fs=sr, output="sos"), x, axis=0)


def _pad_chord(rng, root, intervals, dur, sr, bright, tension, t0):
    """One sustained chord (stereo) with slow attack/release and an evolving timbre."""
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros((n, 2), np.float32)
    base = 48 + root if root < 6 else 36 + root  # keep the voicing centred around C3-G3
    for j, iv in enumerate(intervals):
        note = base + iv + (12 if rng.random() < 0.25 and iv < 12 else 0)  # occasional re-voicing
        f = _midi(note)
        pan = 0.5 + 0.35 * np.sin(j * 2.1 + root)
        amp = 0.16 / (1 + 0.15 * j)
        # Timbre morph: slow, per-note LFO around the section brightness.
        morph = np.clip(bright + 0.18 * np.sin(2 * np.pi * (t + t0) / (19 + 4 * j) + j), 0, 1).astype(np.float32)
        for v, cents in enumerate((-6 - 10 * tension, 6 + 10 * tension)):
            # Tension adds slow, uneasy beating between the two voices.
            wobble = 1 + tension * 0.004 * np.sin(2 * np.pi * (0.13 + 0.05 * j) * (t + t0))
            freq = f * 2 ** (cents / 1200) * wobble
            phase = np.cumsum(freq / sr) + rng.random()
            idx = ((phase % 1.0) * TABLE).astype(np.int32)
            wave = DARK[idx] * (1 - morph) + BRIGHT[idx] * morph
            side = (pan if v else 1 - pan)
            out[:, 0] += amp * wave * (1 - side * 0.6)
            out[:, 1] += amp * wave * (0.4 + side * 0.6)
    return out


def _bell(rng, f, sr, dur=5.5):
    n = int(dur * sr)
    t = np.arange(n) / sr
    index = 2.4 * np.exp(-t / 0.5)
    y = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * 3.51 * t))
    y += 0.25 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.9)
    env = np.exp(-t / 2.2) * np.minimum(t / 0.004, 1)
    return (y * env * 0.09).astype(np.float32)


def _impulse(sr, seconds=3.4, seed=7):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-t / 0.85)[:, None]
    ir = _lowpass(ir, sr, 5200)
    ir[: int(0.012 * sr)] = 0  # small pre-delay
    return (ir / np.sqrt((ir ** 2).sum(axis=0))).astype(np.float32)


def render_music(timeline, sr=48000, seed=2026):
    rng = np.random.default_rng(seed)
    total = timeline["duration"]
    n = int(np.ceil(total * sr))
    starts = {s["id"]: s["start"] for s in timeline["segments"]}
    secs = [(0.0 if sid == "intro" else starts[sid], cfg) for sid, cfg in SECTIONS]
    secs.append((total + 6, None))

    pad = np.zeros((n, 2), np.float32)
    bells = np.zeros((n, 2), np.float32)
    drone = np.zeros(n, np.float32)
    XF = 4.0  # chord crossfade (s)

    for (s0, cfg), (s1, _) in zip(secs[:-1], secs[1:]):
        # Chords: alternate through the section's set every 10-16 s, overlapping by XF.
        t, k = s0, int(rng.integers(0, len(cfg["chords"])))
        while t < min(s1, total):
            seg = float(rng.uniform(10, 16))
            end = min(t + seg, s1 if s1 < total else total + 2)
            dur = end - t + XF
            root, ivs = cfg["chords"][k % len(cfg["chords"])]
            c = _pad_chord(rng, root, ivs, dur, sr, cfg["bright"], cfg["tension"], t)
            m = len(c)
            env = np.ones(m, np.float32)
            a = int(XF * sr)
            env[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
            env[-a:] = np.cos(np.linspace(0, np.pi / 2, a)) ** 2
            i0 = int(t * sr)
            i1 = min(i0 + m, n)
            pad[i0:i1] += (c[: i1 - i0] * env[: i1 - i0, None]) * cfg["gain"]
            # Drone on the chord root, two octaves down, crossfaded the same way.
            f = _midi(24 + 12 + root if root < 5 else 24 + root)
            tt = np.arange(i1 - i0) / sr + t
            breathe = 0.75 + 0.25 * np.sin(2 * np.pi * tt / 17)
            d = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * breathe * 0.22
            drone[i0:i1] += (d * env[: i1 - i0]).astype(np.float32)
            t, k = end, k + 1

        # Bells: Poisson times, chord tones in the upper octaves.
        bt = s0 + float(rng.exponential(1 / cfg["bells"]))
        while bt < min(s1, total - 3):
            root, ivs = cfg["chords"][int(rng.integers(0, len(cfg["chords"])))]
            note = 72 + root + int(rng.choice(ivs)) % 12 + (12 if rng.random() < 0.3 else 0)
            if cfg["tension"] > 0.5:
                note -= 12  # lower, darker bells while things are uncertain
            b = _bell(rng, _midi(note), sr)
            p = float(rng.uniform(0.2, 0.8))
            i0 = int(bt * sr)
            i1 = min(i0 + len(b), n)
            bells[i0:i1, 0] += b[: i1 - i0] * (1 - p)
            bells[i0:i1, 1] += b[: i1 - i0] * p
            bt += float(rng.exponential(1 / cfg["bells"]))

    # Tension: a slow, soft heartbeat under the untrained-network section of chapter 7.
    if "ch7" in starts and "7.12" in starts:
        hb0, hb1 = starts["ch7"] + 4, starts["7.12"]
        bt = hb0
        while bt < hb1:
            for off, a in ((0.0, 1.0), (0.28, 0.6)):
                i0 = int((bt + off) * sr)
                m = int(0.5 * sr)
                tt = np.arange(m) / sr
                thump = np.sin(2 * np.pi * 46 * tt) * np.exp(-tt / 0.11) * 0.22 * a
                fade = min((bt - hb0) / 6, (hb1 - bt) / 6, 1)
                drone[i0:i0 + m] += (thump * fade).astype(np.float32)[: max(0, n - i0)]
            bt += 1.7

    # Verdict (6.5): a filtered-noise riser into the beat, then the release.
    if "6.5" in starts:
        v = starts["6.5"]
        r0 = max(v - 5.0, 0)
        m = int((v + 1.0 - r0) * sr)
        tt = np.arange(m) / sr
        noise = rng.standard_normal((m, 2)).astype(np.float32)
        noise = _lowpass(noise, sr, 2400)
        env = (np.clip(tt / (v - r0), 0, 1) ** 3) * np.clip((v + 1.0 - r0 - tt) / 1.0, 0, 1)
        i0 = int(r0 * sr)
        pad[i0:i0 + m] += noise[: max(0, n - i0)] * env[:, None][: max(0, n - i0)] * 0.05

    # Verdict swell: +5 dB rising into the pause before 6.5's narration, then a slow release.
    if "6.5" in starts:
        v = starts["6.5"]
        pts_t = np.array([v - 6.0, v + 1.7, v + 4.0, v + 12.0])
        pts_db = np.array([0.0, 5.0, 3.0, 0.0])
        tt = np.arange(n) / sr
        g = 10 ** (np.interp(tt, pts_t, pts_db) / 20)
        pad *= g[:, None].astype(np.float32)
        bells *= g[:, None].astype(np.float32)

    # Reverb on pad + bells; drone stays dry (and centred) to keep the low end clean.
    wet_in = pad + bells * 1.4
    ir = _impulse(sr)
    wet = np.stack([oaconvolve(wet_in[:, c], ir[:, c])[:n] for c in range(2)], axis=1).astype(np.float32)
    music = 0.7 * (pad + bells) + 0.55 * wet + drone[:, None] * 0.9

    # Gentle fade in at the start and out at the end.
    fi = int(3.0 * sr)
    fo = int((timeline["fadeOut"] + 4.0) * sr)
    music[:fi] *= np.linspace(0, 1, fi)[:, None] ** 2
    music[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2

    # Level: about -24 dBFS RMS before the mix (narration sits at about -20).
    rms = np.sqrt(np.mean(music ** 2))
    music *= 10 ** (-24 / 20) / max(rms, 1e-6)
    peak = np.abs(music).max()
    if peak > 0.95:
        music *= 0.95 / peak
    return music.astype(np.float32)

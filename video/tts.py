"""
Text-to-speech engines for the narration, behind one small interface:

    engine = make_engine(args)          # from --tts / --voice / --speaker / --speed / --kokoro-dir
    x, sr = engine.synth(text)          # float32 mono audio at the engine's native rate
    engine.id                           # stable string for the clip cache

  kokoro (default)  Kokoro-82M (hexgrad), Apache 2.0, via the `kokoro` pip package. CPU is fine.
                    Downloads its weights from Hugging Face on first use, or reads them from
                    --kokoro-dir (config.json, kokoro-v1_0.pth, voices/<name>.pt).
  piper (fallback)  Piper with the LibriTTS-high voice used for the Stage 1 test.

Shared by both: spoken-only pronunciation fixes, and prepare_clip() (trim, resample to
48 kHz, level to about -20 dBFS RMS) so the timeline logic doesn't care which engine spoke.
"""

import io
import os
import re
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "video-out")

# Spoken-only fixes (subtitles keep the original text). Checked against Kokoro's phonemizer:
# "784", "26,000" and "softmax" already read correctly; "ReLU" would be spelled out.
PRONOUNCE = {r"\bReLU\b": "ray-loo"}

# Kokoro voices shortlisted as calm, clear narrators (see video/README.md).
KOKORO_VOICES = {
    "af_heart": "US English, female, warm and even; the highest-rated Kokoro voice",
    "am_michael": "US English, male, mid-low pitch, steady",
    "bf_emma": "British English (southern), female, clear and measured",
}
KOKORO_DEFAULT = "af_heart"
KOKORO_SPEED = 0.92  # a little slower than Kokoro's default, for a calm narration pace

PIPER_DEFAULT = os.path.join(OUT, "voices", "en-us-libritts-high", "en-us-libritts-high.onnx")
PIPER_SPEAKER = 228   # LibriTTS reader 5876
PIPER_LENGTH = 1.06


def spoken_text(text):
    for pat, rep in PRONOUNCE.items():
        text = re.sub(pat, rep, text)
    return text


def prepare_clip(x, sr, out_sr=48000):
    """Trim leading/trailing near-silence, resample, and level to about -20 dBFS RMS."""
    idx = np.where(np.abs(x) > 0.01)[0]
    if len(idx):
        x = x[max(idx[0] - int(0.03 * sr), 0): idx[-1] + int(0.08 * sr)]
    x = resample_poly(x, out_sr, sr).astype(np.float32)
    voiced = x[np.abs(x) > 0.02]
    rms = np.sqrt(np.mean(voiced ** 2)) if len(voiced) else 0.1
    return (x * (10 ** (-20 / 20) / rms)).astype(np.float32)


class PiperTTS:
    def __init__(self, model=PIPER_DEFAULT, speaker=PIPER_SPEAKER, length_scale=PIPER_LENGTH):
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError:
            sys.exit("Piper is not installed: pip install -r video/requirements.txt")
        if not os.path.exists(model):
            sys.exit(f"Piper voice not found: {model}\nSee video/README.md for where to download it.")
        self.voice = PiperVoice.load(model)
        self.config = SynthesisConfig(speaker_id=speaker, length_scale=length_scale, noise_scale=0.5, noise_w_scale=0.6)
        self.id = f"piper|{os.path.basename(model)}|{speaker}|{length_scale}"
        self.name = f"piper-{os.path.basename(model).replace('.onnx', '')}-{speaker}"

    def synth(self, text):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(spoken_text(text), w, syn_config=self.config)
        buf.seek(0)
        with wave.open(buf) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x, sr


class KokoroTTS:
    SR = 24000
    REPO = "hexgrad/Kokoro-82M"

    def __init__(self, voice=KOKORO_DEFAULT, speed=KOKORO_SPEED, local_dir=None):
        try:
            from kokoro import KModel, KPipeline
        except ImportError:
            sys.exit("Kokoro is not installed: pip install -r video/requirements.txt  (or use --tts piper)")
        self.speed = speed
        # Voice names start with the accent: a = American English, b = British English.
        lang = voice[0] if voice[0] in "ab" else "a"
        try:
            if local_dir:
                model = KModel(repo_id=self.REPO, config=os.path.join(local_dir, "config.json"),
                               model=os.path.join(local_dir, "kokoro-v1_0.pth")).eval()
                self.voice = os.path.join(local_dir, "voices", f"{voice}.pt")
                if not os.path.exists(self.voice):
                    sys.exit(f"Kokoro voice file not found: {self.voice}")
            else:
                model = KModel(repo_id=self.REPO).eval()
                self.voice = voice
            self.pipeline = KPipeline(lang_code=lang, repo_id=self.REPO, model=model)
            self.pipeline.load_voice(self.voice)  # fail now (not mid-run) if it can't be fetched
        except SystemExit:
            raise
        except Exception as err:  # most often: Hugging Face unreachable
            sys.exit(
                f"Kokoro could not load its model or voice '{voice}': {type(err).__name__}: {err}\n"
                f"It downloads {self.REPO} from huggingface.co on first use. Either allow that host,\n"
                "pre-download the files and pass --kokoro-dir (see video/README.md), or use --tts piper.")
        self.id = f"kokoro|{voice}|{speed}"
        self.name = f"kokoro-{voice}"

    def synth(self, text):
        chunks = [r.audio.numpy() for r in self.pipeline(spoken_text(text), voice=self.voice, speed=self.speed)
                  if r.audio is not None]
        return np.concatenate(chunks).astype(np.float32), self.SR


def add_tts_args(ap):
    ap.add_argument("--tts", choices=["kokoro", "piper"], default="kokoro", help="speech engine (default: kokoro)")
    ap.add_argument("--voice", default=None,
                    help=f"kokoro: voice name (default {KOKORO_DEFAULT}); piper: path to an .onnx model")
    ap.add_argument("--speaker", type=int, default=PIPER_SPEAKER, help="piper only: speaker index")
    ap.add_argument("--speed", type=float, default=KOKORO_SPEED, help="kokoro only: speaking speed")
    ap.add_argument("--kokoro-dir", default=os.environ.get("KOKORO_DIR"),
                    help="kokoro only: local folder with config.json, kokoro-v1_0.pth and voices/")


def make_engine(args, voice=None):
    voice = voice or args.voice
    if args.tts == "piper":
        return PiperTTS(model=voice or PIPER_DEFAULT, speaker=args.speaker)
    return KokoroTTS(voice=voice or KOKORO_DEFAULT, speed=args.speed, local_dir=args.kokoro_dir)

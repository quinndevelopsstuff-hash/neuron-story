# Neuron Story → MP4

Turns the site into a narrated video with an original soundtrack. Nothing is screen-recorded:
the site has a capture mode (`?capture`) that draws one exact frame on request, and the frames
are piped straight into ffmpeg. All output goes to `video-out/` (git-ignored).

## Requirements

- Node 20+ and the repo's npm dependencies (`npm install`)
- Python 3.10+ with `pip install -r video/requirements.txt` (Kokoro, Piper, numpy, scipy), see below
- ffmpeg 6+ on your PATH
- A Chromium for Playwright: `npx playwright install chromium`
  (or point `CHROME_PATH` at an existing Chrome/Chromium)

### Python environment

Use a virtual environment with a current pip/setuptools (an old system setuptools can fail to
build one of Kokoro's dependencies, `docopt`):

```sh
python3 -m venv video-out/.venv
source video-out/.venv/bin/activate          # Windows: video-out\.venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r video/requirements.txt
```

Optional, used by Kokoro for words missing from its dictionary: `sudo apt-get install espeak-ng`
(Linux) or `brew install espeak-ng` (macOS).

### The voice (TTS engine is a flag)

`--tts kokoro` (default) or `--tts piper` (fallback), on both `build_audio.py` and
`voice_samples.py`.

**Kokoro** (Kokoro-82M, Apache 2.0, CPU is fine). On first use it downloads about 330 MB from
Hugging Face (`hexgrad/Kokoro-82M`). Shortlisted narrator voices, all spoken at speed 0.92:

| Voice | Sounds like |
| --- | --- |
| `af_heart` (default) | US English, female, warm and even; the highest-rated Kokoro voice |
| `am_michael` | US English, male, mid-low pitch, steady |
| `bf_emma` | British English (southern), female, clear and measured |

Audition them first (writes one file per voice to `video-out/voice-samples/`, each speaking
beats 1.2, 3.7 and 6.8):

```sh
python video/voice_samples.py                       # the three voices above
python video/voice_samples.py --voices bm_george    # any other Kokoro voice
python video/voice_samples.py --tts piper           # the Piper fallback, for comparison
```

No network on the render machine? Download the files once and point at them:

```sh
pip install -U huggingface_hub
hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth \
  voices/af_heart.pt voices/am_michael.pt voices/bf_emma.pt --local-dir video-out/kokoro
python video/voice_samples.py --kokoro-dir video-out/kokoro      # or set KOKORO_DIR
```

(Older `huggingface_hub` versions call the command `huggingface-cli download`.)

**In Google Colab** (free CPU runtime is enough for the samples):

```python
!git clone https://github.com/quinndevelopsstuff-hash/neuron-story.git
%cd neuron-story
!git checkout claude/friendly-turing-3034k1
!apt-get -qq install -y espeak-ng
!pip install -q kokoro soundfile numpy scipy
!python video/voice_samples.py
from IPython.display import Audio, display
import glob
for f in sorted(glob.glob('video-out/voice-samples/*.wav')):
    print(f); display(Audio(f))
```

(If the repository is private, clone with a GitHub token or upload the repo folder.)

**Piper** (fallback): voice `en-us-libritts-high`, speaker index 228 (LibriTTS reader 5876),
from the official Piper GitHub release:

```sh
mkdir -p video-out/voices && cd video-out/voices
curl -LO https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz
mkdir en-us-libritts-high && tar -xzf voice-en-us-libritts-high.tar.gz -C en-us-libritts-high
cd ../..
```

Clips are cached per engine, voice and text, so switching voice only re-speaks what changed.

## Steps

```sh
# 0. Pick a narrator (see above)
python video/voice_samples.py

# 1. Audio: narration clips, timeline, subtitles, soundtrack, mix (about 5 min).
#    --script video speaks video/VIDEO_SCRIPT.md (written for the ear) instead of STORY.md.
python video/build_audio.py --script video --voice af_heart      # or --tts piper

# 2. Optional: a 30-second test with audio -> video-out/test.mp4
node video/render.mjs --test 30 --gpu

# 3. Optional: chapter 1 only, with its slice of the final audio, to check voice, music
#    and sync -> video-out/chapter1-preview.mp4. The full render reuses this chunk.
node video/render.mjs --preview --gpu

# 4. Optional: estimate the full render time on this machine
node video/render.mjs --bench 8 --gpu

# 5. Full render -> video-out/neuron-story.mp4 (+ video-out/neuron-story.srt)
node video/render.mjs --gpu
```

Leave out `--gpu` on a machine without one (software WebGL, much slower). The script prints
the WebGL renderer it got, so you can confirm the GPU is in use.

**Resuming:** the full render is split into one chunk per chapter (`video-out/chunks/`).
Finished chunks are kept, so if the render stops, run the same command again and it
continues with the next chapter. A chunk is reused only if it was rendered from the same
`timeline.json` and frame settings, so after rebuilding the audio (for example with a new voice)
the affected chapters render again automatically. Delete `video-out/chunks/` to start over,
which you should also do after changing the site's code.

**Chapter 1 preview:** `--preview` renders just the first chunk (title, intro and chapter 1, up
to where chapter 2's card begins) and muxes the same time range of the final mix with a 0.5 s
fade in and 1 s fade out. The picture cuts at the end rather than fading.

## How it fits together

| File | Role |
| --- | --- |
| `video/VIDEO_SCRIPT.md` | Video-only narration: same chapters, beat IDs and stage directions as STORY.md, rewritten for the ear (`--script video`). In the video, the on-screen text and subtitles use it too |
| `video/tts.py` | TTS engines behind one interface (`--tts kokoro` default, `--tts piper`), pronunciation fixes, clip levelling |
| `video/voice_samples.py` | Voice audition: beats 1.2, 3.7, 6.8 per voice → `video-out/voice-samples/` |
| `video/build_audio.py` | STORY.md → per-beat TTS clips → timeline built from the narration → `narration.wav`, `timeline.json`, `.srt`; calls `music.py`; ducks and loudness-normalises the mix |
| `video/music.py` | Original procedural soundtrack (pad, sub-bass drone, FM bells, reverb), mood per chapter |
| `video/render.mjs` | Builds and serves the site, maps each frame to story progress, captures frames into ffmpeg in resumable chunks, muxes the audio |
| `src/main.js` (`?capture`) | Capture mode: `window.__renderAt(p, time, fade)` draws one frame; no adaptive resolution, DPR 1, no interactive chrome |

**Timing:** each beat's scroll range is mapped linearly onto its audio span (0.2 s silence +
the clip + 0.2 s silence); chapter cards add 1.5 s; the title holds for 3.5 s (fading in from
black) and the end card for 9 s. The same smoothing as the live site is applied (read slightly
ahead to cancel its lag), so camera moves ease rather than jump at beat boundaries.

**Audio:** narration clips are levelled to about −20 dBFS RMS; the music is ducked about
12 dB while the voice speaks (`sidechaincompress`), then the mix is normalised to −16 LUFS
(two-pass `loudnorm`), 48 kHz stereo. Video: H.264 yuv420p CRF 18, AAC 192 kbps, +faststart.

## Credits

- Narration voice (default): Kokoro-82M by hexgrad, via the `kokoro` package. Licence: Apache
  License 2.0 (model weights and code).
- Fallback voice: Piper (Open Home Foundation / Rhasspy) with the LibriTTS-trained
  `en-us-libritts-high` model. LibriTTS (Zen et al., 2019) is CC BY 4.0, derived from
  LibriVox public-domain audiobook recordings.
- Music: generated procedurally by `video/music.py` for this project.

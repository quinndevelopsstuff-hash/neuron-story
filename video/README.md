# Neuron Story → MP4

Turns the site into a narrated video with an original soundtrack. Nothing is screen-recorded:
the site has a capture mode (`?capture`) that draws one exact frame on request, and the frames
are piped straight into ffmpeg. All output goes to `video-out/` (git-ignored).

## Requirements

- Node 20+ and the repo's npm dependencies (`npm install`)
- Python 3.10+ with `pip install -r video/requirements.txt` (Piper TTS, numpy, scipy)
- ffmpeg 6+ on your PATH
- A Chromium for Playwright: `npx playwright install chromium`
  (or point `CHROME_PATH` at an existing Chrome/Chromium)

### The voice

Piper voice **en-us-libritts-high**, speaker index **228** (LibriTTS reader 5876), from the
official Piper GitHub release. It was chosen for licence (CC BY 4.0, usable in a published
video with attribution) and by measurement: cleanest recording, clear voicing, calm pace and
natural pitch movement among 76 speakers tested.

```sh
mkdir -p video-out/voices && cd video-out/voices
curl -LO https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz
mkdir en-us-libritts-high && tar -xzf voice-en-us-libritts-high.tar.gz -C en-us-libritts-high
cd ../..
```

To try another speaker: `python3 video/build_audio.py --speaker 144` (clips are cached per
voice/speaker/text, so only changed beats are re-spoken).

## Steps

```sh
# 1. Audio: narration clips, timeline, subtitles, soundtrack, mix (about 5 min)
python3 video/build_audio.py

# 2. Optional: a 30-second test with audio -> video-out/test.mp4
node video/render.mjs --test 30 --gpu

# 3. Optional: estimate the full render time on this machine
node video/render.mjs --bench 8 --gpu

# 4. Full render -> video-out/neuron-story.mp4 (+ video-out/neuron-story.srt)
node video/render.mjs --gpu
```

Leave out `--gpu` on a machine without one (software WebGL, much slower). The script prints
the WebGL renderer it got, so you can confirm the GPU is in use.

**Resuming:** the full render is split into one chunk per chapter (`video-out/chunks/`).
Finished chunks are kept, so if the render stops, run the same command again and it
continues with the next chapter. Delete `video-out/chunks/` to start over.

## How it fits together

| File | Role |
| --- | --- |
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

- Narration voice: Piper (Open Home Foundation / Rhasspy) with the LibriTTS-trained
  `en-us-libritts-high` model. LibriTTS (Zen et al., 2019) is CC BY 4.0, derived from
  LibriVox public-domain audiobook recordings.
- Music: generated procedurally by `video/music.py` for this project.

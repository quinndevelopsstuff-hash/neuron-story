# Inside the Machine

A scroll-driven 3D flight through a real neural network (784 → 32 → 16 → 10, trained on MNIST) as it recognizes a handwritten 7. Built with Vite, vanilla JavaScript and Three.js.

## Run locally

```sh
npm install
npm run dev        # http://localhost:5173
```

Click **Auto-play** on the title screen for a hands-free ~10-minute run (space toggles, 1x/1.5x/2x speeds; any manual scroll pauses it).

Append `?stage` to the URL to show each beat's stage direction under the narration (useful when reviewing STORY.md).

## Build and preview

```sh
npm run build      # outputs dist/
npm run preview    # serves dist/ at http://localhost:4173
```

## Deploy to Vercel

```sh
npm i -g vercel
vercel             # first run links the project (preview deploy)
vercel --prod      # production deploy
```

Or import the GitHub repo in the Vercel dashboard. `vercel.json` sets the Vite framework, `npm run build`, and `dist/`.

## How it fits together

| File | Role |
| --- | --- |
| `STORY.md` | Narration + stage directions (source of truth) |
| `scripts/story-to-json.mjs` | Generates `src/data/story.json` from STORY.md (runs before dev/build) |
| `src/timeline.js` | Maps beats to scroll progress, weighted by reading length |
| `src/scroll.js` | Smoothed scroll progress + CatmullRom camera rig |
| `src/director.js` | Choreography: camera keyframes, chapter moods, per-beat scene tracks |
| `src/scene.js` | Renderer, camera, starfield, fog, bloom |
| `src/network.js` | Loads real weights, forward pass, neuron/connection rendering |
| `src/props.js` | Beat-specific props (ghost sevens, bias dial, label tag, loss readout…) |
| `src/autoplay.js` | Hands-free mode: constant-speed native scroll + play/pause/speed controls |
| `src/overlay.js` | Narration, chapter cards, progress bar, intro/end cards |
| `training/train.py` | Trains the network (NumPy) and exports `public/data/network.{json,bin}` |

## Retrain the network

```sh
pip install numpy
sh training/fetch_mnist.sh
python3 training/train.py
```

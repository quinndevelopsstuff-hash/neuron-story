/**
 * Stage 2 of the video: deterministic frame-by-frame render of the site, then muxing.
 *
 * Not a screen recording. The site's capture mode (?capture) exposes
 * window.__renderAt(p, time, fade), which draws exactly one frame. For every video frame
 * this script computes the story progress p from the narration timeline (built by
 * build_audio.py), asks the page to draw it, screenshots it, and pipes the JPEG straight
 * into ffmpeg. No frame files touch the disk.
 *
 * The video is rendered in chunks (one per chapter) into video-out/chunks/. Finished chunks
 * are skipped on the next run, so an interrupted render resumes where it stopped. A chunk is
 * only reused if it was rendered from the same timeline and frame settings (fingerprint in its
 * .done file), so rebuilding the audio with a new voice re-renders what changed.
 *
 * Usage (from the repo root, after `python3 video/build_audio.py`):
 *   node video/render.mjs --test 30     # first 30 s, with audio -> video-out/test.mp4
 *   node video/render.mjs               # full video -> video-out/neuron-story.mp4
 *   node video/render.mjs --gpu         # use the GPU (recommended on your own computer)
 *   node video/render.mjs --bench 8     # time 8 frames per chapter, estimate the full render
 *   node video/render.mjs --preview     # chapter 1 only, with its slice of the final audio
 *                                       #   -> video-out/chapter1-preview.mp4 (chunk reused by the full render)
 * Options: --port 4179, --quality 95 (JPEG quality of captured frames),
 *          CHROME_PATH=/path/to/chrome to use a specific Chromium build.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { Timeline } from '../src/timeline.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'video-out');
const CHUNKS = join(OUT, 'chunks');
const W = 1920;
const H = 1080;

/* ------------------------------------------------------------------ options */

const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const v = argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
};
const TEST = opt('test', false) === false ? 0 : Number(opt('test')) || 30;
const BENCH = opt('bench', false) === false ? 0 : Number(opt('bench')) || 8;
const PREVIEW = opt('preview', false) === true; // chapter 1 only
const GPU = opt('gpu', false) === true;
const PORT = Number(opt('port', 4179));
const QUALITY = Number(opt('quality', 95));

/* ------------------------------------------------------------------ timeline -> progress */

const timelinePath = join(OUT, 'timeline.json');
if (!existsSync(timelinePath)) {
  console.error('video-out/timeline.json not found. Run: python3 video/build_audio.py');
  process.exit(1);
}
const video = JSON.parse(readFileSync(timelinePath, 'utf8'));
const story = JSON.parse(readFileSync(join(ROOT, 'src/data/story.json'), 'utf8'));
const tl = new Timeline(story); // the site's own scroll layout: segment id -> progress range

const FPS = video.fps;
const TOTAL_FRAMES = Math.ceil(video.duration * FPS);
const segs = video.segments.map((s) => {
  const seg = tl.get(s.id); // throws if STORY.md and timeline.json are out of sync
  return { ...s, p0: seg.start, p1: seg.end };
});

/** Story progress at video time t: each segment's scroll range mapped linearly onto its audio span. */
function targetProgress(t) {
  if (t <= segs[0].start) return 0;
  for (const s of segs) {
    if (t < s.start + s.dur) return s.p0 + ((t - s.start) / s.dur) * (s.p1 - s.p0);
  }
  return 1;
}

/*
 * Same exponential smoothing as the live site (ScrollController, rate 3.2/s), so camera
 * moves ease across segment boundaries instead of changing speed abruptly. The target is
 * read ~0.3 s ahead to cancel the smoothing lag, keeping text and voice in sync.
 * Precomputed for every frame, so any chunk can be rendered independently.
 */
const RATE = 3.2;
const LEAD = 1 / RATE;
const progress = new Float64Array(TOTAL_FRAMES);
{
  const k = 1 - Math.exp(-RATE / FPS);
  let p = targetProgress(LEAD);
  for (let f = 0; f < TOTAL_FRAMES; f++) {
    p += (targetProgress(f / FPS + LEAD) - p) * k;
    progress[f] = Math.min(Math.max(p, 0), 1);
  }
}

/** Black overlay: fade in from black at the start, out to black at the end. */
function fadeAt(t) {
  if (t < video.fadeIn) return 1 - t / video.fadeIn;
  const out0 = video.duration - video.fadeOut;
  return t > out0 ? Math.min((t - out0) / video.fadeOut, 1) : 0;
}

/* ------------------------------------------------------------------ chunks */

/** Identifies what a chunk was rendered from; a chunk is reused only if this matches. */
const FINGERPRINT = createHash('sha1')
  .update(JSON.stringify({ video, W, H, FPS, QUALITY }))
  .digest('hex')
  .slice(0, 16);

function chunkDone(c) {
  const marker = join(CHUNKS, `${c.name}.done`);
  if (!existsSync(marker) || !existsSync(join(CHUNKS, `${c.name}.mp4`))) return false;
  try {
    const m = JSON.parse(readFileSync(marker, 'utf8'));
    return m.f0 === c.f0 && m.f1 === c.f1 && m.fingerprint === FINGERPRINT;
  } catch {
    return false;
  }
}

function chunkPlan() {
  if (BENCH) {
    // A few frames from the middle of every chapter (the heavy scenes differ a lot).
    return segs.filter((s) => s.type === 'chapter').map((s, i, arr) => {
      const next = arr[i + 1] ? arr[i + 1].start : video.duration;
      const mid = Math.round(((s.start + next) / 2) * FPS);
      return { name: s.id, f0: mid, f1: mid + BENCH, until: Math.round(next * FPS), from: Math.round(s.start * FPS) };
    });
  }
  if (TEST) return [{ name: 'test', f0: 0, f1: Math.min(Math.round(TEST * FPS), TOTAL_FRAMES) }];
  const cuts = segs.filter((s) => s.type === 'chapter' && s.id !== 'ch1').map((s) => Math.round(s.start * FPS));
  const bounds = [0, ...cuts, TOTAL_FRAMES];
  return bounds.slice(0, -1).map((f0, i) => ({ name: `chunk-${String(i + 1).padStart(2, '0')}`, f0, f1: bounds[i + 1] }));
}

/* ------------------------------------------------------------------ helpers */

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: 'inherit', cwd: ROOT, ...opts });
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')} failed (${r.status})`);
}

function fmt(s) {
  if (!Number.isFinite(s)) return '?';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m${String(Math.round(s % 60)).padStart(2, '0')}s`;
}

async function waitForServer(url, ms = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try {
      if ((await fetch(url)).ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Server did not start at ${url}`);
}

/* ------------------------------------------------------------------ main */

async function main() {
  mkdirSync(CHUNKS, { recursive: true });
  const plan = chunkPlan();
  // --preview: only the first chunk (chapter 1), rendered exactly as the full render would.
  const wanted = PREVIEW ? plan.slice(0, 1) : plan;
  const todo = wanted.filter((c) => TEST || BENCH || !chunkDone(c));
  const framesTodo = todo.reduce((n, c) => n + c.f1 - c.f0, 0);
  console.log(`Video ${video.duration.toFixed(1)} s, ${TOTAL_FRAMES} frames at ${FPS} fps, ${W}x${H}.`);
  console.log(BENCH ? `Benchmark: ${BENCH} frames per chapter.` : TEST ? `Test render: first ${TEST} s.`
    : PREVIEW ? `Chapter 1 preview: ${todo.length ? `${framesTodo} frames to render` : 'chunk-01 already rendered, reusing it'}.`
      : `${plan.length} chunks, ${plan.length - todo.length} already done, ${framesTodo} frames to render.`);

  if (todo.length) {
    // Build the site and serve the production bundle.
    run('npm', ['run', 'build', '--silent']);
    const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { cwd: ROOT, stdio: 'ignore' });
    const stop = () => server.kill();
    process.on('exit', stop);
    await waitForServer(`http://localhost:${PORT}/`);

    const args = GPU
      ? ['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--use-angle=default']
      : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
    const browser = await chromium.launch({
      headless: true,
      // Full Chromium in new-headless mode can use the GPU; the headless shell cannot.
      channel: GPU && !process.env.CHROME_PATH ? 'chromium' : undefined,
      executablePath: process.env.CHROME_PATH || undefined,
      args,
    });
    const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1, reducedMotion: 'no-preference' });
    const problems = [];
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') problems.push(`[${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => problems.push(`[pageerror] ${e.message}`));

    await page.goto(`http://localhost:${PORT}/?capture`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.__captureReady, null, { timeout: 120000 });
    await page.evaluate(() => window.__captureReady);
    const renderer = await page.evaluate(() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
      return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown';
    });
    console.log(`WebGL renderer: ${renderer}`);
    await page.evaluate((secs) => window.__setSegmentSeconds(secs), Object.fromEntries(segs.map((s) => [s.id, s.dur])));

    if (BENCH) {
      // Time a few frames per chapter (no video written) and estimate the full render.
      let estimate = 0;
      for (const c of todo) {
        await page.evaluate(([p, time]) => window.__renderAt(p, time, 0), [progress[c.f0], c.f0 / FPS]); // warm-up
        await page.screenshot({ type: 'jpeg', quality: QUALITY });
        const b0 = Date.now();
        for (let f = c.f0; f < c.f1; f++) {
          await page.evaluate(([p, time]) => window.__renderAt(p, time, 0), [progress[f], f / FPS]);
          await page.screenshot({ type: 'jpeg', quality: QUALITY });
        }
        const spf = (Date.now() - b0) / 1000 / (c.f1 - c.f0);
        const frames = c.until - c.from;
        estimate += spf * frames;
        console.log(`  ${c.name.padEnd(4)} ${spf.toFixed(2)} s/frame x ${frames} frames = ${fmt(spf * frames)}`);
      }
      console.log(`Estimated full render: ${fmt(estimate)} for ${TOTAL_FRAMES} frames (${(estimate / TOTAL_FRAMES).toFixed(2)} s/frame average).`);
      writeFileSync(join(OUT, 'bench.json'), JSON.stringify({ estimateSeconds: estimate, renderer, gpu: GPU }, null, 1));
      if (problems.length) for (const p of [...new Set(problems)].slice(0, 20)) console.log('  ' + p);
      await browser.close();
      stop();
      return;
    }

    const t0 = Date.now();
    let done = 0;
    for (const c of todo) {
      const target = TEST ? join(OUT, 'test-video.mp4') : join(CHUNKS, `${c.name}.mp4`);
      const tmp = `${target}.part.mp4`;
      const ff = spawn('ffmpeg', [
        '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-r', String(FPS), '-movflags', '+faststart', tmp,
      ], { stdio: ['pipe', 'inherit', 'inherit'] });
      const ffDone = new Promise((res, rej) => ff.on('close', (code) => (code === 0 ? res() : rej(new Error(`ffmpeg exited ${code}`)))));

      console.log(`\n${c.name}: frames ${c.f0}-${c.f1 - 1} (${((c.f1 - c.f0) / FPS).toFixed(1)} s of video)`);
      for (let f = c.f0; f < c.f1; f++) {
        const t = f / FPS;
        await page.evaluate(([p, time, fade]) => window.__renderAt(p, time, fade), [progress[f], t, fadeAt(t)]);
        const jpg = await page.screenshot({ type: 'jpeg', quality: QUALITY });
        if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r));
        done++;
        if (done % 30 === 0 || f === c.f1 - 1) {
          const spf = (Date.now() - t0) / 1000 / done;
          process.stdout.write(`\r  ${done}/${framesTodo} frames  ${spf.toFixed(2)} s/frame  elapsed ${fmt(spf * done)}  ETA ${fmt(spf * (framesTodo - done))}   `);
        }
      }
      ff.stdin.end();
      await ffDone;
      renameSync(tmp, target);
      if (!TEST) writeFileSync(join(CHUNKS, `${c.name}.done`), JSON.stringify({ f0: c.f0, f1: c.f1, fingerprint: FINGERPRINT, when: new Date().toISOString() }));
    }
    const spf = (Date.now() - t0) / 1000 / Math.max(done, 1);
    console.log(`\nRendered ${done} frames in ${fmt(spf * done)} (${spf.toFixed(2)} s/frame).`);
    writeFileSync(join(OUT, 'render-stats.json'), JSON.stringify({ frames: done, secondsPerFrame: spf, renderer, gpu: GPU, totalFrames: TOTAL_FRAMES }, null, 1));
    if (problems.length) {
      console.log(`Console warnings/errors during render (${problems.length}):`);
      for (const p of [...new Set(problems)].slice(0, 20)) console.log('  ' + p);
    } else {
      console.log('No console errors or warnings.');
    }
    await browser.close();
    stop();
  }

  // Mux: video + mixed audio (AAC 192k, 48 kHz), +faststart.
  const mix = join(OUT, 'audio', 'mix.wav');
  if (TEST) {
    run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', join(OUT, 'test-video.mp4'), '-i', mix,
      '-t', String(TEST), '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
      '-movflags', '+faststart', join(OUT, 'test.mp4')]);
    rmSync(join(OUT, 'test-video.mp4'));
    console.log('Wrote video-out/test.mp4');
    return;
  }
  if (PREVIEW) {
    // Chapter 1's chunk + the same time range cut from the final mix, with short audio fades.
    // The chunk starts at 0 (title and intro included) and ends where chapter 2's card begins.
    const c = plan[0];
    const start = c.f0 / FPS;
    const dur = (c.f1 - c.f0) / FPS;
    const fadeOut = Math.min(1.0, dur / 4);
    run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', join(CHUNKS, `${c.name}.mp4`),
      '-ss', start.toFixed(3), '-t', dur.toFixed(3), '-i', mix,
      '-af', `afade=t=in:st=0:d=0.5,afade=t=out:st=${(dur - fadeOut).toFixed(3)}:d=${fadeOut.toFixed(3)}`,
      '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
      '-t', dur.toFixed(3), '-movflags', '+faststart', join(OUT, 'chapter1-preview.mp4')]);
    console.log(`Wrote video-out/chapter1-preview.mp4 (${start.toFixed(1)}-${(start + dur).toFixed(1)} s of the timeline).`);
    console.log('The full render (node video/render.mjs) will reuse this chapter 1 chunk.');
    return;
  }
  const list = join(CHUNKS, 'list.txt');
  writeFileSync(list, plan.map((c) => `file '${c.name}.mp4'`).join('\n') + '\n');
  run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-i', mix,
    '-t', video.duration.toFixed(3), '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
    '-ar', '48000', '-movflags', '+faststart', join(OUT, 'neuron-story.mp4')]);
  console.log('Wrote video-out/neuron-story.mp4 (subtitles: video-out/neuron-story.srt)');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

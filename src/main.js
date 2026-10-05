/**
 * Entry point: loads the story and the trained network, builds the scene and runs the
 * render loop. Each frame: smoothed scroll progress -> director state -> camera,
 * network, props, overlay -> render.
 */
import './style.css';
import story from './data/story.json';
import { QUALITY, SCROLL_VIEWPORTS, motion } from './config.js';
import { Timeline } from './timeline.js';
import { Overlay } from './overlay.js';
import { ScrollController, CameraRig } from './scroll.js';
import { AutoPlay } from './autoplay.js';

const tl = new Timeline(story);
const scroll = new ScrollController(document.getElementById('scroll-space'), SCROLL_VIEWPORTS);
const overlay = new Overlay(document.getElementById('ui'), story, tl, {
  onJump: (p) => scroll.scrollTo(p),
  onAutoplay: () => autoplay.start(),
});
// Hands-free mode: stops once the end card has faded in. Never starts on its own.
const autoplay = new AutoPlay(document.getElementById('ui'), tl.at('end', 0.3));

// Debug hook for automated checks: ?debug exposes the timeline and scroll controller.
if (new URLSearchParams(window.location.search).has('debug')) window.__story = { tl, scroll };

async function start() {
  // Three.js and the scene modules load in parallel with the network data.
  const [{ Stage }, { loadNetwork, NetworkView }, { Director }, { Props }] = await Promise.all([
    import('./scene.js'),
    import('./network.js'),
    import('./director.js'),
    import('./props.js'),
  ]);

  let stage;
  try {
    stage = new Stage(document.getElementById('scene'));
  } catch (err) {
    console.warn('WebGL unavailable, narration only:', err);
    overlay.showError('3D graphics are unavailable on this device, but the story still works: scroll to read.');
    return textOnlyLoop();
  }

  const net = await loadNetwork(`${import.meta.env.BASE_URL}data/`);
  const view = new NetworkView(stage.scene, net, stage.resolution);
  const director = new Director(tl, view, net);
  const rig = new CameraRig(stage.camera, director.cameraKeys());
  if (window.__story) Object.assign(window.__story, { rig, director, view, stage });
  const props = new Props(stage.scene, net, view, rig.positionAt(tl.at('1.5', 0.5)));

  // Compile every shader up front so the first scroll doesn't hitch.
  stage.renderer.compile(stage.scene, stage.camera);
  overlay.ready();

  let last = performance.now();
  let slowFrames = 0;
  let frames = 0;

  function frame(now) {
    const rawDt = (now - last) / 1000;
    const dt = Math.min(rawDt, 0.1);
    last = now;
    const time = now / 1000;

    // Auto-play gets the real elapsed time (capped only for tab switches) so its speed
    // stays correct even when frames are slow.
    autoplay.tick(Math.min(rawDt, 0.5));
    const p = scroll.update(dt);
    const s = director.update(p, time);

    rig.update(p, time);
    stage.applyMood(s.mood);
    stage.beacon.material.opacity = s.beacon;
    stage.beacon.visible = s.beacon > 0.002;
    stage.beacon.scale.setScalar(s.beaconScale);
    const stars = stage.stars.material.uniforms;
    stars.uTime.value = time;
    stars.uAlpha.value = s.stars;
    stars.uTwinkle.value = motion.reduced ? 0 : 1;

    view.update(s, stage.camera, stage.dpr);
    props.update(s);
    overlay.update(p);
    stage.render();

    // Adaptive resolution: if frames are consistently slow, render fewer pixels.
    frames++;
    if (frames > 60) {
      slowFrames = dt > 1 / 45 ? slowFrames + 1 : Math.max(slowFrames - 1, 0);
      if (slowFrames > 45 && stage.dpr > QUALITY.minDpr) {
        stage.setDpr(stage.dpr - 0.25);
        slowFrames = 0;
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/** Fallback when WebGL is missing: keep the narration and progress bar working. */
function textOnlyLoop() {
  let last = performance.now();
  const loop = (now) => {
    const rawDt = (now - last) / 1000;
    const dt = Math.min(rawDt, 0.1);
    last = now;
    autoplay.tick(Math.min(rawDt, 0.5));
    overlay.update(scroll.update(dt));
    requestAnimationFrame(loop);
  };
  overlay.ready();
  requestAnimationFrame(loop);
}

start().catch((err) => {
  console.error(err);
  overlay.showError('Something went wrong while loading the network. Please reload the page.');
});

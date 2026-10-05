/**
 * Scroll and camera control.
 *
 * ScrollController turns native page scroll (wheel, keyboard, touch) into a smoothed
 * progress value 0..1. Everything visual is a pure function of that value, so
 * scrolling backwards simply plays the story in reverse.
 *
 * CameraRig flies the camera along CatmullRomCurve3 paths (position and look target)
 * through keyframes pinned to story beats.
 */
import { CatmullRomCurve3, Vector3 } from 'three';
import { motion } from './config.js';

export class ScrollController {
  constructor(spacer, viewports) {
    this.spacer = spacer;
    // `lvh` keeps the scroll length stable while mobile browser toolbars show and hide.
    const unit = CSS.supports('height', '1lvh') ? 'lvh' : 'vh';
    spacer.style.height = `${viewports * 100}${unit}`;
    this.target = 0;
    this.progress = 0;
    this.velocity = 0;
    this._read = this._read.bind(this);
    window.addEventListener('scroll', this._read, { passive: true });
    window.addEventListener('resize', this._read, { passive: true });
    this._read();
    this.progress = this.target; // start where the page is (e.g. after a reload mid-story)
  }

  _read() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    this.target = max > 0 ? Math.min(Math.max(window.scrollY / max, 0), 1) : 0;
  }

  /** Ease the displayed progress toward the scroll position (frame-rate independent). */
  update(dt) {
    const rate = motion.reduced ? 2.2 : 3.2;
    const k = 1 - Math.exp(-dt * rate);
    const prev = this.progress;
    this.progress += (this.target - this.progress) * k;
    if (Math.abs(this.target - this.progress) < 1e-6) this.progress = this.target;
    this.velocity = dt > 0 ? (this.progress - prev) / dt : 0;
    return this.progress;
  }

  /** Jump the page to a progress value (used by chapter markers). */
  scrollTo(p) {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollTo({ top: p * max, behavior: motion.reduced ? 'auto' : 'smooth' });
  }
}

export class CameraRig {
  /**
   * @param {{p:number,pos:number[],look:number[]}[]} keyframes  sorted by p
   */
  constructor(camera, keyframes) {
    this.camera = camera;
    this.keys = keyframes;
    this.posCurve = new CatmullRomCurve3(keyframes.map((k) => new Vector3(...k.pos)), false, 'centripetal');
    this.lookCurve = new CatmullRomCurve3(keyframes.map((k) => new Vector3(...k.look)), false, 'centripetal');
    this._pos = new Vector3();
    this._look = new Vector3();
  }

  /** Curve parameter for progress p. Each keyframe sits exactly at t = i / (n - 1). */
  curveParam(p) {
    const keys = this.keys;
    const n = keys.length;
    if (p <= keys[0].p) return 0;
    if (p >= keys[n - 1].p) return 1;
    let k = 0;
    while (k < n - 2 && p >= keys[k + 1].p) k++;
    const local = (p - keys[k].p) / (keys[k + 1].p - keys[k].p);
    // Partly eased so the camera lingers a little near each keyframe.
    const eased = local * local * (3 - 2 * local);
    return (k + local * 0.4 + eased * 0.6) / (n - 1);
  }

  positionAt(p, target = new Vector3()) {
    return this.posCurve.getPoint(this.curveParam(p), target);
  }

  update(p, time) {
    const t = this.curveParam(p);
    this.posCurve.getPoint(t, this._pos);
    this.lookCurve.getPoint(t, this._look);
    if (!motion.reduced) {
      // A gentle floating sway so the camera never feels locked down.
      this._pos.x += Math.sin(time * 0.21) * 0.35;
      this._pos.y += Math.sin(time * 0.17 + 1.3) * 0.25;
    }
    this.camera.position.copy(this._pos);
    this.camera.lookAt(this._look);
  }
}

/**
 * Beats 6.3-6.4: softmax probability bars beneath the output layer.
 *
 * 6.3  Ten bars rise to the network's raw scores (logits): uneven, some below zero.
 *      They then rescale into probabilities that add up to 1, filling a full-length
 *      "adds up to 1" bar below them. At this point softmax is shown *softened*
 *      (scores divided by a temperature T = 6) so the shares still look fairly even.
 * 6.4  T slides from 6 down to 1, the network's actual softmax: the winner's share
 *      shoots up and the rest shrink. That is the "spotlight" effect.
 *
 * Every bar height comes from the real forward pass. The logits are recomputed here
 * from the second hidden layer's activations and the trained output weights, and the
 * final bars are exactly the probabilities the output spheres use.
 */
import {
  CanvasTexture,
  Color,
  InstancedMesh,
  Line,
  LineBasicMaterial,
  BufferGeometry,
  Float32BufferAttribute,
  Matrix4,
  MeshBasicMaterial,
  PlaneGeometry,
  Sprite,
  SpriteMaterial,
  SRGBColorSpace,
} from 'three';
import { PALETTE } from './network.js';

const BASELINE_Y = -6.5;   // the zero line the bars stand on
const PROB_HEIGHT = 6;     // height of a bar with probability 1
const LOGIT_SCALE = 0.42;  // world units per unit of raw score
const BAR_WIDTH = 1.1;
const STACK_Y = -8.6;      // the "adds up to 1" bar
const STACK_HEIGHT = 0.55;
const T0 = 6;              // starting softmax temperature for beat 6.3

const _m = new Matrix4();
const _c = new Color();
const _amberBar = PALETTE.amber.clone().multiplyScalar(0.85);
const _amberSeg = PALETTE.amber.clone().multiplyScalar(0.8);

export class SoftmaxBars {
  constructor(scene, net, view) {
    // Real logits: z = h2 · W3 + b3 (the output layer before softmax).
    const { W, b } = net.params.trained[2];
    const h2 = net.runs.trained[2];
    this.logits = Float32Array.from(b, (bj, j) => {
      let s = bj;
      for (let i = 0; i < h2.length; i++) s += h2[i] * W[i * 10 + j];
      return s;
    });
    this.winner = this.logits.indexOf(Math.max(...this.logits));
    this.probs = new Float32Array(10); // scratch: softmax at the current temperature
    this.heights = new Float32Array(10);

    const out = view.layout[3];
    this.x = Float32Array.from({ length: 10 }, (_, k) => out[k * 3]);
    this.left = this.x[0] - BAR_WIDTH / 2;
    this.length = this.x[9] + BAR_WIDTH / 2 - this.left;
    // One flat row just in front of the middle spheres, so every bar meets the straight zero line.
    const zMid = -100 + 1.6;
    this.zMid = zMid;

    const plane = new PlaneGeometry(1, 1);
    const mat = () => new MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
    this.bars = new InstancedMesh(plane, mat(), 10);
    this.stack = new InstancedMesh(plane, mat(), 10);
    for (const mesh of [this.bars, this.stack]) {
      for (let k = 0; k < 10; k++) mesh.setColorAt(k, PALETTE.cyan);
      mesh.frustumCulled = false;
      mesh.visible = false;
      scene.add(mesh);
    }

    // Faint zero line under the bars, and the outline of the full "1" bar.
    const lineMat = new LineBasicMaterial({ color: 0x8fd8ff, transparent: true, opacity: 0, depthWrite: false });
    const line = (pts) => {
      const g = new BufferGeometry();
      g.setAttribute('position', new Float32BufferAttribute(pts, 3));
      const l = new Line(g, lineMat);
      l.visible = false;
      scene.add(l);
      return l;
    };
    const x0 = this.left - 0.6;
    const x1 = this.left + this.length + 0.6;
    this.zeroLine = line([x0, BASELINE_Y, zMid, x1, BASELINE_Y, zMid]);
    this.lineMat = lineMat;
    const outlineMat = lineMat.clone();
    this.outlineMat = outlineMat;
    const y0 = STACK_Y - STACK_HEIGHT / 2 - 0.12;
    const y1 = STACK_Y + STACK_HEIGHT / 2 + 0.12;
    const l0 = this.left - 0.12;
    const l1 = this.left + this.length + 0.12;
    const og = new BufferGeometry();
    og.setAttribute('position', new Float32BufferAttribute([l0, y0, zMid, l1, y0, zMid, l1, y1, zMid, l0, y1, zMid, l0, y0, zMid], 3));
    this.outline = new Line(og, outlineMat);
    this.outline.visible = false;
    scene.add(this.outline);

    // Small caption at the right end of the full bar.
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(190,232,255,1)';
    ctx.font = '500 30px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.fillText('adds up to 1', 128, 34);
    const tex = new CanvasTexture(canvas);
    tex.colorSpace = SRGBColorSpace;
    this.caption = new Sprite(new SpriteMaterial({ map: tex, transparent: true, opacity: 0, depthWrite: false }));
    this.caption.scale.set(4.8, 1.2, 1);
    // Centred just above the full bar, so it stays on screen on narrow phones too.
    this.caption.center.set(0.5, 0);
    this.caption.position.set((l0 + l1) / 2, y1 + 0.1, zMid);
    this.caption.visible = false;
    scene.add(this.caption);
  }

  /** Softmax of the real logits at temperature T, into this.probs. */
  _softmax(T) {
    const z = this.logits;
    const p = this.probs;
    const max = z[this.winner];
    let sum = 0;
    for (let k = 0; k < 10; k++) sum += (p[k] = Math.exp((z[k] - max) / T));
    for (let k = 0; k < 10; k++) p[k] /= sum;
  }

  update(s) {
    const b = s.bars;
    const visible = b.alpha > 0.002;
    this.bars.visible = this.zeroLine.visible = visible;
    const stackVisible = visible && b.sum > 0.002;
    this.stack.visible = this.outline.visible = this.caption.visible = stackVisible;
    if (!visible) return;

    // T goes from T0 to 1 geometrically, so the sharpening looks even across the beat.
    this._softmax(Math.pow(T0, 1 - b.temp));
    const p = this.probs;
    const cyan = PALETTE.cyan;
    const zMid = this.zMid;

    let cursor = this.left;
    for (let k = 0; k < 10; k++) {
      // Raw score -> probability: a straight morph between the two real values.
      const raw = this.logits[k] * LOGIT_SCALE * b.rise;
      const h = raw + (p[k] * PROB_HEIGHT - raw) * b.morph;
      const bottom = Math.min(BASELINE_Y, BASELINE_Y + h);
      const height = Math.max(Math.abs(h), 0.02);
      _m.makeScale(BAR_WIDTH, height, 1).setPosition(this.x[k], bottom + height / 2, zMid);
      this.bars.setMatrixAt(k, _m);

      // Cyan bars; negative raw scores dimmer; the winner turns amber at the verdict.
      _c.copy(cyan).multiplyScalar(h < 0 ? 0.35 : 0.7);
      if (k === this.winner) _c.lerp(_amberBar, b.amber);
      this.bars.setColorAt(k, _c);

      // The same probabilities laid end to end fill the full-length "1" bar.
      const w = Math.max(p[k] * this.length, 0.0001);
      _m.makeScale(w, STACK_HEIGHT, 1).setPosition(cursor + w / 2, STACK_Y, zMid);
      cursor += w;
      this.stack.setMatrixAt(k, _m);
      // Alternate shades so neighbouring segments stay distinguishable.
      _c.copy(cyan).multiplyScalar(k === this.winner ? 0.85 : k % 2 ? 0.22 : 0.42);
      if (k === this.winner) _c.lerp(_amberSeg, b.amber);
      this.stack.setColorAt(k, _c);
    }
    this.bars.instanceMatrix.needsUpdate = true;
    this.bars.instanceColor.needsUpdate = true;
    this.stack.instanceMatrix.needsUpdate = true;
    this.stack.instanceColor.needsUpdate = true;

    this.bars.material.opacity = 0.9 * b.alpha;
    this.lineMat.opacity = 0.35 * b.alpha;
    this.stack.material.opacity = 0.9 * b.alpha * b.sum;
    this.outlineMat.opacity = 0.6 * b.alpha * b.sum;
    this.caption.material.opacity = 0.85 * b.alpha * b.sum;
  }
}

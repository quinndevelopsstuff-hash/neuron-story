/**
 * Director: the choreography that ties STORY.md's stage directions to scroll progress.
 *
 * Every visual parameter is a "track": a list of keys [beatId, fractionThroughBeat, value],
 * smoothly interpolated. Because keys are pinned to beat ids (not raw scroll numbers),
 * narration timing can change in STORY.md and the visuals follow automatically.
 *
 * update(p, time) writes into one reusable `state` object read by NetworkView, Props
 * and the Stage. Nothing here allocates per frame.
 */
import { Color } from 'three';
import { motion } from './config.js';
import { HEATMAP_NEURON } from './props.js';

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const smooth = (t) => t * t * (3 - 2 * t);

class Track {
  /** @param keys {[string, number, number][]} [beatId, fraction, value] */
  constructor(tl, keys, { ease = true } = {}) {
    this.ps = Float64Array.from(keys, ([id, f]) => tl.at(id, f));
    this.vs = Float64Array.from(keys, (k) => k[2]);
    this.ease = ease;
    for (let i = 1; i < this.ps.length; i++) {
      if (this.ps[i] < this.ps[i - 1]) throw new Error(`Track keys out of order at ${keys[i][0]}`);
    }
  }

  at(p) {
    const { ps, vs } = this;
    const n = ps.length;
    if (p <= ps[0]) return vs[0];
    if (p >= ps[n - 1]) return vs[n - 1];
    let i = 0;
    while (p >= ps[i + 1]) i++;
    const span = ps[i + 1] - ps[i];
    let t = span > 0 ? (p - ps[i]) / span : 1;
    if (this.ease) t = smooth(t);
    return vs[i] + (vs[i + 1] - vs[i]) * t;
  }
}

export class Director {
  /**
   * @param {import('./timeline.js').Timeline} tl
   * @param {import('./network.js').NetworkView} view
   * @param {object} net  data from loadNetwork()
   */
  constructor(tl, view, net) {
    this.tl = tl;
    this.view = view;
    this.net = net;
    const T = (keys, opts) => new Track(tl, keys, opts);

    this.state = {
      time: 0,
      beacon: 0,
      beaconScale: 0.04,
      stars: 0,
      vis: [0, 0, 0, 0],
      act: [1, 0, 0, 0],
      inputReveal: 0,
      demoPixel: 0,
      demoPixelIndex: 14 * 28 + 9, // an empty pixel just left of centre
      jitter: 0,
      scribble: 0,
      weightMix: 0,
      grey: 0,
      labels: 0,
      focus: 0,
      focusFire: 0,
      focusWaver: 0,
      out7Highlight: 0,
      out7Pulse: 0,
      outProbs: new Float32Array(10),
      conn: [0, 1, 2].map(() => ({ opacity: 0, lit: 0, sign: 1, pulse: 0, width: 1, back: 0, backPos: 2 })),
      props: {
        ghosts: 0, ghostT: 0, biasRing: 0, biasAngle: 0, spark: 0, sparkT: 0,
        fragments: 0, assemble: 0, perfect7: 0, weightMaps: 0, gabors: 0,
        tag: 0, tagFlip: 0, loss: 0, heatmap: 0,
      },
      bars: { alpha: 0, rise: 0, morph: 0, sum: 0, temp: 0, amber: 0 },
      mood: { bg: new Color(), fog: 0.004, bloom: 1, bloomRadius: 0.55 },
    };

    /* ---------------- chapter 1: the dark ---------------- */
    this.beacon = T([['intro', 0, 1], ['1.10', 0.5, 1], ['2.1', 0.4, 0], ['7.15', 0.2, 0], ['7.15', 0.9, 1], ['7.16', 0.8, 1], ['end', 0.6, 0]]);
    this.beaconScale = T([['intro', 0, 0.03], ['1.7', 0.5, 0.05], ['1.10', 0.5, 0.09], ['2.1', 0.4, 0.12], ['7.15', 0.1, 0.035], ['7.16', 0.4, 0.03], ['7.16', 0.6, 0.05], ['7.16', 0.9, 0.03]]);
    this.stars = T([['1.2', 0, 0], ['1.2', 0.7, 0.55], ['1.7', 0.8, 1], ['7.16', 0, 1], ['end', 0.8, 0.5]]);
    this.ghosts = T([['1.5', 0, 0], ['1.5', 0.15, 1], ['1.5', 1, 1], ['1.6', 0.1, 0]]);
    this.ghostT = T([['1.5', 0, 0], ['1.6', 0.1, 1]], { ease: false });
    this.vis = T([['1.8', 0, 0], ['1.8', 0.8, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]);

    /* ---------------- chapter 2: input layer ---------------- */
    this.demoPixel = T([['2.3', 0.1, 0], ['2.3', 0.3, 1], ['2.3', 0.45, 1], ['2.3', 0.6, 0.45], ['2.3', 0.75, 0.45], ['2.3', 0.9, 0]]);
    this.inputReveal = T([['2.4', 0.08, 0], ['2.4', 0.8, 1]], { ease: false });
    this.jitter = T([['2.5', 0.6, 0], ['2.6', 0.2, 1], ['2.6', 0.9, 1], ['2.7', 0.2, 0]]);

    /* ---------------- chapter 3: neurons ---------------- */
    this.focus = T([['3.2', 0, 0], ['3.2', 0.5, 1], ['3.9', 1, 1], ['3.10', 0.4, 0]]);
    this.focusWaver = T([['3.4', 0.9, 0], ['3.5', 0.3, 1], ['3.5', 0.9, 1], ['3.6', 0.2, 0]]);
    this.focusFire = T([['3.6', 0.9, 0], ['3.7', 0.55, 0.55], ['3.9', 1, 0.55], ['3.10', 0.5, 0]]);
    this.biasRing = T([['3.5', 0.9, 0], ['3.6', 0.3, 1], ['3.6', 0.9, 1], ['3.7', 0.3, 0]]);
    this.biasAngle = T([['3.6', 0, 0], ['3.6', 1, 1.6]]);
    this.spark = T([['3.8', 0.1, 0], ['3.8', 0.2, 1], ['3.8', 0.85, 1], ['3.8', 0.95, 0]]);
    this.sparkT = T([['3.8', 0.15, 0], ['3.8', 0.9, 1]], { ease: false });

    /* ---------------- activations per layer ---------------- */
    this.act1 = T([['3.10', 0, 0], ['3.10', 0.5, 1], ['6.10', 0.1, 1], ['6.10', 0.7, 0.12], ['7.2', 0, 0.12], ['7.2', 0.5, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]);
    // Dimmed during the softmax bars (6.3-6.4) so the layer above the camera doesn't glare.
    this.act2 = T([['5.3', 0, 0], ['5.3', 0.7, 1], ['6.3', 0, 1], ['6.3', 0.25, 0.25], ['6.4', 1, 0.25], ['6.5', 0.4, 1], ['6.10', 0.1, 1], ['6.10', 0.7, 0.12], ['7.2', 0, 0.12], ['7.2', 0.5, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]);
    this.act3 = T([['6.2', 0.4, 0], ['6.2', 0.9, 0.22], ['6.5', 0.1, 0.22], ['6.5', 0.6, 1], ['6.10', 0.1, 1], ['6.10', 0.7, 0.08], ['7.2', 0, 0.08], ['7.2', 0.5, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]);
    this.labels = T([['6.1', 0, 0], ['6.1', 0.5, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]);

    /* ---------------- connections ---------------- */
    this.c0 = {
      opacity: T([['1.9', 0, 0], ['1.9', 0.4, 0.3], ['1.9', 1, 0], ['3.3', 0, 0], ['3.3', 0.5, 1], ['3.9', 1, 1], ['3.10', 0.5, 0.45], ['3.11', 1, 0.45], ['4.1', 0.3, 1], ['4.10', 1, 1], ['5.1', 0.5, 0.3], ['6.7', 0, 0.3], ['6.7', 0.5, 0.7], ['6.8', 0.5, 0.5], ['7.1', 0, 0.6], ['7.16', 0, 0.6], ['7.16', 0.5, 0]]),
      lit: T([['3.3', 1, 0], ['3.4', 0.4, 1], ['3.9', 1, 1], ['3.10', 0.5, 0.5], ['3.11', 1, 0.5], ['4.1', 0.3, 0], ['4.3', 1, 0], ['4.4', 0.3, 0.85], ['4.4', 1, 0.85], ['4.5', 0.4, 0.45], ['6.10', 0.2, 0.45], ['6.10', 0.6, 0]]),
      sign: T([['3.4', 1, 0], ['3.5', 0.4, 1]]),
      pulse: T([['4.5', 0, 0], ['4.5', 0.3, 1], ['4.7', 1, 1], ['4.8', 0.5, 0.25], ['5.1', 0.5, 0.25], ['6.7', 0, 0.25], ['6.7', 0.5, 1], ['6.8', 0.5, 0.3], ['6.10', 0.3, 0], ['7.12', 0, 0], ['7.12', 0.3, 1], ['7.14', 1, 1], ['7.15', 0.5, 0]]),
      width: T([['4.2', 0, 1], ['4.2', 0.4, 1.7], ['4.3', 0.5, 1]]),
    };
    this.c1 = {
      opacity: T([['1.9', 0, 0], ['1.9', 0.4, 0.3], ['1.9', 1, 0], ['5.1', 0.6, 0], ['5.2', 0.4, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]),
      lit: T([['5.2', 1, 0], ['5.3', 0.4, 0.6], ['6.10', 0.2, 0.6], ['6.10', 0.6, 0]]),
      sign: T([['intro', 0, 1]]),
      pulse: T([['5.3', 0, 0], ['5.3', 0.4, 1], ['5.5', 0, 0.35], ['6.7', 0, 0.35], ['6.7', 0.5, 1], ['6.8', 0.5, 0.3], ['6.10', 0.3, 0], ['7.12', 0, 0], ['7.12', 0.3, 1], ['7.14', 1, 1], ['7.15', 0.5, 0]]),
      width: T([['intro', 0, 1]]),
    };
    this.c2 = {
      opacity: T([['1.9', 0, 0], ['1.9', 0.4, 0.3], ['1.9', 1, 0], ['5.9', 0, 0], ['5.9', 0.6, 0.8], ['6.2', 0.3, 1], ['6.3', 0, 1], ['6.3', 0.3, 0.25], ['6.4', 1, 0.25], ['6.5', 0.4, 1], ['7.16', 0, 1], ['7.16', 0.5, 0]]),
      lit: T([['6.2', 0, 0], ['6.2', 0.5, 0.6], ['6.10', 0.2, 0.6], ['6.10', 0.6, 0]]),
      sign: T([['intro', 0, 1]]),
      pulse: T([['5.9', 0.2, 0], ['5.9', 0.6, 1], ['6.2', 0.6, 1], ['6.4', 1, 0.3], ['6.7', 0.5, 1], ['6.8', 0.5, 0.3], ['6.10', 0.3, 0], ['7.12', 0, 0], ['7.12', 0.3, 1], ['7.14', 1, 1], ['7.15', 0.5, 0]]),
      width: T([['intro', 0, 1]]),
    };
    // Network recedes while the (pass 2) loss landscape is shown.
    this.netDim = T([['7.5', 0, 1], ['7.5', 0.4, 0.35], ['7.8', 0.8, 0.35], ['7.9', 0.3, 1]]);

    /* ---------------- chapter 6/7: verdict, untrained, learning ---------------- */
    this.scribble = T([['6.9', 0, 0], ['6.9', 0.2, 1], ['6.9', 0.85, 1], ['6.10', 0.1, 0]]);
    this.grey = T([['6.10', 0, 0], ['6.10', 0.6, 1], ['7.1', 0, 1], ['7.1', 0.6, 0.35], ['7.12', 0, 0.35], ['7.12', 0.5, 0]]);
    this.weightMix = T([['4.9', 0.3, 0], ['4.9', 0.42, 1], ['4.9', 0.55, 1], ['4.9', 0.68, 0], ['6.10', 0.6, 0], ['7.1', 0.6, 1], ['7.11', 0, 1], ['7.11', 0.8, 0.92], ['7.12', 0.1, 0.92], ['7.13', 0.8, 0]]);
    this.curveActive = T([['7.11', 0.9, 0], ['7.12', 0.1, 1]]);
    this.curveT = T([['7.12', 0.1, 0], ['7.13', 0.8, 1]], { ease: false });
    this.out7Pulse = T([['7.2b', 0.5, 0], ['7.2b', 0.65, 1], ['7.2b', 0.85, 0]]);
    this.out7Highlight = T([['7.3', 0.8, 0], ['7.4', 0.3, 1], ['7.4', 1, 1], ['7.5', 0.3, 0]]);
    this.tag = T([['7.2b', 0, 0], ['7.2b', 0.15, 1], ['7.4', 1, 1], ['7.5', 0.2, 0]]);
    this.tagFlip = T([['7.2b', 0.2, 0], ['7.2b', 0.5, 1]]);
    this.loss = T([['7.3', 0.1, 0], ['7.3', 0.35, 1], ['7.4', 1, 1], ['7.5', 0.3, 0]]);
    // 4.6 heat map: fades in as the camera arrives, holds through 4.7, fades as it pulls back.
    this.heatmap = T([['4.5', 0.85, 0], ['4.6', 0.3, 1], ['4.7', 0.85, 1], ['4.8', 0.2, 0]]);
    this.back = T([['7.9', 0.1, 0], ['7.9', 0.2, 1], ['7.10', 0.95, 1], ['7.11', 0.1, 0]]);
    this.backG = T([['7.9', 0.2, 0], ['7.10', 0.9, 3]], { ease: false });

    /* ---------------- 6.3-6.4: softmax bars ---------------- */
    this.bars = {
      alpha: T([['6.3', 0, 0], ['6.3', 0.1, 1], ['6.5', 1, 1], ['6.6', 0.4, 0]]),
      rise: T([['6.3', 0.05, 0], ['6.3', 0.35, 1]]),   // bars rise to the raw scores
      morph: T([['6.3', 0.45, 0], ['6.3', 0.8, 1]]),   // raw scores -> probabilities (softened)
      sum: T([['6.3', 0.65, 0], ['6.3', 0.9, 1]]),     // "adds up to 1" bar appears
      temp: T([['6.4', 0.1, 0], ['6.4', 0.75, 1]]),    // temperature 6 -> 1: the spotlight
      amber: T([['6.4', 0.8, 0], ['6.5', 0.25, 1]]),   // winner turns amber at the verdict
    };

    /* ---------------- chapter 5 props ---------------- */
    this.fragments = T([['5.4', 0, 0], ['5.4', 0.3, 1], ['5.5', 0.6, 1], ['5.5', 0.75, 0]]);
    this.assemble = T([['5.5', 0, 0], ['5.5', 0.6, 1]]);
    this.perfect7 = T([['5.5', 0.4, 0], ['5.5', 0.7, 0.9], ['5.6', 0.1, 0.9], ['5.6', 0.3, 0]]);
    this.weightMaps = T([['5.6', 0.15, 0], ['5.6', 0.4, 1], ['5.7', 1, 1], ['5.8', 0.3, 0.4], ['5.9', 0.3, 0]]);
    this.gabors = T([['5.8', 0.1, 0], ['5.8', 0.3, 0.9], ['5.8', 0.8, 0.9], ['5.9', 0.1, 0]]);

    /* ---------------- mood per chapter ---------------- */
    const M = (id, f, bg, fog, bloom, bloomRadius = 0.55) => [id, f, { bg: new Color(bg), fog, bloom, bloomRadius }];
    this.moodKeys = [
      M('intro', 0, 0x010205, 0.0055, 1.2, 0.7),
      M('1.7', 0.5, 0x010205, 0.004, 1.2, 0.7),
      M('1.10', 0.8, 0x01040a, 0.0035, 1.1, 0.65),
      M('2.1', 0.5, 0x02070d, 0.008, 0.85),
      M('2.9', 0.6, 0x02070d, 0.008, 0.85),
      M('3.1', 0.4, 0x05041a, 0.014, 1.0),
      M('3.11', 0.6, 0x05041a, 0.014, 1.0),
      M('4.1', 0.3, 0x021214, 0.008, 0.9),
      M('4.10', 0.6, 0x021214, 0.009, 0.9),
      M('5.1', 0.4, 0x08041c, 0.017, 1.05),
      M('5.10', 0.4, 0x08041c, 0.015, 1.05),
      M('6.1', 0.3, 0x02060b, 0.009, 1.0),
      M('6.5', 0.2, 0x02060b, 0.009, 1.0),
      M('6.5', 0.7, 0x07050a, 0.009, 1.5, 0.7),
      M('6.7', 0.5, 0x02060b, 0.006, 1.2),
      M('6.9', 0.5, 0x02060b, 0.0055, 1.1),
      M('6.10', 0.6, 0x040506, 0.006, 0.9),
      M('7.2', 0.5, 0x050506, 0.006, 0.9),
      M('7.5', 0.5, 0x07080b, 0.012, 0.9),
      M('7.6', 0.5, 0x07080b, 0.02, 0.9),
      M('7.8', 0.6, 0x07080b, 0.012, 0.9),
      M('7.9', 0.5, 0x060406, 0.006, 1.0),
      M('7.12', 0.6, 0x02060c, 0.006, 1.1),
      M('7.14', 0.6, 0x02060c, 0.006, 1.1),
      M('7.15', 0.8, 0x010205, 0.004, 1.2, 0.7),
      M('end', 1, 0x010205, 0.004, 1.2, 0.7),
    ];
    this.moodPs = this.moodKeys.map(([id, f]) => tl.at(id, f));
  }

  /** Camera keyframes [{p, pos, look}], pinned to beats. */
  cameraKeys() {
    const tl = this.tl;
    const v = this.view;
    const F = v.position(1, v.focusIndex);
    const f = (dx, dy, dz) => [F[0] + dx, F[1] + dy, F[2] + dz];
    const H = v.position(1, HEATMAP_NEURON);
    const out7 = v.position(3, 7);
    const out1 = v.position(3, 1);
    // A bright top-bar pixel feeding the focus neuron, for following "one bright thread" (4.4).
    const hero = this.net.images.hero;
    const W0 = this.net.params.trained[0].W;
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < 784; i++) {
      const score = hero[i] * W0[i * 32 + v.focusIndex];
      if (score > bestScore) { bestScore = score; best = i; }
    }
    const P = v.position(0, best);
    const along = (t, dx, dy, dz) => [P[0] + (F[0] - P[0]) * t + dx, P[1] + (F[1] - P[1]) * t + dy, P[2] + (F[2] - P[2]) * t + dz];
    const C = [0, 0, -50];

    const K = (id, frac, pos, look) => ({ p: tl.at(id, frac), pos, look });
    return [
      K('intro', 0, [0, 0, 440], C),
      K('1.1', 0.6, [0, 0, 420], C),
      K('1.5', 0.5, [0, 1, 330], C),
      K('1.8', 0.5, [0, 0, 240], C),
      K('1.10', 0.6, [0, 0, 155], [0, 0, -40]),
      K('ch2', 0.5, [0, 0, 98], [0, 0, 0]),
      K('2.1', 0.6, [0, 0, 64], [0, 0, 0]),
      K('2.3', 0.5, [0, 0, 44], [0, 0, 0]),
      K('2.4', 0.7, [0, 0, 38], [0, 0, 0]),
      K('2.5', 0.6, [1.5, 2, 13], [0.5, 1.5, 0]),
      K('2.6', 0.6, [-2.5, -1, 9], [-0.5, -2.5, 0]),
      K('2.7', 0.6, [0, 1, 46], [0, 0, 0]),
      K('2.8', 0.6, [7, 2, 36], [0, 0, 0]),
      K('2.9', 0.7, [25, 7, 14], [0, 0, -30]),
      K('ch3', 0.5, [14, 4, -10], [0, 0, -40]),
      K('3.1', 0.6, [6, 2, -17], [0, 0, -40]),
      K('3.2', 0.6, f(3, 1.5, 9), [...F]),
      K('3.3', 0.6, [F[0] + 24, F[1] + 9, -10], [F[0] * 0.5, F[1] * 0.4, -20]),
      K('3.4', 0.6, [F[0] + 20, F[1] + 6, -14], [F[0] * 0.6, F[1] * 0.5, -24]),
      K('3.5', 0.6, f(5, 3, 12), f(0, 0, 4)),
      K('3.6', 0.6, f(2, 1, 10), [...F]),
      K('3.7', 0.6, f(-2, 1, 10), [...F]),
      K('3.8', 0.6, f(7, 3, 8), [F[0] * 0.5, F[1] * 0.5, -56]),
      K('3.9', 0.6, f(3, 2, 12), [...F]),
      K('3.10', 0.6, [0, 3, -10], [0, 0, -40]),
      K('3.11', 0.6, [-4, 2, -14], [0, 0, -40]),
      K('ch4', 0.5, [0, 18, -6], [0, 0, -25]),
      K('4.1', 0.6, [0, 42, -6], [0, 0, -22]),
      K('4.2', 0.6, [14, 30, -14], [0, 0, -22]),
      K('4.3', 0.6, [20, 22, -6], [0, 0, -22]),
      K('4.4', 0.3, along(0.12, 3, 3, 6), [...F]),
      K('4.4', 0.8, along(0.7, 5, 4, 4), [...F]),
      K('4.5', 0.6, [16, 8, 8], [0, 0, -22]),
      // 4.6-4.7 frame the heat-map neuron and its weight panel (to its outer side).
      K('4.6', 0.6, [H[0] - 2.5, H[1] + 1, H[2] + 20], [H[0] - 4.2, H[1] - 1.6, H[2]]),
      K('4.7', 0.6, [H[0] - 5.5, H[1] + 0.5, H[2] + 18.5], [H[0] - 4.4, H[1] - 1.6, H[2]]),
      K('4.8', 0.6, [34, 22, 22], [0, 0, -45]),
      K('4.9', 0.6, [26, 14, 8], [0, 0, -30]),
      K('4.10', 0.7, [0, 4, -22], [0, 0, -45]),
      K('ch5', 0.5, [0, 2, -34], [0, 0, -60]),
      K('5.1', 0.6, [0, 1, -52], [0, 0, -72]),
      K('5.2', 0.6, [14, 6, -50], [0, 0, -60]),
      K('5.3', 0.6, [18, 5, -46], [0, 0, -64]),
      K('5.4', 0.6, [26, 7, -52], [0, 0, -62]),
      K('5.5', 0.6, [20, 5, -66], [0, 0, -82]),
      K('5.6', 0.6, [30, 8, -28], [0, 0, -46]),
      K('5.7', 0.6, [10, 4, -60], [0, 0, -72]),
      K('5.8', 0.6, [-10, 5, -60], [0, 3, -72]),
      K('5.9', 0.6, [-12, 3, -80], [0, 0, -90]),
      K('5.10', 0.6, [0, 2, -80], [0, 0, -100]),
      K('ch6', 0.5, [0, 3, -77], [0, 0, -100]),
      K('6.1', 0.6, [0, 4, -74], [0, 0, -100]),
      K('6.2', 0.6, [-14, 8, -78], [0, 0, -92]),
      // Framed lower so the bars beneath the spheres sit above the narration panel.
      // Pulled back to fit all ten digits, looking up from below hidden layer 2 so it stays out of frame.
      K('6.3', 0.15, [-15, -10, -73], [0, -4, -100]), // swing wide of hidden layer 2's neurons
      K('6.3', 0.6, [0, -14, -66], [0, -4, -100]),
      K('6.4', 0.6, [0, -14, -67], [0, -4, -100]),
      K('6.5', 0.6, [out7[0] + 2, 2, -88], [out7[0], 0, out7[2]]),
      K('6.6', 0.6, [out1[0] + 3, 2, -90], [out1[0], 0, out1[2]]),
      K('6.7', 0.6, [10, 12, -124], [0, 0, -40]),
      K('6.8', 0.6, [42, 14, -62], [0, 0, -50]),
      K('6.9', 0.5, [32, 12, 36], [0, 0, -50]),
      K('6.10', 0.6, [38, 16, 30], [0, 0, -50]),
      K('ch7', 0.5, [40, 18, 28], [0, 0, -50]),
      K('7.1', 0.6, [38, 16, 22], [0, 0, -50]),
      K('7.2', 0.6, [34, 12, 12], [0, 0, -55]),
      K('7.2b', 0.6, [28, 8, 30], [8, 4, -8]),
      K('7.3', 0.6, [0, 6, -84], [0, 3, -100]),
      K('7.4', 0.6, [out7[0], 3, -87], [out7[0], 1, -100]),
      K('7.5', 0.6, [0, 70, -20], [0, 0, -62]),
      K('7.6', 0.6, [0, 55, -40], [0, 0, -70]),
      K('7.7', 0.6, [10, 52, -50], [0, 0, -72]),
      K('7.8', 0.6, [0, 64, -60], [0, 0, -64]),
      K('7.9', 0.6, [55, 10, -55], [0, 0, -50]),
      K('7.10', 0.6, [52, 8, -45], [0, 0, -50]),
      K('7.11', 0.6, [48, 10, -30], [0, 0, -50]),
      K('7.12', 0.5, [36, 12, 30], [0, 0, -50]),
      K('7.13', 0.6, [30, 10, 24], [0, 0, -50]),
      K('7.14', 0.6, [20, 8, 42], [0, 0, -45]),
      K('7.15', 0.7, [0, 0, 330], C),
      K('7.16', 0.6, [0, 0, 420], C),
      K('end', 0.5, [0, 0, 440], C),
    ];
  }

  update(p, time) {
    const s = this.state;
    const reduced = motion.reduced;
    s.time = reduced ? time * 0.4 : time;

    /* chapter 1 */
    let beacon = this.beacon.at(p);
    const b13 = this.tl.get('1.3');
    if (!reduced && p > b13.start && p < b13.end) {
      const local = (p - b13.start) / (b13.end - b13.start);
      beacon *= 1 - 0.7 * Math.exp(-Math.pow((local - 0.35) / 0.06, 2)); // one flicker
    }
    s.beacon = beacon;
    s.beaconScale = this.beaconScale.at(p) * (1 + (reduced ? 0 : 0.08 * Math.sin(time * 1.4)));
    s.stars = this.stars.at(p);
    s.props.ghosts = this.ghosts.at(p);
    s.props.ghostT = this.ghostT.at(p);

    const netDim = this.netDim.at(p);
    const vis = this.vis.at(p);
    s.vis[0] = s.vis[1] = s.vis[2] = s.vis[3] = vis * (0.6 + 0.4 * netDim);

    /* chapter 2 */
    s.demoPixel = this.demoPixel.at(p);
    s.inputReveal = this.inputReveal.at(p);
    s.jitter = reduced ? 0 : this.jitter.at(p);

    /* chapter 3 */
    s.focus = this.focus.at(p);
    s.focusWaver = reduced ? 0 : this.focusWaver.at(p);
    s.focusFire = this.focusFire.at(p);
    s.props.biasRing = this.biasRing.at(p);
    s.props.biasAngle = this.biasAngle.at(p);
    s.props.spark = this.spark.at(p);
    s.props.sparkT = this.sparkT.at(p);

    /* activations */
    s.act[0] = 1;
    s.act[1] = this.act1.at(p) * netDim;
    s.act[2] = this.act2.at(p) * netDim;
    s.act[3] = this.act3.at(p) * netDim;
    s.labels = this.labels.at(p) * netDim;

    /* connections */
    const tracks = [this.c0, this.c1, this.c2];
    for (let li = 0; li < 3; li++) {
      const c = s.conn[li];
      const t = tracks[li];
      c.opacity = t.opacity.at(p) * netDim;
      c.lit = t.lit.at(p);
      c.sign = t.sign.at(p);
      c.pulse = t.pulse.at(p) * (reduced ? 0.4 : 1);
      c.width = t.width.at(p);
    }

    /* backpropagation wave: output layer first, then backwards */
    const back = this.back.at(p);
    const g = this.backG.at(p);
    for (let li = 0; li < 3; li++) {
      const local = g - (2 - li);
      const band = clamp01((local + 0.1) / 0.15) * clamp01((1.1 - local) / 0.15);
      s.conn[li].back = back * band * 1.6;
      s.conn[li].backPos = 1 - local;
    }

    /* verdict, untrained network, training */
    s.scribble = this.scribble.at(p);
    s.grey = this.grey.at(p);
    let mix = this.weightMix.at(p);
    if (reduced) {
      // Calmer: the 4.9 flicker becomes a gentle partial crossfade.
      const b49 = this.tl.get('4.9');
      if (p > b49.start && p < b49.end) mix *= 0.5;
    }
    s.weightMix = mix;
    s.out7Pulse = this.out7Pulse.at(p);
    s.out7Highlight = this.out7Highlight.at(p);
    s.props.tag = this.tag.at(p);
    s.props.tagFlip = this.tagFlip.at(p);
    s.props.loss = this.loss.at(p);
    s.props.heatmap = this.heatmap.at(p);
    this._outputs(p, mix);

    /* chapter 5 props */
    s.props.fragments = this.fragments.at(p);
    s.props.assemble = this.assemble.at(p);
    let p7 = this.perfect7.at(p);
    const b56 = this.tl.get('5.6');
    if (!reduced && p > b56.start && p < b56.start + (b56.end - b56.start) * 0.3) {
      p7 *= 0.55 + 0.45 * Math.sin(time * 38); // flicker as the tidy story breaks apart
    }
    s.props.perfect7 = p7;
    s.props.weightMaps = this.weightMaps.at(p);
    s.props.gabors = this.gabors.at(p);

    for (const key in this.bars) s.bars[key] = this.bars[key].at(p);

    this._mood(p);
    return s;
  }

  /** Output probabilities: trained / untrained / training-curve / scribble, blended. */
  _outputs(p, mix) {
    const out = this.state.outProbs;
    const { runs, meta } = this.net;
    const tr = runs.trained[3];
    const un = runs.untrained[3];
    const sc = runs.scribble[3];
    const curveActive = this.curveActive.at(p);
    const curve = meta.hero.trainingCurve;
    let ci = 0;
    let cf = 0;
    if (curveActive > 0) {
      const x = this.curveT.at(p) * (curve.length - 1);
      ci = Math.min(Math.floor(x), curve.length - 2);
      cf = x - ci;
    }
    const scribble = this.state.scribble;
    for (let k = 0; k < 10; k++) {
      let v = tr[k] + (un[k] - tr[k]) * mix;
      if (curveActive > 0) {
        const c = curve[ci].probs[k] + (curve[ci + 1].probs[k] - curve[ci].probs[k]) * cf;
        v += (c - v) * curveActive;
      }
      out[k] = v + (sc[k] - v) * scribble;
    }
  }

  _mood(p) {
    const keys = this.moodKeys;
    const ps = this.moodPs;
    const m = this.state.mood;
    let i = 0;
    while (i < ps.length - 2 && p >= ps[i + 1]) i++;
    const t = smooth(clamp01((p - ps[i]) / (ps[i + 1] - ps[i] || 1)));
    const a = keys[i][2];
    const b = keys[i + 1][2];
    m.bg.copy(a.bg).lerp(b.bg, t);
    m.fog = a.fog + (b.fog - a.fog) * t;
    m.bloom = a.bloom + (b.bloom - a.bloom) * t;
    m.bloomRadius = a.bloomRadius + (b.bloomRadius - a.bloomRadius) * t;
  }
}

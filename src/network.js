/**
 * The neural network: real data, 3D layout and rendering.
 *
 *  - loadNetwork(): fetches the trained/untrained weights exported by training/train.py
 *    and runs the actual forward pass on the story's images.
 *  - NetworkView: neurons as instanced spheres, connections as instanced screen-space
 *    ribbons whose width and brightness come from the real weights.
 *
 * NetworkView.update() reads a plain `state` object produced by the choreography
 * (director.js) each frame. It writes into preallocated typed arrays only.
 */
import {
  AdditiveBlending,
  BufferAttribute,
  CanvasTexture,
  Color,
  IcosahedronGeometry,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  InstancedMesh,
  Matrix4,
  Mesh,
  ShaderMaterial,
  Sprite,
  SpriteMaterial,
  SRGBColorSpace,
} from 'three';
import { FOG_GLSL, fogUniforms } from './scene.js';
import { QUALITY } from './config.js';

export const SIZES = [784, 32, 16, 10];

/* ------------------------------------------------------------------ data */

export async function loadNetwork(base = './data/') {
  const [meta, buffer] = await Promise.all([
    fetch(`${base}network.json`).then((r) => {
      if (!r.ok) throw new Error(`network.json: ${r.status}`);
      return r.json();
    }),
    fetch(`${base}network.bin`).then((r) => {
      if (!r.ok) throw new Error(`network.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  const floats = new Float32Array(buffer);

  const params = { trained: [], untrained: [] };
  for (const entry of meta.weightsLayout) {
    const arr = floats.subarray(entry.offset, entry.offset + entry.length);
    const layers = params[entry.set];
    layers[entry.layer] ??= {};
    layers[entry.layer][entry.kind === 'weights' ? 'W' : 'b'] = arr;
  }

  const toUnit = (pixels) => Float32Array.from(pixels, (v) => v / 255);
  const hero = toUnit(meta.hero.pixels);
  const scribble = toUnit(meta.scribble.pixels);

  return {
    meta,
    params,
    images: { hero, scribble, ghosts: meta.ghostSevens.map(toUnit) },
    // Real forward passes, computed once here in the browser from the real weights.
    runs: {
      trained: forward(params.trained, hero),
      untrained: forward(params.untrained, hero),
      scribble: forward(params.trained, scribble),
    },
  };
}

/** Forward pass: weighted sum + bias, ReLU on hidden layers, softmax on the output. */
export function forward(layers, input) {
  const acts = [input];
  let x = input;
  layers.forEach(({ W, b }, li) => {
    const nIn = x.length;
    const nOut = b.length;
    const z = new Float32Array(nOut);
    for (let j = 0; j < nOut; j++) {
      let s = b[j];
      for (let i = 0; i < nIn; i++) s += x[i] * W[i * nOut + j];
      z[j] = s;
    }
    if (li < layers.length - 1) {
      for (let j = 0; j < nOut; j++) z[j] = Math.max(0, z[j]);
    } else {
      let max = -Infinity;
      for (const v of z) max = Math.max(max, v);
      let sum = 0;
      for (let j = 0; j < nOut; j++) sum += (z[j] = Math.exp(z[j] - max));
      for (let j = 0; j < nOut; j++) z[j] /= sum;
    }
    acts.push(z);
    x = z;
  });
  return acts; // [input, hidden1, hidden2, probabilities]
}

/**
 * Time-lapse data for beats 7.12-7.14 (training/train.py -> timelapse.{json,bin}), loaded
 * after the site has started. Contains the real weight snapshots saved during training,
 * real MNIST training digits to stream through the grid, and real test digits with the
 * final network's real predictions.
 */
export async function loadTimelapse(net, base = './data/') {
  const [meta, buffer] = await Promise.all([
    fetch(`${base}timelapse.json`).then((r) => {
      if (!r.ok) throw new Error(`timelapse.json: ${r.status}`);
      return r.json();
    }),
    fetch(`${base}timelapse.bin`).then((r) => {
      if (!r.ok) throw new Error(`timelapse.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  const bytes = new Uint8Array(buffer);
  const int8 = new Int8Array(buffer);
  const last = meta.snapshots.length - 1;

  // Snapshot k: [{W, b}] per layer. The endpoints are the float32 untrained/trained
  // weights already loaded; intermediate ones are dequantised from int8.
  const snapshots = meta.snapshots.map((snap, k) => ({
    step: snap.step,
    layers: snap.layers.map((info, li) => {
      if (k === 0) return net.params.untrained[li];
      if (k === last) return net.params.trained[li];
      const W = new Float32Array(info.length);
      for (let i = 0; i < info.length; i++) W[i] = int8[info.offset + i] * info.scale;
      return { W, b: Float32Array.from(info.biases) };
    }),
  }));
  // The hero 7's real activations at every snapshot (one forward pass each, at load).
  const heroRuns = snapshots.map((s) => forward(s.layers, net.images.hero));

  const t = meta.testDigits;
  return {
    snapshots,
    heroRuns,
    bytes,
    train: { offset: meta.trainDigits.offset, count: meta.trainDigits.count },
    test: {
      offset: t.offset,
      count: t.count,
      labels: t.labels,
      preds: t.preds,
      probs: Float32Array.from(t.probs.flat()),
    },
  };
}

/* ------------------------------------------------------------------ layout */

/** World-space positions of every neuron, layer by layer (Float32Array of xyz). */
export function buildLayout() {
  const input = new Float32Array(784 * 3);
  for (let i = 0; i < 784; i++) {
    const r = Math.floor(i / 28);
    const c = i % 28;
    input.set([c - 13.5, 13.5 - r, 0], i * 3);
  }
  const h1 = new Float32Array(32 * 3);
  for (let j = 0; j < 32; j++) {
    const r = Math.floor(j / 8);
    const c = j % 8;
    const x = (c - 3.5) * 3.4;
    const y = (1.5 - r) * 3.4;
    h1.set([x, y, -40 - 0.03 * x * x], j * 3);
  }
  const h2 = new Float32Array(16 * 3);
  for (let j = 0; j < 16; j++) {
    const r = Math.floor(j / 4);
    const c = j % 4;
    const x = (c - 1.5) * 3.2;
    const y = (1.5 - r) * 3.2;
    h2.set([x, y, -72 - 0.04 * (x * x + y * y)], j * 3);
  }
  const out = new Float32Array(10 * 3);
  for (let k = 0; k < 10; k++) {
    const x = (k - 4.5) * 4.2;
    out.set([x, 0, -100 + 0.012 * x * x], k * 3);
  }
  return [input, h1, h2, out];
}

export const LAYER_Z = { input: 0, h1: -40, h2: -72, out: -100 };

/*
 * Beat 2.7: where each pixel sits when the grid is unrolled into one long line.
 * Pixel i (row-major, so i = row * 28 + col) goes to slot i, so the line is the 28 rows
 * laid end to end in order. It starts at the grid's right edge and recedes diagonally
 * into the fog: 784 slots x 0.38 units is ~300 units long.
 */
const LINE_START = [16, 0, 0];
const LINE_DIR = [0.954, 0, -0.3];
const LINE_SPACING = 0.38;
/** How many rows are in flight at once while unrolling. */
const ROWS_IN_FLIGHT = 4;

export function buildUnrolledLine() {
  const out = new Float32Array(784 * 3);
  for (let i = 0; i < 784; i++) {
    const d = i * LINE_SPACING;
    out[i * 3] = LINE_START[0] + LINE_DIR[0] * d;
    out[i * 3 + 1] = LINE_START[1] + LINE_DIR[1] * d;
    out[i * 3 + 2] = LINE_START[2] + LINE_DIR[2] * d;
  }
  return out;
}

/* ------------------------------------------------------------------ shaders */

const NODE_VERT = /* glsl */ `
  attribute vec4 aGlow;          // rgb = emitted light, w = size multiplier
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vView;
  varying float vDepth;
  void main() {
    vec4 mv = modelViewMatrix * instanceMatrix * vec4(position * aGlow.w, 1.0);
    vNormal = normalize(normalMatrix * mat3(instanceMatrix) * normal);
    vView = -mv.xyz;
    vDepth = -mv.z;
    vColor = aGlow.rgb;
    gl_Position = projectionMatrix * mv;
  }
`;

const NODE_FRAG = /* glsl */ `
  ${FOG_GLSL}
  uniform vec3 uBase;
  uniform float uGrey;
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vView;
  varying float vDepth;
  void main() {
    float facing = max(dot(normalize(vNormal), normalize(vView)), 0.0);
    float rim = pow(1.0 - facing, 2.0);
    // As a neuron heats up its cool body colour gives way, so firing reads as pure amber.
    float heat = clamp(vColor.r * 2.5, 0.0, 1.0);
    vec3 col = uBase * (0.35 + 1.1 * rim) * (1.0 - 0.85 * heat) + vColor * (0.6 + 0.5 * rim + 0.2 * facing);
    float lum = dot(col, vec3(0.3, 0.59, 0.11));
    col = mix(col, vec3(lum) * vec3(0.8, 0.85, 0.9), uGrey);
    col = mix(col, uFogColor, fogFactor(vDepth));
    gl_FragColor = vec4(col, 1.0);
  }
`;

/*
 * Connections are drawn as quads expanded in screen space, so width can follow
 * weight strength (WebGL ignores lineWidth). Segments crossing the near plane are
 * clipped in view space first, which matters because the camera flies through layers.
 */
const CONN_VERT = /* glsl */ `
  attribute vec2 corner;   // x: 0 at source, 1 at target; y: -1 / +1 side
  attribute vec3 aStart;
  attribute vec3 aEnd;
  attribute vec2 aW;       // normalised weight: trained, untrained
  attribute vec2 aSrc;     // normalised source activation: trained, untrained
  attribute float aDst;
  attribute float aRand;
  uniform vec2 uResolution;
  uniform float uPixelRatio;
  uniform float uNear;
  uniform float uMix;
  uniform float uWidth;
  uniform float uFocus;
  varying float vT;
  varying float vW;
  varying float vSrc;
  varying float vFocus;
  varying float vRand;
  varying float vDepth;
  varying float vFade;
  void main() {
    vW = mix(aW.x, aW.y, uMix);
    vSrc = mix(aSrc.x, aSrc.y, uMix);
    vFocus = abs(aDst - uFocus) < 0.5 ? 1.0 : 0.0;
    vRand = aRand;

    vec4 a = modelViewMatrix * vec4(aStart, 1.0);
    vec4 b = modelViewMatrix * vec4(aEnd, 1.0);
    float zn = -uNear * 1.5;
    float ta = 0.0;
    float tb = 1.0;
    vT = corner.x;
    vDepth = 0.0;
    vFade = 0.0;
    if (a.z > zn && b.z > zn) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // fully behind the camera: discard
      return;
    }
    if (a.z > zn) {
      float k = (a.z - zn) / (a.z - b.z);
      a = mix(a, b, k);
      ta = k;
    } else if (b.z > zn) {
      float k = (b.z - zn) / (b.z - a.z);
      b = mix(b, a, k);
      tb = 1.0 - k;
    }
    vT = mix(ta, tb, corner.x);

    vec4 ca = projectionMatrix * a;
    vec4 cb = projectionMatrix * b;
    vec2 half_res = uResolution * 0.5;
    vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * half_res;
    float len = length(d);
    vec2 dir = len > 1e-5 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-dir.y, dir.x);

    vec4 c = mix(ca, cb, corner.x);
    float depth = -mix(a.z, b.z, corner.x);
    float aw = abs(vW);
    float px = uWidth * (0.35 + 2.2 * aw * aw) * clamp(40.0 / max(depth, 1.0), 0.4, 1.8) * uPixelRatio;
    float drawPx = max(px, uPixelRatio);
    vFade = px / drawPx; // thinner-than-a-pixel lines fade instead of shimmering
    c.xy += n * corner.y * (drawPx * 0.5) / half_res * c.w;
    vDepth = depth;
    gl_Position = c;
  }
`;

const CONN_FRAG = /* glsl */ `
  ${FOG_GLSL}
  uniform float uOpacity;
  uniform float uLit;
  uniform float uSign;
  uniform float uFocusMix;
  uniform float uPulse;
  uniform float uTime;
  uniform float uGrey;
  uniform float uBack;
  uniform float uBackPos;
  uniform float uEndFade;
  uniform float uBoost;
  uniform vec3 uPos;
  uniform vec3 uNeg;
  uniform vec3 uAmber;
  uniform vec3 uRed;
  varying float vT;
  varying float vW;
  varying float vSrc;
  varying float vFocus;
  varying float vRand;
  varying float vDepth;
  varying float vFade;
  void main() {
    float aw = abs(vW);
    float contrib = vSrc * aw;
    float alpha = (0.012 + 0.4 * aw * aw) * uBoost * mix(1.0, 2.0, uFocusMix);
    // "Lit" mode: threads carrying signal from active sources stand out, the rest recede.
    alpha *= mix(1.0, min(0.1 + 4.0 * contrib, 2.5), uLit);
    alpha *= mix(1.0, vFocus, uFocusMix);
    // Fade near the target so thousands of converging threads don't form a white blob.
    // In focus mode the camera looks down the converging cone, so fade much earlier.
    float fadeStart = mix(0.55, 0.3, uFocusMix);
    alpha *= smoothstep(0.0, 0.04, vT) * mix(1.0, smoothstep(1.0, fadeStart, vT), uEndFade);

    vec3 signCol = vW >= 0.0 ? uPos : uNeg;
    vec3 col = mix(uPos, signCol, uSign) * alpha;

    // Signal pulses: amber beads travelling source -> target, strength = |input x weight|.
    float pos = fract(uTime * 0.28 + vRand);
    float bead = 1.0 - smoothstep(0.0, 0.035, abs(vT - pos));
    col += uAmber * bead * uPulse * min(contrib * 2.0, 1.0) * min(uBoost * 4.0, 1.0) * mix(1.0, vFocus, uFocusMix);

    // Backpropagation wave: red-orange band travelling target -> source.
    float dw = (vT - uBackPos) / 0.07;
    float wave = exp(-dw * dw);
    col += uRed * wave * uBack * (0.08 + 0.6 * aw) * uBoost;

    float lum = dot(col, vec3(0.3, 0.59, 0.11));
    col = mix(col, vec3(lum) * 0.8, uGrey);
    col *= 1.0 - fogFactor(vDepth);
    gl_FragColor = vec4(col * uOpacity * vFade, 1.0);
  }
`;

/* ------------------------------------------------------------------ colours */

export const PALETTE = {
  cyan: new Color(0.25, 0.8, 1.0),
  violet: new Color(0.55, 0.32, 1.0),
  amber: new Color(1.0, 0.36, 0.04),
  red: new Color(1.0, 0.28, 0.12),
  pixel: new Color(0.55, 0.9, 1.1),
};

/* ------------------------------------------------------------------ view */

/** Robust scale for normalising weights: the 99th percentile of |w|. */
function percentileAbs(arr, q) {
  const abs = Float32Array.from(arr, Math.abs).sort();
  return abs[Math.floor(q * (abs.length - 1))] || 1;
}

const _m = new Matrix4();

export class NetworkView {
  constructor(scene, net, resolution) {
    this.net = net;
    this.layout = buildLayout();
    this.resolution = resolution;
    const { runs } = net;

    // Activations normalised for display (0..1) per layer, for each of the three runs.
    this.display = ['trained', 'untrained', 'scribble'].reduce((acc, key) => {
      acc[key] = runs[key].map((layer, li) => {
        if (li === 0 || li === 3) return layer;
        const ref = Math.max(...runs.trained[li]) || 1;
        return Float32Array.from(layer, (v) => Math.min(v / ref, 1.3));
      });
      return acc;
    }, {});

    // The neuron we zoom in on in chapter 3: the most active first-layer neuron.
    const h1 = runs.trained[1];
    this.focusIndex = h1.indexOf(Math.max(...h1));

    // Pixel reveal order for beat 2.4: top-to-bottom, roughly the order a 7 is drawn.
    this.revealRank = new Float32Array(784);
    {
      const order = [...Array(784).keys()].sort((a, b) => {
        const ra = Math.floor(a / 28);
        const rb = Math.floor(b / 28);
        return ra - rb || (a % 28) - (b % 28);
      });
      order.forEach((idx, rank) => { this.revealRank[idx] = rank / 783; });
    }
    this.pixelPhase = Float32Array.from({ length: 784 }, (_, i) => (Math.sin(i * 12.9898) * 43758.5453) % (Math.PI * 2));

    this.nodes = this._buildNodes(scene);
    this.lineTargets = buildUnrolledLine();
    this._unroll = 0; // last unroll amount written to the pixel matrices
    this.connections = this._buildConnections(scene);
    this.labels = this._buildLabels(scene);
    this.timelapse = null;
    this._segA = -1; // which snapshots are currently in the aW buffers (see _applyWeightSegment)
    this._segB = -1;
  }

  /**
   * Attach time-lapse data (beats 7.12-7.14). Precomputes, per training snapshot, the
   * normalised weight of every drawn connection and the display activations, so the
   * per-frame work is just copying between preallocated arrays when a checkpoint changes.
   */
  setTimelapse(tl) {
    const K = tl.snapshots.length;
    this.timelapse = tl;
    this.snapWeights = []; // [k][li] -> Float32Array per drawn connection
    for (let k = 0; k < K; k++) {
      // Same normalisation as the endpoints in _buildConnections: untrained x0.7, trained x1,
      // intermediate checkpoints in between.
      const factor = 0.7 + 0.3 * (k / (K - 1));
      this.snapWeights.push(this.connections.map((c, li) => {
        const W = tl.snapshots[k].layers[li].W;
        const scale = percentileAbs(W, 0.995);
        return Float32Array.from(c.indices, (idx) => Math.max(-1, Math.min(1, W[idx] / scale)) * factor);
      }));
    }
    this.snapActs = tl.heroRuns.map((run) => [1, 2].map((li) => {
      const ref = Math.max(...this.net.runs.trained[li]) || 1;
      return Float32Array.from(run[li], (v) => Math.min(v / ref, 1.3));
    }));
    this._segA = this._segB = -1;
  }

  /** Load snapshots a and b into the connection weight buffers (x and y of aW). */
  _applyWeightSegment(a, b) {
    if (a === this._segA && b === this._segB) return;
    this._segA = a;
    this._segB = b;
    for (let li = 0; li < 3; li++) {
      const { aW } = this.connections[li];
      const arr = aW.array;
      const wa = this.snapWeights[a][li];
      const wb = this.snapWeights[b][li];
      for (let i = 0, n = wa.length; i < n; i++) {
        arr[i * 2] = wa[i];
        arr[i * 2 + 1] = wb[i];
      }
      aW.needsUpdate = true;
    }
  }

  _buildNodes(scene) {
    const radii = [0.36, 0.95, 1.0, 1.45];
    const bases = [
      new Color(0.035, 0.07, 0.12),
      new Color(0.05, 0.08, 0.3),
      new Color(0.09, 0.06, 0.3),
      new Color(0.05, 0.1, 0.25),
    ];
    return this.layout.map((pos, li) => {
      const count = SIZES[li];
      const geometry = new IcosahedronGeometry(radii[li], li === 0 ? 1 : QUALITY.sphereDetail);
      const glow = new InstancedBufferAttribute(new Float32Array(count * 4), 4);
      glow.setUsage(35048); // DynamicDrawUsage
      geometry.setAttribute('aGlow', glow);
      const material = new ShaderMaterial({
        uniforms: { ...fogUniforms, uBase: { value: bases[li] }, uGrey: { value: 0 } },
        vertexShader: NODE_VERT,
        fragmentShader: NODE_FRAG,
      });
      const mesh = new InstancedMesh(geometry, material, count);
      for (let i = 0; i < count; i++) {
        _m.makeTranslation(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        mesh.setMatrixAt(i, _m);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.frustumCulled = false;
      scene.add(mesh);
      return { mesh, glow, material, count };
    });
  }

  _buildConnections(scene) {
    const { params } = this.net;
    const result = [];
    for (let li = 0; li < 3; li++) {
      const nIn = SIZES[li];
      const nOut = SIZES[li + 1];
      const Wt = params.trained[li].W;
      const Wu = params.untrained[li].W;
      const st = percentileAbs(Wt, 0.995);
      const su = percentileAbs(Wu, 0.995);
      const srcT = this.display.trained[li];
      const srcU = this.display.untrained[li];

      // On mobile, draw only the strongest input connections.
      let indices = [...Array(nIn * nOut).keys()];
      if (li === 0 && QUALITY.inputConnectionFraction < 1) {
        indices.sort((a, b) => Math.abs(Wt[b]) - Math.abs(Wt[a]));
        indices = indices.slice(0, Math.floor(indices.length * QUALITY.inputConnectionFraction));
      }
      const n = indices.length;
      const aStart = new Float32Array(n * 3);
      const aEnd = new Float32Array(n * 3);
      const aW = new Float32Array(n * 2);
      const aSrc = new Float32Array(n * 2);
      const aDst = new Float32Array(n);
      const aRand = new Float32Array(n);
      const from = this.layout[li];
      const to = this.layout[li + 1];
      indices.forEach((k, idx) => {
        const i = Math.floor(k / nOut);
        const j = k % nOut;
        aStart.set(from.subarray(i * 3, i * 3 + 3), idx * 3);
        aEnd.set(to.subarray(j * 3, j * 3 + 3), idx * 3);
        aW[idx * 2] = Math.max(-1, Math.min(1, Wt[k] / st));
        aW[idx * 2 + 1] = Math.max(-1, Math.min(1, Wu[k] / su)) * 0.7;
        aSrc[idx * 2] = Math.min(srcT[i], 1);
        aSrc[idx * 2 + 1] = Math.min(srcU[i], 1);
        aDst[idx] = j;
        aRand[idx] = (Math.sin(k * 78.233) * 43758.5453) % 1;
      });

      const geometry = new InstancedBufferGeometry();
      geometry.setAttribute('corner', new BufferAttribute(new Float32Array([0, -1, 1, -1, 1, 1, 0, 1]), 2));
      geometry.setIndex([0, 1, 2, 0, 2, 3]);
      geometry.setAttribute('aStart', new InstancedBufferAttribute(aStart, 3));
      geometry.setAttribute('aEnd', new InstancedBufferAttribute(aEnd, 3));
      geometry.setAttribute('aW', new InstancedBufferAttribute(aW, 2));
      geometry.setAttribute('aSrc', new InstancedBufferAttribute(aSrc, 2));
      geometry.setAttribute('aDst', new InstancedBufferAttribute(aDst, 1));
      geometry.setAttribute('aRand', new InstancedBufferAttribute(aRand, 1));
      geometry.instanceCount = n;

      const material = new ShaderMaterial({
        uniforms: {
          ...fogUniforms,
          uResolution: { value: this.resolution },
          uPixelRatio: { value: 1 },
          uNear: { value: 0.1 },
          uMix: { value: 0 },
          uWidth: { value: 1 },
          uFocus: { value: -1 },
          uFocusMix: { value: 0 },
          uOpacity: { value: 0 },
          uLit: { value: 0 },
          uSign: { value: 1 },
          uPulse: { value: 0 },
          uTime: { value: 0 },
          uGrey: { value: 0 },
          uBack: { value: 0 },
          uBackPos: { value: 2 },
          uEndFade: { value: li === 0 ? 1 : 0.8 },
          // Mobile keeps only the strongest input threads, so each one is drawn dimmer.
          uBoost: { value: li === 0 ? 0.11 * Math.min(1, QUALITY.inputConnectionFraction * 2) : 1.0 },
          uPos: { value: PALETTE.cyan },
          uNeg: { value: PALETTE.violet },
          uAmber: { value: PALETTE.amber },
          uRed: { value: PALETTE.red },
        },
        vertexShader: CONN_VERT,
        fragmentShader: CONN_FRAG,
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
      });
      const mesh = new Mesh(geometry, material);
      mesh.frustumCulled = false;
      mesh.renderOrder = 1;
      scene.add(mesh);
      result.push({ mesh, material, count: n, indices, aW: geometry.getAttribute('aW') });
    }
    return result;
  }

  _buildLabels(scene) {
    const out = this.layout[3];
    const sprites = [];
    for (let k = 0; k < 10; k++) {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#dff4ff';
      ctx.font = '600 84px "Inter", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(k), 64, 68);
      const tex = new CanvasTexture(canvas);
      tex.colorSpace = SRGBColorSpace;
      const sprite = new Sprite(new SpriteMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0 }));
      sprite.position.set(out[k * 3], out[k * 3 + 1] + 2.9, out[k * 3 + 2]);
      sprite.scale.setScalar(2.2);
      scene.add(sprite);
      sprites.push(sprite);
    }
    return sprites;
  }

  /**
   * Move the 784 existing pixel instances between the grid (u = 0) and the line (u = 1).
   * Rows leave top-first, a few in flight at once, each on a shallow arc toward the
   * camera. Only runs while u changes, so it costs nothing outside beat 2.7.
   */
  _applyUnroll(u, lift) {
    if (u === this._unroll) return;
    this._unroll = u;
    const { mesh } = this.nodes[0];
    const grid = this.layout[0];
    const line = this.lineTargets;
    const span = 28 + ROWS_IN_FLIGHT;
    for (let i = 0; i < 784; i++) {
      const row = Math.floor(i / 28);
      let t = (u * span - row) / ROWS_IN_FLIGHT;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const e = t * t * (3 - 2 * t);
      const k = i * 3;
      const arc = Math.sin(Math.PI * e) * 3 * lift;
      _m.makeTranslation(
        grid[k] + (line[k] - grid[k]) * e,
        grid[k + 1] + (line[k + 1] - grid[k + 1]) * e,
        grid[k + 2] + (line[k + 2] - grid[k + 2]) * e + arc,
      );
      mesh.setMatrixAt(i, _m);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  /** Position of neuron `i` in layer `li` (returns a view into the layout array). */
  position(li, i) {
    return this.layout[li].subarray(i * 3, i * 3 + 3);
  }

  /**
   * Push the frame's state into GPU buffers and uniforms.
   * @param {object} s  state from Director
   */
  update(s, camera, dpr) {
    const amber = PALETTE.amber;
    const cyan = PALETTE.cyan;
    const D = this.display;

    /*
     * Which weights the connections show. Normally x = trained, y = untrained, mixed by
     * weightMix. During the training time-lapse, x and y are two neighbouring real
     * checkpoints and the mix is the position between them.
     */
    const tl = this.timelapse;
    let mix = s.weightMix;
    if (tl) {
      const last = tl.snapshots.length - 1;
      if (s.curveMode) {
        this._applyWeightSegment(s.curveIdx, Math.min(s.curveIdx + 1, last));
        mix = s.curveFrac;
      } else {
        this._applyWeightSegment(last, 0);
      }
    } else if (s.curveMode) {
      mix = 1 - s.curveT; // data not loaded yet: fall back to a straight blend
    }

    /* --- input pixels --- */
    this._applyUnroll(s.unroll, s.unrollLift);
    {
      const { glow } = this.nodes[0];
      const g = glow.array;
      const hero = this.net.images.hero;
      const scrib = this.net.images.scribble;
      const vis = s.vis[0];
      const revealEdge = s.inputReveal * 1.08;
      const streamOn = tl && s.streamMix > 0;
      const bytes = streamOn ? tl.bytes : null;
      const set = s.streamSet === 1 ? tl?.test : tl?.train;
      const sA = streamOn ? set.offset + s.streamA * 784 : 0;
      const sB = streamOn ? set.offset + s.streamB * 784 : 0;
      const sP = streamOn ? set.offset + s.streamPrev * 784 : 0;
      for (let i = 0; i < 784; i++) {
        const reveal = Math.min(Math.max((revealEdge - this.revealRank[i]) / 0.08, 0), 1);
        let v = hero[i] * reveal * (1 - s.scribble) + scrib[i] * s.scribble;
        if (i === s.demoPixelIndex) v = Math.max(v, s.demoPixel);
        // 7.12 / 7.14: real digits streaming through the grid (crossfade + faint trail = blur).
        if (streamOn) {
          let d = bytes[sA + i] * (1 - s.streamF) + bytes[sB + i] * s.streamF + bytes[sP + i] * s.streamTrail;
          d = d > 255 ? 1 : d / 255;
          v += (d - v) * s.streamMix;
        }
        if (s.jitter > 0 && v > 0) v *= 1 + s.jitter * 0.45 * Math.sin(s.time * 2.3 + this.pixelPhase[i]);
        v *= s.act[0];
        // Lit pixels glow a little brighter while the grid is unrolled, so they stay legible far down the line.
        if (s.unroll > 0) v *= 1 + 0.6 * s.unroll;
        const k = i * 4;
        g[k] = PALETTE.pixel.r * v * 0.75;
        g[k + 1] = PALETTE.pixel.g * v * 0.75;
        g[k + 2] = PALETTE.pixel.b * v * 0.75;
        g[k + 3] = vis * (0.75 + 0.45 * v);
      }
      glow.needsUpdate = true;
    }

    /* --- hidden and output layers --- */
    for (let li = 1; li < 4; li++) {
      const { glow, count } = this.nodes[li];
      const g = glow.array;
      const vis = s.vis[li];
      const t = D.trained[li];
      const u = D.untrained[li];
      const sc = D.scribble[li];
      for (let j = 0; j < count; j++) {
        let a;
        if (li === 3) {
          a = s.outProbs[j];
        } else if (tl && s.curveMode) {
          const A = this.snapActs[this._segA][li - 1];
          const B = this.snapActs[this._segB][li - 1];
          a = A[j] + (B[j] - A[j]) * mix;
        } else {
          a = t[j] + (u[j] - t[j]) * mix;
          a += (sc[j] - a) * s.scribble;
        }
        a *= s.act[li];
        let size = vis;
        let warm = a;
        if (li === 1 && s.focus > 0) {
          const isFocus = j === this.focusIndex;
          if (isFocus) {
            warm = Math.max(a * (1 - s.focus), s.focusFire);
            size = vis * (1 + 0.25 * s.focus);
          } else {
            warm *= 1 - s.focus;
            size = vis * (1 - 0.55 * s.focus);
          }
          if (isFocus && s.focusWaver > 0) warm += s.focusWaver * (0.3 + 0.25 * Math.sin(s.time * 3.1));
        }
        if (li === 3) {
          warm *= 0.75;
          if (j === 7) {
            warm += s.out7Pulse;
            size *= 1 + 0.35 * s.out7Highlight + 0.2 * s.out7Pulse;
          }
          size *= 1 + 0.25 * a;
        }
        const k = j * 4;
        // Dim nodes glow faintly cyan; active ones run warm amber (reserved for firing).
        const cool = 0.08 * vis;
        const amberAmt = Math.min(warm, 1.4) * 1.0;
        g[k] = cyan.r * cool + amber.r * amberAmt;
        g[k + 1] = cyan.g * cool + amber.g * amberAmt;
        g[k + 2] = cyan.b * cool + amber.b * amberAmt;
        g[k + 3] = size;
      }
      glow.needsUpdate = true;
    }

    for (const n of this.nodes) n.material.uniforms.uGrey.value = s.grey;

    /* --- connections --- */
    for (let li = 0; li < 3; li++) {
      const c = s.conn[li];
      const u = this.connections[li].material.uniforms;
      u.uOpacity.value = c.opacity;
      u.uLit.value = c.lit;
      u.uSign.value = c.sign;
      u.uPulse.value = c.pulse;
      u.uWidth.value = c.width;
      u.uMix.value = mix;
      u.uTime.value = s.time;
      u.uGrey.value = s.grey;
      u.uFocus.value = li === 0 ? this.focusIndex : -1;
      u.uFocusMix.value = li === 0 ? s.focus : 0;
      u.uBack.value = c.back;
      u.uBackPos.value = c.backPos;
      u.uPixelRatio.value = dpr;
      u.uNear.value = camera.near;
      this.connections[li].mesh.visible = c.opacity > 0.001;
    }

    /* --- output labels --- */
    for (let k = 0; k < 10; k++) {
      this.labels[k].material.opacity = s.labels * (0.45 + 0.55 * Math.min(s.outProbs[k] * s.act[3] * 2, 1));
      this.labels[k].visible = s.labels > 0.001;
    }
  }
}

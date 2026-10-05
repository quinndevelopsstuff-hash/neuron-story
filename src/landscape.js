/**
 * Beats 7.5-7.8: the loss landscape.
 *
 * A procedural terrain (a few summed Gaussians: one deep valley, some hills, a gentle
 * swell) drawn as glowing cyan contour lines and a faint wireframe on near-black. A
 * red-orange marker (the loss) sits high on a slope, then walks downhill by actual
 * gradient descent on this surface: each step is learning-rate x slope, so steps shrink
 * by themselves as the ground flattens. The path is computed once at startup.
 *
 * Honest caveat (the narration says it too): a real network's loss landscape has
 * ~26,000 dimensions, not two. This one is illustrative, not computed from the network.
 */
import {
  AdditiveBlending,
  ArrowHelper,
  BufferAttribute,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  IcosahedronGeometry,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  NormalBlending,
  PlaneGeometry,
  Points,
  PointsMaterial,
  ShaderMaterial,
  Vector2,
  Vector3,
} from 'three';
import { FOG_GLSL, fogUniforms } from './scene.js';
import { PALETTE } from './network.js';
import { QUALITY } from './config.js';

const SIZE = 170;                         // terrain extent (world units)
const CENTER = new Vector3(0, -22, -60);  // world position of terrain origin
const START = [-24, -12];                 // marker start (local u, v): high on a slope
const LEARNING_RATE = 9;
const STEPS = 10;
const TRAIL_SUB = 8;                      // trail samples per step, so it hugs the ground
const SPOT_RADIUS = 6;                    // visible circle around the marker in 7.6

/** [amplitude, u, v, sigma]: one deep valley, hills, a small decoy dip. */
const BUMPS = [[-26, 20, 10, 22], [14, -30, -25, 16], [9, -40, 35, 14], [7, 45, -40, 13], [-6, -5, 45, 9], [5, -5, -55, 12]];

/** Terrain height at local (u, v). Pure function: no allocation. */
export function terrainHeight(u, v) {
  let s = 0;
  for (let i = 0; i < BUMPS.length; i++) {
    const b = BUMPS[i];
    const du = u - b[1];
    const dv = v - b[2];
    s += b[0] * Math.exp(-(du * du + dv * dv) / (2 * b[3] * b[3]));
  }
  return s + 1.4 * Math.sin(u * 0.11) * Math.cos(v * 0.09) + 0.0022 * (u * u + v * v);
}

/** The 7.8 ripple, shared by the shader (same formula) and the marker. */
function ripple(u, v, amount, phase) {
  return amount * Math.sin(Math.sqrt(u * u + v * v) * 0.12 - phase) * 1.6;
}

const TERRAIN_VERT = /* glsl */ `
  attribute float aH;
  uniform float uRise;
  uniform float uScale;
  uniform float uOffsetY;
  uniform float uRipple;
  uniform float uPhase;
  uniform vec2 uMarker;
  varying float vH;
  varying vec2 vUV;
  varying float vDist;
  varying float vDepth;
  void main() {
    vec3 p = position;
    float rip = uRipple * sin(length(p.xz) * 0.12 - uPhase) * 1.6;
    float h = aH * uScale * uRise + rip;
    p.y = h + uOffsetY;
    vH = aH * uScale + rip;
    vUV = p.xz;
    vDist = length(p.xz - uMarker);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vDepth = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const TERRAIN_FRAG = /* glsl */ `
  ${FOG_GLSL}
  uniform float uAlpha;
  uniform float uFill;
  uniform float uTight;
  uniform float uSpot;
  uniform float uRise;
  uniform vec3 uLine;
  varying float vH;
  varying vec2 vUV;
  varying float vDist;
  varying float vDepth;
  void main() {
    // Contour lines every 2 units of height, anti-aliased with screen-space derivatives.
    float c = vH / 2.0;
    float contour = 1.0 - min(abs(fract(c - 0.5) - 0.5) / max(fwidth(c), 1e-4), 1.0);
    // Faint wireframe grid.
    vec2 g = vUV / 6.0;
    vec2 gd = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-4));
    float grid = 1.0 - min(min(gd.x, gd.y), 1.0);
    // Lower ground glows a little more, so the valley reads as the destination.
    float depthGlow = clamp(-vH / 26.0, 0.0, 1.0);
    vec3 col = vec3(0.008, 0.016, 0.03) * uFill
      + uLine * (contour * (0.55 + 0.6 * depthGlow) * mix(0.35, 1.0, uRise) + grid * 0.12);
    col = mix(col, uFogColor, fogFactor(vDepth));
    // Thick fog: only a small circle around the marker stays visible.
    float spot = 1.0 - smoothstep(uSpot, uSpot * 1.7, vDist);
    col = mix(col, uFogColor, uTight * (1.0 - spot));
    gl_FragColor = vec4(col, uAlpha);
  }
`;

const _p = new Vector3();
const _dir = new Vector3();

export class Landscape {
  constructor(scene) {
    const seg = QUALITY.sphereDetail === 1 ? 64 : 96; // modest grid; lighter on mobile
    const geometry = new PlaneGeometry(SIZE, SIZE, seg, seg);
    geometry.rotateX(-Math.PI / 2); // lie flat in xz
    const pos = geometry.getAttribute('position');
    const aH = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) aH[i] = terrainHeight(pos.getX(i), pos.getZ(i));
    geometry.setAttribute('aH', new Float32BufferAttribute(aH, 1));

    const uniforms = (scale, offsetY, fill) => ({
      ...fogUniforms,
      uAlpha: { value: 0 },
      uFill: { value: fill },
      uTight: { value: 0 },
      uSpot: { value: SPOT_RADIUS },
      uRise: { value: 0 },
      uScale: { value: scale },
      uOffsetY: { value: offsetY },
      uRipple: { value: 0 },
      uPhase: { value: 0 },
      uMarker: { value: new Vector2(START[0], START[1]) },
      uLine: { value: new Color(0.22, 0.75, 1.0) },
    });
    const make = (scale, offsetY, fill, additive) => {
      const mat = new ShaderMaterial({
        uniforms: uniforms(scale, offsetY, fill),
        vertexShader: TERRAIN_VERT,
        fragmentShader: TERRAIN_FRAG,
        transparent: true,
        depthWrite: !additive,
        blending: additive ? AdditiveBlending : NormalBlending,
      });
      const mesh = new Mesh(geometry, mat);
      mesh.position.copy(CENTER);
      mesh.frustumCulled = false;
      mesh.visible = false;
      scene.add(mesh);
      return mesh;
    };
    this.terrain = make(1, 0, 1, false);
    this.terrain.renderOrder = -2; // draw the ground first, so the marker and trail sit on top
    // 7.8: faint "other slices" of a much higher-dimensional surface.
    this.ghosts = [make(0.6, 10, 0, true), make(1.4, -9, 0, true)];
    for (const g of this.ghosts) g.renderOrder = -1;

    /* Gradient descent path on this terrain (computed once). */
    this.path = new Float32Array((STEPS + 1) * 2);
    {
      let u = START[0];
      let v = START[1];
      this.path[0] = u;
      this.path[1] = v;
      for (let k = 1; k <= STEPS; k++) {
        const [gu, gv] = this._grad(u, v);
        u -= LEARNING_RATE * gu;
        v -= LEARNING_RATE * gv;
        this.path[k * 2] = u;
        this.path[k * 2 + 1] = v;
      }
    }

    /* The marker: the loss, in red-orange. */
    const red = PALETTE.red;
    this.marker = new Mesh(
      new IcosahedronGeometry(1.1, 2),
      new MeshBasicMaterial({ color: red.clone().multiplyScalar(1.8), transparent: true, opacity: 0 }),
    );
    this.halo = new Mesh(
      new IcosahedronGeometry(2.2, 2),
      new MeshBasicMaterial({ color: red.clone().multiplyScalar(0.5), transparent: true, opacity: 0, blending: AdditiveBlending, depthWrite: false }),
    );
    this.marker.add(this.halo);
    this.marker.visible = false;
    scene.add(this.marker);

    // Trail: the path so far, sampled along the surface and lifted slightly above it.
    this.trailPos = new Float32Array((STEPS * TRAIL_SUB + 1) * 3);
    const tg = new BufferGeometry();
    // BufferAttribute wraps the array (Float32BufferAttribute would copy it), so per-frame writes reach the GPU.
    tg.setAttribute('position', new BufferAttribute(this.trailPos, 3));
    this.trail = new Line(tg, new LineBasicMaterial({ color: red, transparent: true, opacity: 0, depthWrite: false }));
    this.trail.frustumCulled = false;
    this.trail.visible = false;
    scene.add(this.trail);
    // Footprints: a small dot at every point the marker has paused on.
    this.footPos = new Float32Array((STEPS + 1) * 3);
    const fg = new BufferGeometry();
    fg.setAttribute('position', new BufferAttribute(this.footPos, 3));
    this.footprints = new Points(fg, new PointsMaterial({
      color: red, size: 6, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false, depthTest: false,
    }));
    this.footprints.frustumCulled = false;
    this.footprints.visible = false;
    scene.add(this.footprints);

    // "Feel which way the ground slopes": a short arrow pointing downhill.
    this.arrow = new ArrowHelper(new Vector3(1, 0, 0), new Vector3(), 6, red.getHex(), 1.6, 1.0);
    for (const m of [this.arrow.line.material, this.arrow.cone.material]) {
      m.transparent = true;
      m.opacity = 0;
      m.depthWrite = false;
    }
    this.arrow.visible = false;
    scene.add(this.arrow);
    for (const o of [this.marker, this.halo, this.trail, this.footprints, this.arrow.line, this.arrow.cone]) o.renderOrder = 3;
  }

  /** Numerical gradient of the terrain (returns a reused array). */
  _grad(u, v) {
    const e = 0.01;
    this._g ??= [0, 0];
    this._g[0] = (terrainHeight(u + e, v) - terrainHeight(u - e, v)) / (2 * e);
    this._g[1] = (terrainHeight(u, v + e) - terrainHeight(u, v - e)) / (2 * e);
    return this._g;
  }

  /** World position of local (u, v) on the displayed surface, into `out`. */
  _surface(u, v, s, lift, out) {
    const h = terrainHeight(u, v) * s.rise + ripple(u, v, s.ripple, s.phase);
    return out.set(u + CENTER.x, h + CENTER.y + lift, v + CENTER.z);
  }

  update(state) {
    const s = state.land;
    const on = s.alpha > 0.003;
    this.terrain.visible = on;
    for (const g of this.ghosts) g.visible = on && s.ghost > 0.003;
    this.marker.visible = this.trail.visible = this.footprints.visible = on && s.marker > 0.003;
    this.arrow.visible = on && s.arrow > 0.003;
    if (!on) return;

    /* Marker: step k moves during the first 60% of its slot, then pauses. */
    const x = Math.min(Math.max(s.step, 0), STEPS);
    const k = Math.min(Math.floor(x), STEPS - 1);
    const local = x - k;
    const m = Math.min(local / 0.6, 1);
    const e = m * m * (3 - 2 * m);
    const P = this.path;
    const u = P[k * 2] + (P[k * 2 + 2] - P[k * 2]) * (x >= STEPS ? 1 : e);
    const v = P[k * 2 + 1] + (P[k * 2 + 3] - P[k * 2 + 1]) * (x >= STEPS ? 1 : e);
    this._surface(u, v, s, 1.1, _p);
    this.marker.position.copy(_p);
    this.marker.material.opacity = s.marker;
    this.halo.material.opacity = 0.35 * s.marker;

    /* Trail along the ground from the start to the marker; footprints where it paused. */
    const done = x >= STEPS ? STEPS : k;
    const progress = x >= STEPS ? STEPS : k + e; // how far along the path, in steps
    const T = this.trailPos;
    const samples = Math.ceil(progress * TRAIL_SUB);
    for (let i = 0; i <= samples; i++) {
      const t = Math.min(i / TRAIL_SUB, progress);
      const j = Math.min(Math.floor(t), STEPS - 1);
      const f = t - j;
      const tu = P[j * 2] + (P[j * 2 + 2] - P[j * 2]) * f;
      const tv = P[j * 2 + 1] + (P[j * 2 + 3] - P[j * 2 + 1]) * f;
      this._surface(tu, tv, s, 0.5, _dir);
      T[i * 3] = _dir.x;
      T[i * 3 + 1] = _dir.y;
      T[i * 3 + 2] = _dir.z;
    }
    this.trail.geometry.setDrawRange(0, samples + 1);
    this.trail.geometry.getAttribute('position').needsUpdate = true;
    const F = this.footPos;
    for (let i = 0; i <= done; i++) {
      this._surface(P[i * 2], P[i * 2 + 1], s, 0.5, _dir);
      F[i * 3] = _dir.x;
      F[i * 3 + 1] = _dir.y;
      F[i * 3 + 2] = _dir.z;
    }
    this.footprints.geometry.setDrawRange(0, done + 1);
    this.footprints.geometry.getAttribute('position').needsUpdate = true;
    this.trail.material.opacity = 0.7 * s.marker * Math.min(s.step / 0.5, 1);
    this.footprints.material.opacity = 0.85 * s.marker * Math.min(s.step / 0.5, 1);

    /* Downhill arrow, shown while "feeling" the slope (fades while a step is moving). */
    const [gu, gv] = this._grad(u, v);
    const mag = Math.hypot(gu, gv);
    if (mag > 1e-4) {
      _dir.set(-gu, -mag * mag, -gv).normalize(); // downhill along the surface
      this.arrow.setDirection(_dir);
    }
    this.arrow.position.copy(_p);
    const moving = local < 0.6 && s.step > 0 && x < STEPS ? Math.sin(Math.PI * e) : 0;
    const a = s.arrow * (1 - moving) * Math.min(mag * 3, 1); // flat ground: no arrow
    this.arrow.line.material.opacity = 0.8 * a;
    this.arrow.cone.material.opacity = 0.8 * a;

    /* Terrain uniforms. */
    const meshes = [this.terrain, ...this.ghosts];
    for (let i = 0; i < meshes.length; i++) {
      const U = meshes[i].material.uniforms;
      U.uAlpha.value = i === 0 ? s.alpha : s.alpha * s.ghost * 0.32;
      U.uRise.value = s.rise;
      U.uTight.value = i === 0 ? s.tight : 0;
      U.uRipple.value = s.ripple * (i === 0 ? 1 : 1.6);
      U.uPhase.value = s.phase + i * 1.7;
      U.uMarker.value.x = u;
      U.uMarker.value.y = v;
    }
  }
}

export const LANDSCAPE = { CENTER, START, STEPS, terrainHeight };

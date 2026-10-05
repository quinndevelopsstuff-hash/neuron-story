/**
 * Scene setup: renderer, camera, starfield, shared fog and bloom post-processing.
 *
 * Custom shaders in this project do their own fog using `fogUniforms`, so one object
 * controls depth fog for every layer of the network.
 */
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  HalfFloatType,
  PerspectiveCamera,
  Points,
  ACESFilmicToneMapping,
  Scene,
  ShaderMaterial,
  SRGBColorSpace,
  Sprite,
  SpriteMaterial,
  Vector2,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { QUALITY } from './config.js';

/** Fog shared by every custom shader (exponential-squared, like THREE.FogExp2). */
export const fogUniforms = {
  uFogColor: { value: new Color(0x010205) },
  uFogDensity: { value: 0.004 },
};

/** GLSL snippet: fog factor from view-space depth. */
export const FOG_GLSL = /* glsl */ `
  uniform vec3 uFogColor;
  uniform float uFogDensity;
  float fogFactor(float depth) {
    float d = uFogDensity * depth;
    return 1.0 - exp(-d * d);
  }
`;

/** Network centre: the "faint light" of chapter 1 sits here. */
export const NETWORK_CENTER_Z = -50;

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new WebGLRenderer({
      canvas,
      antialias: false, // MSAA happens on the composer's render target instead
      powerPreference: 'high-performance',
      alpha: false,
      stencil: false,
    });
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = SRGBColorSpace;

    this.scene = new Scene();
    this.background = new Color(0x010205);
    this.scene.background = this.background;

    this.camera = new PerspectiveCamera(55, 1, 0.1, 4000);
    this.camera.position.set(0, 0, 420);

    this.dpr = Math.min(window.devicePixelRatio || 1, QUALITY.maxDpr);
    this.resolution = new Vector2(1, 1); // drawing-buffer size, used by line shaders

    // Post-processing: render -> bloom -> tone mapping / sRGB output.
    const target = new WebGLRenderTarget(1, 1, { type: HalfFloatType, samples: QUALITY.msaa });
    this.composer = new EffectComposer(this.renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new Vector2(256, 256), 1.0, 0.55, 0.28);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.stars = createStarfield(QUALITY.stars);
    this.scene.add(this.stars);

    this.beacon = createBeacon();
    this.scene.add(this.beacon);

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  /** Lower or raise resolution at runtime (adaptive quality). */
  setDpr(dpr) {
    const clamped = Math.max(QUALITY.minDpr, Math.min(dpr, QUALITY.maxDpr));
    if (Math.abs(clamped - this.dpr) < 0.01) return;
    this.dpr = clamped;
    this.resize();
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(this.dpr);
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.dpr);
    this.composer.setSize(w, h);
    // Bloom is soft anyway; render it at reduced resolution on weaker devices.
    this.bloom.setSize(w * this.dpr * QUALITY.bloomScale, h * this.dpr * QUALITY.bloomScale);
    this.resolution.set(w * this.dpr, h * this.dpr);

    // Keep a similar horizontal field of view on portrait screens so the network fits.
    const aspect = w / h;
    const hfov = 62 * (Math.PI / 180);
    const vfov = 2 * Math.atan(Math.tan(hfov / 2) / aspect) * (180 / Math.PI);
    this.camera.fov = Math.min(Math.max(vfov, 50), 88);
    this.camera.aspect = aspect;
    // Shift the view so the centre of interest sits above the narration panel.
    this.camera.setViewOffset(w, h, 0, h * 0.08, w, h);
    this.camera.updateProjectionMatrix();
    this.stars.material.uniforms.uPixelRatio.value = this.dpr;
  }

  /** Apply the current chapter mood (background, fog, bloom). */
  applyMood(mood) {
    this.background.copy(mood.bg);
    fogUniforms.uFogColor.value.copy(mood.bg);
    fogUniforms.uFogDensity.value = mood.fog;
    // Mood values are authored relative; 0.6 keeps large close-up spheres from blowing out.
    this.bloom.strength = mood.bloom * 0.6;
    this.bloom.radius = mood.bloomRadius * 0.8;
  }

  render() {
    this.composer.render();
  }
}

/** A shell of faint, slowly twinkling stars around the whole scene. */
function createStarfield(count) {
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const tints = new Float32Array(count);
  // Deterministic pseudo-random so the sky is identical on every load.
  let seed = 12345;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < count; i++) {
    const u = rand() * 2 - 1;
    const theta = rand() * Math.PI * 2;
    const r = 900 + rand() * 900;
    const s = Math.sqrt(1 - u * u);
    positions[i * 3] = r * s * Math.cos(theta);
    positions[i * 3 + 1] = r * u * 0.7;
    positions[i * 3 + 2] = r * s * Math.sin(theta) + NETWORK_CENTER_Z;
    sizes[i] = 0.6 + Math.pow(rand(), 4) * 2.4;
    phases[i] = rand() * Math.PI * 2;
    tints[i] = rand();
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('aSize', new BufferAttribute(sizes, 1));
  geometry.setAttribute('aPhase', new BufferAttribute(phases, 1));
  geometry.setAttribute('aTint', new BufferAttribute(tints, 1));

  const material = new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uAlpha: { value: 0 },
      uTwinkle: { value: 1 },
      uPixelRatio: { value: 1 },
    },
    vertexShader: /* glsl */ `
      attribute float aSize;
      attribute float aPhase;
      attribute float aTint;
      uniform float uTime;
      uniform float uTwinkle;
      uniform float uPixelRatio;
      varying float vBright;
      varying float vTint;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = aSize * uPixelRatio * 1.6;
        vBright = 0.55 + 0.45 * mix(1.0, sin(uTime * 0.6 + aPhase), uTwinkle * 0.6);
        vTint = aTint;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uAlpha;
      varying float vBright;
      varying float vTint;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float a = smoothstep(0.5, 0.0, length(c));
        vec3 col = mix(vec3(0.55, 0.7, 1.0), vec3(1.0, 0.92, 0.8), step(0.85, vTint));
        gl_FragColor = vec4(col * vBright * uAlpha * 0.55 * a, 1.0);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const points = new Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

/** The single faint light of the opening: a screen-space glow at the network's centre. */
function createBeacon() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(220,240,255,1)');
  g.addColorStop(0.12, 'rgba(140,200,255,0.8)');
  g.addColorStop(0.4, 'rgba(60,120,255,0.18)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  const material = new SpriteMaterial({
    map: texture,
    blending: AdditiveBlending,
    depthWrite: false,
    depthTest: false,
    transparent: true,
    sizeAttenuation: false,
  });
  const sprite = new Sprite(material);
  sprite.position.set(0, 0, NETWORK_CENTER_Z);
  sprite.renderOrder = 10;
  sprite.scale.setScalar(0.05);
  return sprite;
}

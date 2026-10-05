/**
 * Story props: the smaller scene elements that illustrate individual beats
 * (ghost sevens, the bias dial, stroke fragments, weight pictures, the label tag,
 * the loss readout...). All are built once; update() only changes opacity/transforms.
 */
import {
  AdditiveBlending,
  CanvasTexture,
  Group,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  Sprite,
  SpriteMaterial,
  SRGBColorSpace,
  TorusGeometry,
  Vector3,
} from 'three';
import { PALETTE } from './network.js';

/** Upscaled, glowing texture from a 28x28 image (values 0..1, or signed for weights). */
function imageTexture(values, { signed = false, size = 112, tint = [150, 230, 255] } = {}) {
  const small = document.createElement('canvas');
  small.width = small.height = 28;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(28, 28);
  let scale = 1;
  if (signed) {
    const abs = Array.from(values, Math.abs).sort((a, b) => a - b);
    scale = abs[Math.floor(abs.length * 0.98)] || 1;
  }
  for (let i = 0; i < 784; i++) {
    const v = signed ? values[i] / scale : values[i];
    const a = Math.min(Math.abs(v), 1);
    const [r, g, b] = signed ? (v >= 0 ? [90, 215, 255] : [150, 95, 255]) : tint;
    img.data.set([r, g, b, Math.round(a * 255)], i * 4);
  }
  sctx.putImageData(img, 0, 0);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(small, 0, 0, size, size);
  const tex = new CanvasTexture(canvas);
  tex.colorSpace = SRGBColorSpace;
  return tex;
}

/** Texture drawn with the 2D canvas API (strokes, text, patterns). */
function drawnTexture(size, draw, w = size) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = size;
  draw(canvas.getContext('2d'), w, size);
  const tex = new CanvasTexture(canvas);
  tex.colorSpace = SRGBColorSpace;
  return tex;
}

function glowSprite(map, scale, additive = true) {
  const sprite = new Sprite(new SpriteMaterial({
    map,
    transparent: true,
    depthWrite: false,
    opacity: 0,
    blending: additive ? AdditiveBlending : undefined,
  }));
  sprite.scale.setScalar(scale);
  sprite.visible = false;
  return sprite;
}

function setAlpha(obj, a) {
  obj.material.opacity = a;
  obj.visible = a > 0.002;
}

export class Props {
  /**
   * @param {import('three').Scene} scene
   * @param {object} net      data from loadNetwork()
   * @param {import('./network.js').NetworkView} view
   * @param {Vector3} ghostAnchor camera position mid-beat 1.5
   */
  constructor(scene, net, view, ghostAnchor) {
    this.view = view;

    /* 1.5: three real MNIST sevens in different handwriting styles drift past. */
    this.ghosts = net.images.ghosts.map((pix, i) => {
      const s = glowSprite(imageTexture(pix), 9);
      const offsets = [[-11, 4, -32], [12, -2, -48], [-4, -8, -62]];
      s.userData.base = new Vector3(...offsets[i]).add(ghostAnchor);
      s.userData.drift = new Vector3(i === 1 ? -3 : 3, i === 2 ? 2 : -1, 4);
      scene.add(s);
      return s;
    });

    const focus = new Vector3(...view.position(1, view.focusIndex));
    this.focusPos = focus;

    /* 3.6: the bias, shown as a dial ring around the focus neuron. */
    this.biasRing = new Mesh(
      new TorusGeometry(1.75, 0.05, 8, 64, Math.PI * 1.7),
      new MeshBasicMaterial({ color: PALETTE.cyan.clone().multiplyScalar(1.6), transparent: true, opacity: 0, blending: AdditiveBlending, depthWrite: false }),
    );
    this.biasRing.position.copy(focus);
    scene.add(this.biasRing);

    /* 3.8: the fired neuron's signal travelling forward. */
    const sparkTex = drawnTexture(64, (ctx, w, h) => {
      const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
      g.addColorStop(0, 'rgba(255,230,180,1)');
      g.addColorStop(0.3, 'rgba(255,160,60,0.6)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    });
    this.spark = glowSprite(sparkTex, 1.6);
    this.sparkFrom = focus.clone();
    this.sparkTo = new Vector3(0, 0, -72);
    scene.add(this.spark);

    /* 5.4: stroke fragments near first-layer neurons, larger parts near second-layer ones. */
    const stroke = (angle, len) => drawnTexture(96, (ctx, w, h) => {
      ctx.translate(w / 2, h / 2);
      ctx.rotate(angle);
      ctx.strokeStyle = 'rgba(160,235,255,1)';
      ctx.shadowColor = 'rgba(80,200,255,1)';
      ctx.shadowBlur = 10;
      ctx.lineWidth = 7;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(-len / 2, 0);
      ctx.lineTo(len / 2, 0);
      ctx.stroke();
    });
    const fragmentSpecs = [
      [0, 40, 1, 2], [0.15, 36, 1, 10], [-1.1, 40, 1, 21], [-1.0, 34, 1, 13], [0.05, 30, 1, 27], [-1.2, 38, 1, 4],
      [0, 72, 2, 1], [-1.08, 80, 2, 6], [-1.15, 64, 2, 14],
    ];
    this.assembleTo = new Vector3(0, 0, -86);
    this.fragments = fragmentSpecs.map(([angle, len, layer, idx]) => {
      const s = glowSprite(stroke(angle, len), layer === 1 ? 2.8 : 4.5);
      const p = view.position(layer, idx % view.layout[layer].length);
      s.userData.base = new Vector3(p[0], p[1] + 2.2, p[2] + 1.5);
      scene.add(s);
      return s;
    });

    /* 5.5: the idealised "perfect" seven the fragments assemble into. */
    this.perfectSeven = glowSprite(drawnTexture(256, (ctx, w) => {
      ctx.strokeStyle = 'rgba(170,240,255,1)';
      ctx.shadowColor = 'rgba(80,200,255,1)';
      ctx.shadowBlur = 18;
      ctx.lineWidth = 18;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(w * 0.25, w * 0.22);
      ctx.lineTo(w * 0.75, w * 0.22);
      ctx.lineTo(w * 0.42, w * 0.82);
      ctx.stroke();
    }), 10);
    this.perfectSeven.position.copy(this.assembleTo);
    scene.add(this.perfectSeven);

    /* 5.6: what the first-layer neurons really respond to: their actual weight pictures. */
    const W = net.params.trained[0].W;
    this.weightMaps = [];
    for (let j = 0; j < 32; j++) {
      const col = new Float32Array(784);
      for (let i = 0; i < 784; i++) col[i] = W[i * 32 + j];
      const s = glowSprite(imageTexture(col, { signed: true, size: 56 }), 2.4);
      const p = view.position(1, j);
      s.position.set(p[0], p[1] + 1.9, p[2] + 0.6);
      scene.add(s);
      this.weightMaps.push(s);
    }

    /* 5.8: clean striped edge detectors, as larger image networks learn. */
    this.gabors = [0, 0.8, 1.6, 2.4].map((angle, i) => {
      const s = glowSprite(drawnTexture(96, (ctx, w, h) => {
        const img = ctx.createImageData(w, h);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const dx = (x - w / 2) / (w / 2);
            const dy = (y - h / 2) / (h / 2);
            const xr = dx * Math.cos(angle) + dy * Math.sin(angle);
            const env = Math.exp(-(dx * dx + dy * dy) * 2.2);
            const v = Math.cos(xr * 9) * env;
            const k = (y * w + x) * 4;
            img.data.set(v >= 0 ? [90, 215, 255, v * 255] : [150, 95, 255, -v * 255], k);
          }
        }
        ctx.putImageData(img, 0, 0);
      }), 3.2);
      s.position.set((i - 1.5) * 3.8, 6.8, -72);
      scene.add(s);
      return s;
    });

    /* 7.2b: the training label, a tag that flips to reveal the right answer. */
    this.tag = new Group();
    const tagFront = drawnTexture(128, (ctx, w, h) => {
      ctx.fillStyle = 'rgba(14,30,48,0.95)';
      ctx.strokeStyle = 'rgba(140,220,255,0.9)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.roundRect(4, 4, w - 8, h - 8, 14);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'rgba(190,235,255,0.95)';
      ctx.font = '600 34px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('LABEL', w / 2, h / 2);
    }, 200);
    const tagBack = drawnTexture(128, (ctx, w, h) => {
      ctx.fillStyle = 'rgba(40,26,8,0.95)';
      ctx.strokeStyle = 'rgba(255,190,90,0.95)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.roundRect(4, 4, w - 8, h - 8, 14);
      ctx.fill();
      ctx.stroke();
      // A hand-drawn 7, like an answer written on the back of a card.
      ctx.strokeStyle = 'rgba(255,215,150,1)';
      ctx.lineWidth = 8;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(w * 0.36, h * 0.26);
      ctx.bezierCurveTo(w * 0.48, h * 0.22, w * 0.58, h * 0.25, w * 0.66, h * 0.24);
      ctx.lineTo(w * 0.47, h * 0.8);
      ctx.stroke();
    }, 200);
    const plane = new PlaneGeometry(5, 3.2);
    const mk = (map, ry) => {
      const m = new Mesh(plane, new MeshBasicMaterial({ map, transparent: true, opacity: 0, depthWrite: false }));
      m.rotation.y = ry;
      m.position.z = ry ? -0.01 : 0.01;
      return m;
    };
    this.tagFront = mk(tagFront, 0);
    this.tagBack = mk(tagBack, Math.PI);
    this.tag.add(this.tagFront, this.tagBack);
    this.tag.position.set(17.5, 10, 2);
    this.tag.visible = false;
    scene.add(this.tag);

    /* 7.3: the loss, a real number: -ln(p("7")) for the untrained network. */
    const pUntrained = net.runs.untrained[3][7];
    this.lossValue = -Math.log(pUntrained);
    this.loss = new Sprite(new SpriteMaterial({
      map: drawnTexture(128, (ctx, w, h) => {
        ctx.fillStyle = 'rgba(255,120,70,1)';
        ctx.shadowColor = 'rgba(255,80,30,1)';
        ctx.shadowBlur = 16;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '500 30px system-ui, sans-serif';
        ctx.fillText('LOSS', w / 2, h * 0.24);
        ctx.font = '700 64px system-ui, sans-serif';
        ctx.fillText(this.lossValue.toFixed(2), w / 2, h * 0.64);
      }, 256),
      transparent: true,
      depthWrite: false,
      opacity: 0,
    }));
    this.loss.scale.set(8, 4, 1);
    this.loss.position.set(0, 7.5, -100);
    this.loss.visible = false;
    scene.add(this.loss);
  }

  update(s) {
    const p = s.props;

    this.ghosts.forEach((g, i) => {
      const local = Math.min(Math.max(p.ghostT * 1.3 - i * 0.15, 0), 1);
      const a = p.ghosts * Math.sin(Math.PI * local) * 0.55;
      setAlpha(g, a);
      g.position.copy(g.userData.base).addScaledVector(g.userData.drift, local);
    });

    setAlpha(this.biasRing, p.biasRing);
    this.biasRing.rotation.z = p.biasAngle;

    setAlpha(this.spark, p.spark * Math.sin(Math.PI * p.sparkT));
    this.spark.position.lerpVectors(this.sparkFrom, this.sparkTo, p.sparkT);

    this.fragments.forEach((f) => {
      setAlpha(f, p.fragments * (1 - p.assemble * 0.9));
      f.position.lerpVectors(f.userData.base, this.assembleTo, p.assemble * p.assemble);
    });
    setAlpha(this.perfectSeven, p.perfect7);
    this.weightMaps.forEach((m) => setAlpha(m, p.weightMaps));
    this.gabors.forEach((g) => setAlpha(g, p.gabors));

    this.tag.visible = p.tag > 0.002;
    this.tagFront.material.opacity = p.tag;
    this.tagBack.material.opacity = p.tag;
    this.tag.rotation.y = Math.PI * p.tagFlip;

    setAlpha(this.loss, p.loss);
  }
}

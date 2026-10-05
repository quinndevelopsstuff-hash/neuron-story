/**
 * Small HUD panels for the training time-lapse:
 *  - 7.12-7.13: our 7's real confidence at the current stage of training, with the step.
 *  - 7.14: a strip of the unseen test digits so far, each marked with the network's real
 *    verdict (cyan check = correct, red mark + its wrong answer = miss), and a tally.
 *
 * DOM, like the narration. Text and thumbnails are only rewritten when a value changes.
 */
const BATCH = 64; // training batch size, for "images seen"

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export class TimelapseHud {
  constructor(root) {
    this.readout = el('div', 'tl-hud tl-readout');
    this.readout.setAttribute('aria-hidden', 'true');
    this.readout.append(el('span', 'lbl', 'Our 7 · confidence'));
    this.pct = el('span', 'val', '');
    this.stepText = el('span', 'sub', '');
    this.readout.append(this.pct, this.stepText);

    this.tests = el('div', 'tl-hud tl-tests');
    this.tests.setAttribute('aria-hidden', 'true');
    this.tests.append(el('span', 'lbl', 'Unseen test digits'));
    this.strip = el('div', 'strip');
    this.slotCount = window.innerWidth < 520 ? 8 : 12;
    this.slots = Array.from({ length: this.slotCount }, () => {
      const slot = el('div', 'slot');
      const canvas = el('canvas');
      canvas.width = canvas.height = 28;
      const mark = el('span', 'mark');
      slot.append(canvas, mark);
      this.strip.append(slot);
      return { slot, ctx: canvas.getContext('2d'), mark, key: '' };
    });
    this.tally = el('span', 'tally', '');
    this.tests.append(this.strip, this.tally);
    root.append(this.readout, this.tests);

    this.tl = null;
    this._o = [-1, -1];
    this._pct = -1;
    this._step = -1;
    this._testKey = '';
  }

  /** Pre-render each test digit once as a 28x28 image. */
  setTimelapse(tl) {
    this.tl = tl;
    const { bytes, test } = tl;
    this.digitImages = Array.from({ length: test.count }, (_, k) => {
      const img = new ImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        const v = bytes[test.offset + k * 784 + i];
        img.data[i * 4] = 120 * v / 255;
        img.data[i * 4 + 1] = 225 * v / 255;
        img.data[i * 4 + 2] = 255 * v / 255;
        img.data[i * 4 + 3] = 255;
      }
      return img;
    });
  }

  _fade(node, i, o) {
    if (Math.abs(o - this._o[i]) < 0.005) return;
    this._o[i] = o;
    node.style.opacity = o.toFixed(3);
    node.style.visibility = o > 0.005 ? 'visible' : 'hidden';
  }

  update(s) {
    const h = s.hud;
    this._fade(this.readout, 0, h.readout);
    if (h.readout > 0.005) {
      const pct = Math.round(h.p7 * 100);
      if (pct !== this._pct) {
        this._pct = pct;
        this.pct.textContent = `${pct}%`;
      }
      // Step shown to two significant figures, so it ticks rather than flickers.
      const raw = Math.round(h.step);
      const mag = raw < 100 ? 1 : Math.pow(10, Math.floor(Math.log10(raw)) - 1);
      const step = Math.round(raw / mag) * mag;
      if (step !== this._step) {
        this._step = step;
        this.stepText.textContent = step === 0
          ? 'before training'
          : `after ${step.toLocaleString('en-US')} training steps · ${(step * BATCH).toLocaleString('en-US')} images`;
      }
    }

    this._fade(this.tests, 1, this.tl ? h.test : 0);
    if (!this.tl || h.test <= 0.005) return;
    const markOn = h.testMark > 0.5;
    const key = `${h.testCurrent}:${markOn}:${h.testCount}`;
    if (key === this._testKey) return;
    this._testKey = key;
    const { labels, preds } = this.tl.test;
    // Right-aligned window: the newest digit is the last slot.
    const newest = h.testCurrent;
    // Judged so far: every earlier digit, plus the current one once its mark is showing.
    const shown = newest + (markOn ? 1 : 0);
    let correct = 0;
    for (let k = 0; k < shown; k++) if (labels[k] === preds[k]) correct++;
    this.slots.forEach((slot, i) => {
      const k = newest - (this.slotCount - 1 - i);
      const judged = k >= 0 && (k < newest || markOn);
      const sKey = k < 0 ? 'empty' : `${k}:${judged}`;
      if (sKey === slot.key) return;
      slot.key = sKey;
      if (k < 0) {
        slot.ctx.clearRect(0, 0, 28, 28);
        slot.slot.className = 'slot empty';
        slot.mark.textContent = '';
        return;
      }
      slot.ctx.putImageData(this.digitImages[k], 0, 0);
      const ok = labels[k] === preds[k];
      slot.slot.className = `slot${judged ? (ok ? ' ok' : ' miss') : ''}${k === newest ? ' current' : ''}`;
      slot.mark.textContent = judged ? (ok ? '✓' : `✗ ${preds[k]}`) : '';
    });
    this.tally.textContent = `${correct} of ${shown} correct`;
  }
}

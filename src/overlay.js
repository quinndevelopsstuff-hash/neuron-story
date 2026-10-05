/**
 * The HTML overlay: intro screen, narration beats, chapter cards, progress bar with
 * chapter markers, and the end card.
 *
 * All elements are created once. Each frame only opacity/transform/visibility change,
 * and only when the value actually changed, so there is no layout work or layout shift.
 */
import { motion, SHOW_STAGE } from './config.js';

const FADE = 0.16; // fraction of a segment spent fading in (and out)

/** Small inline diagram for beat 3.9: a straight line vs. ReLU's bend at zero. */
const RELU_SVG = `
<svg class="relu" viewBox="0 0 220 74" role="img" aria-label="Left: a straight line. Right: ReLU, flat at zero for negative inputs, then rising.">
  <g fill="none" stroke-linecap="round">
    <line x1="8" y1="60" x2="96" y2="60" class="axis"/><line x1="52" y1="66" x2="52" y2="8" class="axis"/>
    <line x1="14" y1="66" x2="92" y2="10" class="lin"/>
    <line x1="124" y1="60" x2="212" y2="60" class="axis"/><line x1="168" y1="66" x2="168" y2="8" class="axis"/>
    <polyline points="128,60 168,60 208,14" class="hinge"/>
  </g>
</svg>`;

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

export class Overlay {
  /**
   * @param {HTMLElement} root
   * @param {object} story
   * @param {import('./timeline.js').Timeline} tl
   * @param {{onJump: (p: number) => void}} options
   */
  constructor(root, story, tl, { onJump }) {
    this.tl = tl;
    this.onJump = onJump;
    this.root = root;

    /* intro */
    this.intro = el('section', 'intro');
    this.intro.innerHTML = `
      <p class="eyebrow">A flight through a neural network</p>
      <h1>Inside the Machine</h1>
      <p class="lede">How does a machine learn to see a <span class="nowrap">handwritten <span class="seven">7</span>?</span></p>
      <p class="status" aria-live="polite">Loading the network&hellip;</p>
      <div class="hint">
        <div class="scroll-cue" aria-hidden="true"><span>Scroll to begin</span><span class="chevron"></span></div>
      </div>`;
    root.appendChild(this.intro);
    this.status = this.intro.querySelector('.status');

    /* narration + chapter cards: one element per timeline segment */
    this.narration = el('div', 'narration');
    this.narration.setAttribute('aria-hidden', 'true'); // screen readers get the article below
    root.appendChild(this.narration);
    this.items = tl.segments.map((seg) => {
      let node = null;
      if (seg.type === 'beat') {
        node = el('div', 'beat');
        node.appendChild(el('p', 'text', seg.beat.html));
        if (seg.id === '3.9') node.insertAdjacentHTML('beforeend', RELU_SVG);
        if (SHOW_STAGE) node.appendChild(el('p', 'stage', `<b>${seg.id}</b> [${seg.beat.stage}]`));
        this.narration.appendChild(node);
      } else if (seg.type === 'chapter') {
        node = el('div', 'chapter-card');
        node.innerHTML = `<span class="num">Chapter ${seg.chapterInfo.number}</span><h2>${seg.chapterInfo.title}</h2>`;
        root.appendChild(node);
      } else if (seg.type === 'end') {
        node = el('section', 'end-card');
        node.innerHTML = `
          <h2>Inside the Machine</h2>
          <p>A 784&ndash;32&ndash;16&ndash;10 network, trained on the MNIST handwritten digits.
          Every weight, activation and percentage you saw came from it.</p>
          <button type="button" class="restart">Fly again</button>`;
        node.querySelector('.restart').addEventListener('click', () => this.onJump(0));
        root.appendChild(node);
      }
      if (node) node.style.visibility = 'hidden';
      return { seg, node, opacity: -1 };
    });

    /* progress bar with chapter markers */
    const bar = el('nav', 'progress');
    bar.setAttribute('aria-label', 'Chapters');
    this.fill = el('div', 'fill');
    bar.appendChild(this.fill);
    this.markers = tl.chapters.map((ch) => {
      const b = el('button', 'marker');
      b.type = 'button';
      b.style.left = `${ch.start * 100}%`;
      b.setAttribute('aria-label', `Chapter ${ch.number}: ${ch.title}`);
      b.innerHTML = `<span class="tip">${ch.number}. ${ch.title}</span>`;
      // Land just after the chapter card so the first beat is readable.
      b.addEventListener('click', () => this.onJump(ch.start + 0.0005));
      bar.appendChild(b);
      return b;
    });
    root.appendChild(bar);

    // Copyright line, pinned to the bottom of the screen; fades in with the end card.
    // &copy; is the plain text symbol (not the emoji variant); CSS also asks for text presentation.
    this.copyright = el('footer', 'copyright', 'Copyright &copy; 2026 Quinn Koscielak');
    this.copyright.style.visibility = 'hidden';
    root.appendChild(this.copyright);

    this.chapterLabel = el('div', 'chapter-label');
    this.chapterLabel.setAttribute('aria-hidden', 'true');
    root.appendChild(this.chapterLabel);

    /* full narration for screen readers, in reading order */
    const article = el('article', 'sr-only');
    article.innerHTML = `<h1>${story.title}</h1>` + story.chapters.map((c) =>
      `<h2>Chapter ${c.number}: ${c.title}</h2>` + c.beats.map((b) => `<p>${b.html}</p>`).join('')).join('');
    root.appendChild(article);

    this._fill = -1;
    this._chapter = -2;
    this._intro = -1;
    this._active = new Set();
  }

  ready() {
    this.status.textContent = '';
    this.intro.classList.add('ready');
  }

  showError(message) {
    this.status.textContent = message;
    this.status.classList.add('error');
  }

  update(p) {
    /* intro fades as the reader starts scrolling */
    const intro = this.tl.get('intro');
    const io = 1 - Math.min(Math.max((p - intro.start) / (intro.end * 0.85), 0), 1);
    if (Math.abs(io - this._intro) > 0.002) {
      this._intro = io;
      this.intro.style.opacity = io.toFixed(3);
      this.intro.style.visibility = io > 0.001 ? 'visible' : 'hidden';
    }

    /* only the segment under p (and its neighbours, while fading) can be visible */
    const idx = this.tl.indexAt(p);
    for (const i of this._active) if (Math.abs(i - idx) > 1) this._set(this.items[i], 0);
    this._active.clear();
    for (let i = Math.max(idx - 1, 0); i <= Math.min(idx + 1, this.items.length - 1); i++) {
      const item = this.items[i];
      if (!item.node) continue;
      const { start, end } = item.seg;
      const f = (p - start) / (end - start);
      let o = 0;
      if (f >= 0 && f <= 1) {
        const fadeOut = item.seg.type === 'end' ? 1 : Math.min((1 - f) / FADE, 1);
        o = Math.min(f / FADE, 1, fadeOut);
        o = o * o * (3 - 2 * o);
      }
      this._set(item, o);
      if (o > 0) this._active.add(i);
    }

    /* progress bar and current chapter */
    if (Math.abs(p - this._fill) > 0.0002) {
      this._fill = p;
      this.fill.style.transform = `scaleX(${p.toFixed(4)})`;
    }
    const ch = this.tl.chapterAt(p);
    if (ch !== this._chapter) {
      this._chapter = ch;
      this.markers.forEach((m, i) => {
        m.classList.toggle('done', i < ch);
        m.classList.toggle('current', i === ch);
      });
      if (ch >= 0) {
        const c = this.tl.chapters[ch];
        this.chapterLabel.textContent = `${c.number} · ${c.title}`;
      }
      this.chapterLabel.classList.toggle('visible', ch >= 0);
    }
  }

  _set(item, o) {
    if (Math.abs(o - item.opacity) < 0.002) return;
    item.opacity = o;
    if (item.seg.type === 'end') {
      this.copyright.style.opacity = o.toFixed(3);
      this.copyright.style.visibility = o > 0.001 ? 'visible' : 'hidden';
    }
    const s = item.node.style;
    s.opacity = o.toFixed(3);
    s.visibility = o > 0.001 ? 'visible' : 'hidden';
    if (!motion.reduced && item.seg.type === 'beat') {
      s.transform = `translate3d(0, ${((1 - o) * 10).toFixed(2)}px, 0)`;
    } else if (s.transform) {
      s.transform = '';
    }
  }
}

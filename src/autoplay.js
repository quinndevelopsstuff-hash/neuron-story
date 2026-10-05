/**
 * Auto-play: scrolls the page at a constant speed so the story plays hands-free.
 *
 * It drives the *native* scroll position, so the existing ScrollController keeps doing
 * its usual smoothing and the camera, scene and overlay need no changes. Speed is
 * defined in "story per second" and integrated with frame delta time, so it is the same
 * at any frame rate. Any manual input pauses it immediately.
 */

/** Seconds for the whole story at 1x. */
const STORY_SECONDS = 600;
const SPEEDS = [1, 1.5, 2];

const PLAY_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';

const SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End']);

export class AutoPlay {
  /**
   * @param {HTMLElement} root  overlay root to attach the controls to
   * @param {number} stopAt     progress (0..1) at which to stop: the end card
   */
  constructor(root, stopAt) {
    this.stopAt = stopAt;
    this.playing = false;
    this.activated = false; // controls (and the space shortcut) appear after the first start
    this.speedIndex = 0;
    this.pos = 0; // fractional scroll position in px; scrollTo only takes whole pixels
    this.lastSet = -1;

    this.controls = document.createElement('div');
    this.controls.className = 'autoplay-controls';
    this.controls.innerHTML = `
      <button type="button" class="ap-toggle"></button>
      <button type="button" class="ap-speed" aria-label="Auto-play speed"></button>`;
    this.toggleBtn = this.controls.querySelector('.ap-toggle');
    this.speedBtn = this.controls.querySelector('.ap-speed');
    this.toggleBtn.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
    this.speedBtn.addEventListener('click', () => {
      this.speedIndex = (this.speedIndex + 1) % SPEEDS.length;
      this._render();
    });
    // After a mouse/touch click, drop focus so the space bar toggles playback
    // instead of re-activating whichever button was clicked last.
    this.controls.addEventListener('click', (e) => { if (e.detail > 0) e.target.closest('button')?.blur(); });
    root.appendChild(this.controls);
    this._render();

    this._listen();
  }

  /** Start from the very top (the title screen's Auto-play button). */
  start() {
    document.activeElement?.blur?.();
    window.scrollTo({ top: 0, behavior: 'instant' });
    this.pos = 0;
    this.play();
  }

  play() {
    const max = this._max();
    // Resuming at the end card restarts from the top.
    this.pos = window.scrollY >= this.stopAt * max - 2 ? 0 : window.scrollY;
    if (this.pos === 0) window.scrollTo({ top: 0, behavior: 'instant' });
    this.lastSet = Math.round(this.pos);
    this.playing = true;
    this.activated = true;
    this._render();
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    this._render();
  }

  /** Called once per frame, before the scroll controller reads the page. */
  tick(dt) {
    if (!this.playing) return;
    const max = this._max(); // recomputed: mobile toolbars change the viewport height
    const end = this.stopAt * max;
    this.pos += (max / STORY_SECONDS) * SPEEDS[this.speedIndex] * dt;
    if (this.pos >= end) {
      this.pos = end;
      this.playing = false;
      this._render();
    }
    const top = Math.round(this.pos);
    if (top !== this.lastSet) {
      this.lastSet = top;
      window.scrollTo({ top, behavior: 'instant' });
    }
  }

  _max() {
    return Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
  }

  _listen() {
    const isControl = (e) => e.target instanceof Element && e.target.closest('.autoplay-controls, .autoplay-start');
    const manual = (e) => { if (!isControl(e)) this.pause(); };

    window.addEventListener('wheel', manual, { passive: true });
    window.addEventListener('touchstart', manual, { passive: true });
    // Chapter markers, "Fly again" and any other click on the page.
    window.addEventListener('pointerdown', (e) => {
      if (e.target instanceof Element && e.target.closest('button, .progress') && !isControl(e)) this.pause();
    });
    window.addEventListener('keydown', (e) => {
      if (SCROLL_KEYS.has(e.key)) this.pause();
      if (e.key === ' ' && this.activated && !(e.target instanceof Element && e.target.closest('button, input, textarea'))) {
        e.preventDefault(); // otherwise space also scrolls the page
        this.playing ? this.pause() : this.play();
      }
    });
    // Anything else that moves the page (scrollbar drag, find-in-page...) also pauses.
    window.addEventListener('scroll', () => {
      if (this.playing && Math.abs(window.scrollY - this.lastSet) > 3) this.pause();
    }, { passive: true });
  }

  _render() {
    this.controls.classList.toggle('visible', this.activated);
    document.body.classList.toggle('autoplay-on', this.activated);
    this.toggleBtn.innerHTML = this.playing ? PAUSE_ICON : PLAY_ICON;
    this.toggleBtn.setAttribute('aria-label', this.playing ? 'Pause auto-play' : 'Resume auto-play');
    this.speedBtn.textContent = `${SPEEDS[this.speedIndex]}×`;
  }
}

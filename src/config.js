/**
 * Device capability and accessibility settings, decided once at startup.
 * Everything that scales with device power lives here so it is easy to tune.
 */

const coarsePointer = window.matchMedia('(pointer: coarse)').matches;
const smallScreen = Math.min(window.screen.width, window.screen.height) < 768;

/** True on phones and small tablets: fewer particles, lower resolution effects. */
export const isMobile = coarsePointer && smallScreen;

const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

/** Live flag: the user can toggle reduced motion while the page is open. */
export const motion = { reduced: reducedQuery.matches };
reducedQuery.addEventListener?.('change', (e) => { motion.reduced = e.matches; });

export const QUALITY = {
  /** Upper bound on devicePixelRatio. The render loop may lower it further if frames are slow. */
  maxDpr: isMobile ? 1.5 : 2,
  /** Lowest DPR the adaptive scaler will drop to. */
  minDpr: 1,
  /** Number of background stars. */
  stars: isMobile ? 1200 : 3500,
  /** Fraction of the 25,088 input->hidden connections drawn (strongest weights kept). */
  inputConnectionFraction: isMobile ? 0.3 : 1,
  /** Bloom buffer resolution relative to the canvas. */
  bloomScale: isMobile ? 0.5 : 1,
  /** MSAA samples for the main render target (0 disables). */
  msaa: isMobile ? 0 : 4,
  /** Icosahedron detail for neuron spheres. */
  sphereDetail: isMobile ? 1 : 2,
};

/** Total scroll length in viewport heights. Tuned for ~10 minutes of reading. */
export const SCROLL_VIEWPORTS = 140;

/** Show stage directions under the narration (append ?stage to the URL). */
export const SHOW_STAGE = new URLSearchParams(window.location.search).has('stage');

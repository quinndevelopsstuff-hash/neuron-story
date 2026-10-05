/**
 * Timeline: maps the story's beats onto scroll progress (0..1).
 *
 * Each beat gets a slice of the scroll length proportional to how long it takes to read,
 * so long beats stay on screen longer. Chapter title cards, the intro and the end card
 * get their own slices. Scene choreography refers to beats by id ("3.7") and asks for
 * progress values inside them, so text and visuals always stay in sync.
 */

const INTRO_WEIGHT = 9;
const CHAPTER_CARD_WEIGHT = 9;
const END_WEIGHT = 16;
/** A beat's weight is a fixed pause plus time per word. */
const beatWeight = (words) => 7 + words * 0.75;

export class Timeline {
  constructor(story) {
    /** @type {{type:string,id:string,chapter:number,start:number,end:number,beat?:object,chapterInfo?:object}[]} */
    this.segments = [];
    this.byId = new Map();
    this.chapters = [];

    const raw = [];
    raw.push({ type: 'intro', id: 'intro', chapter: 0, weight: INTRO_WEIGHT });
    story.chapters.forEach((chapter, ci) => {
      raw.push({ type: 'chapter', id: `ch${chapter.number}`, chapter: ci, weight: CHAPTER_CARD_WEIGHT, chapterInfo: chapter });
      for (const beat of chapter.beats) {
        raw.push({ type: 'beat', id: beat.id, chapter: ci, weight: beatWeight(beat.words), beat });
      }
    });
    raw.push({ type: 'end', id: 'end', chapter: story.chapters.length - 1, weight: END_WEIGHT });

    const total = raw.reduce((s, r) => s + r.weight, 0);
    let acc = 0;
    for (const r of raw) {
      const seg = { ...r, start: acc / total, end: (acc + r.weight) / total };
      acc += r.weight;
      this.segments.push(seg);
      this.byId.set(seg.id, seg);
      if (seg.type === 'chapter') {
        this.chapters.push({ number: seg.chapterInfo.number, title: seg.chapterInfo.title, start: seg.start });
      }
    }
    this._last = 0;
  }

  /** Segment by id. Throws on typos so choreography mistakes fail loudly. */
  get(id) {
    const seg = this.byId.get(id);
    if (!seg) throw new Error(`Unknown beat id "${id}"`);
    return seg;
  }

  /** Progress value at fraction f (0..1) through segment `id`. */
  at(id, f = 0) {
    const seg = this.get(id);
    return seg.start + (seg.end - seg.start) * f;
  }

  /** Index of the segment containing progress p (cached walk; no allocation). */
  indexAt(p) {
    const segs = this.segments;
    let i = Math.min(this._last, segs.length - 1);
    while (i > 0 && p < segs[i].start) i--;
    while (i < segs.length - 1 && p >= segs[i].end) i++;
    this._last = i;
    return i;
  }

  /** Index of the chapter containing p (-1 during the intro). */
  chapterAt(p) {
    let idx = -1;
    for (let i = 0; i < this.chapters.length; i++) if (p >= this.chapters[i].start) idx = i;
    return idx;
  }
}

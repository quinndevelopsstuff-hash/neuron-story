/**
 * Converts STORY.md into src/data/story.json, the narration the site loads.
 *
 * STORY.md format:
 *   # Title
 *   ## Chapter N: Chapter title
 *   ### <beat id>
 *   [stage direction]
 *   narration paragraph(s)
 *
 * Runs automatically before `npm run dev` and `npm run build`.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const md = readFileSync(resolve(root, 'STORY.md'), 'utf8');
const outPath = resolve(root, 'src/data/story.json');

const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// Only *emphasis* is used in the narration.
const inline = (s) => escapeHtml(s).replace(/\*([^*]+)\*/g, '<em>$1</em>');

const story = { title: '', chapters: [] };
let chapter = null;
let beat = null;

const finishBeat = () => {
  if (!beat) return;
  beat.text = beat.text.trim();
  beat.html = inline(beat.text);
  beat.words = beat.text.split(/\s+/).filter(Boolean).length;
  if (!beat.text) throw new Error(`Beat ${beat.id} has no narration`);
  beat = null;
};

for (const raw of md.split('\n')) {
  const line = raw.trim();
  let m;
  if ((m = line.match(/^# (.+)/))) {
    story.title = m[1];
  } else if ((m = line.match(/^## Chapter (\d+): (.+)/))) {
    finishBeat();
    chapter = { number: Number(m[1]), title: m[2], beats: [] };
    story.chapters.push(chapter);
  } else if ((m = line.match(/^### (\S+)/))) {
    finishBeat();
    if (!chapter) throw new Error(`Beat ${m[1]} is outside a chapter`);
    beat = { id: m[1], stage: '', text: '' };
    chapter.beats.push(beat);
  } else if (beat && line.startsWith('[') && line.endsWith(']') && !beat.stage) {
    beat.stage = line.slice(1, -1);
  } else if (beat && line && line !== '---') {
    beat.text += (beat.text ? ' ' : '') + line;
  }
}
finishBeat();

const beats = story.chapters.flatMap((c) => c.beats);
const words = beats.reduce((n, b) => n + b.words, 0);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(story, null, 2) + '\n');
console.log(`story.json: ${story.chapters.length} chapters, ${beats.length} beats, ${words} words`);

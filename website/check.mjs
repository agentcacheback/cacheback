import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { gzipSync } from 'node:zlib';

const root = dirname(fileURLToPath(import.meta.url));
const dist = resolve(root, 'assets');
const project = resolve(root, '..');
const read = (name) => readFileSync(resolve(dist, name), 'utf8');
const html = readFileSync(resolve(project, 'index.html'), 'utf8');
const css = read('styles.css');
const js = read('app.js');
const data = JSON.parse(read('results.json'));
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, 'Duplicate HTML IDs');
assert.equal((html.match(/<h1\b/g) || []).length, 1, 'One page heading');
assert.match(html, /<html lang="en">/);
assert.match(html, /name="viewport"/);
assert.match(html, /<title>CacheBack \|/);
const sectionOrder = ['result-summary', 'abstract', 'results', 'demo', 'code', 'paper'];
sectionOrder.slice(1).forEach((id, i) => {
  assert.ok(html.indexOf(`id="${sectionOrder[i]}"`) < html.indexOf(`id="${id}"`), `Section order: ${id}`);
});
assert.equal((html.match(/<img src="website\/assets\/figures\//g) || []).length, 5);

for (const [, url] of html.matchAll(/(?:href|src|poster)="([^"]+)"/g)) {
  if (/^(https:|data:)/.test(url)) continue;
  if (url.startsWith('#')) {
    assert.ok(url === '#' || ids.includes(url.slice(1)), `Missing anchor: ${url}`);
  } else {
    const pathname = new URL(url, 'https://example.invalid/rclc/').pathname;
    assert.ok(pathname.startsWith('/rclc/'), `Link escapes the project path: ${url}`);
    const path = pathname.slice('/rclc/'.length);
    const asset = path.endsWith('/') ? `${path}index.html` : path;
    assert.ok(statSync(resolve(project, asset)).isFile(), `Missing local asset: ${url}`);
  }
}
for (const [, id] of js.matchAll(/byId\('([^']+)'\)/g)) {
  assert.ok(ids.includes(id), `Script refers to missing element: ${id}`);
}
assert.equal(data.models.gemma.ratio, 4, 'Use the paper selected point, not Gemma maximum accuracy');
assert.equal(data.models.ministral.ratio, 32);
assert.equal(data.models.nemotron.ratio, 8);
for (const model of Object.values(data.models)) {
  assert.ok(model.ratio >= 2 && Number.isInteger(model.ratio));
  for (const arm of [model.text, model.cache]) {
    assert.ok(arm.accuracy >= 0 && arm.accuracy <= 1);
    assert.ok(arm.seconds > 0 && Number.isFinite(arm.seconds));
  }
}
// The displayed bar lengths must encode the published data on shared zero-based scales.
const plottedWidths = [...html.matchAll(/<i style="width:([\d.]+)%"/g)].map(match => Number(match[1]));
const expectedWidths = Object.values(data.models).flatMap(model => [
  model.text.accuracy * 100, model.cache.accuracy * 100,
  model.text.seconds / 1600 * 100, model.cache.seconds / 1600 * 100,
]);
assert.equal(plottedWidths.length, 16);
plottedWidths.forEach((width, i) => assert.ok(Math.abs(width - expectedWidths[i]) < 0.0001));
const gains = [...html.matchAll(/class="accuracy-gain">\+([\d.]+)<abbr/g)].map(match => Number(match[1]));
const speedups = [...html.matchAll(/<strong class="speedup">([\d.]+)× faster<\/strong>/g)].map(match => Number(match[1]));
assert.deepEqual(gains, Object.values(data.models).map(m => Number(((m.cache.accuracy - m.text.accuracy) * 100).toFixed(1))));
assert.deepEqual(speedups, Object.values(data.models).map(m => Number((m.text.seconds / m.cache.seconds).toFixed(1))));
assert.doesNotMatch(html, /name="model"/);
assert.match(html, /This recording is an example, not a benchmark/);
assert.match(html, /93\.75% removed/);
assert.match(html, /loading="lazy" sandbox="allow-scripts allow-same-origin"/);
assert.match(css, /prefers-reduced-motion:reduce/);
assert.match(css, /:focus-visible/);
assert.match(html, /id="bibtex"/);
assert.match(html, /<iframe[^>]+src="demo\/index\.html\?v=fit-3"/);
assert.match(readFileSync(resolve(project, 'demo/index.html'), 'utf8'), /fetch\("trace-w4\.json"\)/);
const recording = JSON.parse(readFileSync(resolve(project, 'demo/trace-w4.json'), 'utf8')).cases[0];
const plain = text => text.replace(/<[^>]+>/g, '').replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
assert.ok(html.includes(`<h3 id="booking-title">${recording.question}</h3>`));
for (const method of ['rclc', 'text']) {
  const [answer, ...evidence] = recording[method].answer.text.split('\n\n');
  for (const [part, tag, expected] of [['answer', 'p', answer], ['evidence', 'div', evidence.join('\n\n')]]) {
    const displayed = html.match(new RegExp(`<${tag} id="demo-${method}-${part}">([\\s\\S]*?)</${tag}>`));
    assert.ok(displayed, `Missing recorded ${method} ${part}`);
    assert.equal(plain(displayed[1]), plain(expected.replace(/\*\*/g, '')), `Preserve the recorded ${method} ${part}`);
  }
  assert.ok(html.includes(`id="demo-${method}-time">${recording[method].seconds.toFixed(2)} s</span>`), `Measured ${method} completion time`);
}

const coding = JSON.parse(readFileSync(resolve(project, 'demo/coding/evidence.json'), 'utf8')).arms;
assert.ok(html.indexOf('id="coding-demo"') < html.indexOf('id="booking-demo"'), 'Coding video is the main demo');
const video = html.match(/<video\b[^>]*>[\s\S]*?<\/video>/)[0];
assert.match(video, /controls playsinline preload="none"/);
assert.doesNotMatch(video, /autoplay/);
assert.match(video, /src="demo\/coding\/rclc-cacheback-4k.mp4"/);
assert.match(video, /poster="demo\/coding\/poster.jpg"/);
assert.equal(readFileSync(resolve(project, 'demo/coding/rclc-cacheback-4k.mp4')).toString('ascii', 4, 8), 'ftyp', 'Real MP4, not a Git LFS pointer');
assert.ok(html.includes(`${(coding.text.seconds / coding.latent.seconds).toFixed(2)}× faster`));
assert.ok(html.includes(`CacheBack ${coding.latent.seconds.toFixed(2)} s; text ${coding.text.seconds.toFixed(2)} s`));
assert.equal(coding.latent.patch, coding.text.patch);
for (const arm of Object.values(coding)) {
  assert.equal(arm.new_tests_passed + arm.regression_tests_passed, 88);
  assert.equal(arm.failed_tests, 0);
}

assert.ok(gzipSync(js).length < 6000, 'Keep page JavaScript below 6 KB gzip');
assert.ok(!/[\u2013\u2014\u200b-\u200d\ufeff\u2060]/u.test(html), 'Banned punctuation or invisible character: index.html');
let bytes = Buffer.byteLength(html);
for (const name of readdirSync(dist, { recursive: true })) {
  if (!statSync(resolve(dist, name)).isFile()) continue;
  const content = readFileSync(resolve(dist, name));
  bytes += content.length;
  assert.ok(!/[\u2013\u2014\u200b-\u200d\ufeff\u2060]/u.test(content.toString()), `Banned punctuation or invisible character: ${name}`);
}
assert.ok(bytes < 100000, 'Keep first-party static assets below 100 KB');
console.log(`Website checks passed: anchors, assets, benchmark points, accessibility hooks; ${bytes} bytes, ${gzipSync(js).length} bytes gzipped JS.`);

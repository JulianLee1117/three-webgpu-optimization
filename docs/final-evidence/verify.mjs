import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url)));
for (const entry of manifest.reports) {
  const stored = readFileSync(new URL(entry.file, import.meta.url));
  assert.equal(digest(stored), entry.gzipSHA256);
  const raw = gunzipSync(stored);
  assert.equal(raw.length, entry.rawBytes);
  assert.equal(digest(raw), entry.rawSHA256);
  assert.equal(JSON.parse(raw).status, 'passed');
}
console.log(JSON.stringify({verified: manifest.reports.length, gpuRerun: false,
  scope: 'Archived report checksums and recorded status, not rerun inference.'}));

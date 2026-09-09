import { mkdir, readFile, writeFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidate = path.join(root, '.local-research/three-ort-gpu-buffer-current-dev-candidate');
const out = path.join(root, '.generated/ort-surface');
const base = 'c4ffe022f2a4f982b42b7da5af79a87066a138ae';
const git = (...args) => execFileSync('git', ['-c', 'core.autocrlf=false', '-C', candidate, ...args], { windowsHide: true, maxBuffer: 32e6 });
const hash = x => createHash('sha256').update(x).digest('hex');
await mkdir(out, { recursive: true });
// Include both tracked edits and the new external-attribute class and test.
let patch = git('diff', base, '--', 'src', 'test').toString();
for (const file of git('ls-files', '--others', '--exclude-standard', 'src', 'test').toString().trim().split('\n').filter(Boolean)) {
  const content = (await readFile(path.join(candidate, file), 'utf8')).replaceAll('\r\n', '\n');
  const lines = content.trimEnd().split('\n');
  patch += `diff --git a/${file} b/${file}\nnew file mode 100644\n--- /dev/null\n+++ b/${file}\n@@ -0,0 +1,${lines.length} @@\n${lines.map(l => '+' + l).join('\n')}\n`;
}
await writeFile(path.join(root, 'patches/three-external-storage-buffer.patch'), patch);
const { rollup } = await import(pathToFileURL(path.join(candidate, 'node_modules/rollup/dist/es/rollup.js')));
const configs = (await import(pathToFileURL(path.join(candidate, 'utils/build/rollup.config.js')))).default;
for (const config of configs) {
  const input = typeof config.input === 'string' ? path.resolve(candidate, config.input)
    : Object.fromEntries(Object.entries(config.input).map(([n,f]) => [n,path.resolve(candidate,f)]));
  const bundle = await rollup({ ...config, input });
  try { for (const output of [config.output].flat()) await bundle.write({ ...output, dir: out, file: undefined }); }
  finally { await bundle.close(); }
}
const files = {};
for (const file of (await readdir(out)).filter(f => f.endsWith('.js')).sort()) {
  const data = await readFile(path.join(out, file)); files[file] = { bytes: data.length, sha256: hash(data) };
}
const manifest = { baseCommit: base, candidateHead: git('rev-parse', 'HEAD').toString().trim(),
  includesUncommittedCandidateWork: true, patchSha256: hash(patch), files };
await writeFile(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await writeFile(path.join(root, 'patches/three-external-storage-buffer.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));

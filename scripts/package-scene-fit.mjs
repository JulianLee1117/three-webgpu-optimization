import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (ok, message) => { if (!ok) throw Error(message); };
const created = new Date().toISOString();
const output = path.join(root, 'deliverables', 'scene-fit-demo-' + created.replaceAll(':', '-'));
const sourceFiles = ['app.js','engine.js','gpu-equations.js','gpu-loss.js','index.html','least-squares.mjs','optimizers.mjs','PROTOCOL-V2.md','PROTOCOL.md','README.md','scenes.js'];
const reports = [
  '2026-09-09T17-49-15.542Z-robot-11', '2026-09-09T17-49-47.984Z-rover-11',
  '2026-09-09T17-49-55.631Z-pavilion-11', '2026-09-09T17-50-03.232Z-rover-29',
  '2026-09-09T17-50-10.852Z-robot-29', '2026-09-09T17-50-18.415Z-pavilion-29',
  '2026-09-09T17-50-26.015Z-rover-47', '2026-09-09T17-50-33.634Z-robot-47',
  '2026-09-09T17-50-40.432Z-pavilion-47'
];
const sourceCommit = execFileSync('git', ['rev-parse', '494f052^{commit}'], { cwd: root, windowsHide: true, encoding: 'utf8' }).trim();
const payload = new Map();
const add = (name, bytes) => { assert(!payload.has(name), 'Duplicate package file: ' + name); payload.set(name, Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes)); };
async function copy(source, destination = source) { add(destination, await readFile(path.join(root, source))); }
async function verifiedReport(directory, expectedHash) {
  const bytes = await readFile(path.join(root, directory, 'report.json'));
  const sidecar = await readFile(path.join(root, directory, 'report.json.sha256'));
  assert(hash(bytes) === sidecar.toString().trim(), 'Report checksum mismatch: ' + directory);
  if (expectedHash) assert(hash(bytes) === expectedHash, 'Unexpected report identity.');
  assert(!/"[A-Za-z]:[\\/]|\/Users\/|\/home\//.test(bytes.toString()), 'Report contains an absolute user path: ' + directory);
  return { bytes, sidecar, value: JSON.parse(bytes) };
}

for (const file of sourceFiles) await copy('experiments/scene-fit/' + file);
for (const file of ['three.webgpu.js', 'three.core.js']) await copy('node_modules/three/build/' + file);
await copy('node_modules/three/LICENSE');
const threeVersion = JSON.parse(await readFile(path.join(root, 'node_modules/three/package.json'))).version;
assert(threeVersion === '0.185.1', 'The portable package requires stock three@0.185.1.');

let recordedSources, sourceIdentitySha256;
const cells = new Set(), reportRecords = [];
let trainingWins = 0, heldoutWins = 0;
for (const name of reports) {
  const { bytes, sidecar, value } = await verifiedReport('results/development/scene-fit/' + name);
  assert(value.kind === 'editable-scene-fit-v2' && value.status === 'passed', 'Expected passed V2 report: ' + name);
  assert(!value.errors.length && value.finalState.disposed && !value.finalState.errors.length && value.fixture.equationParity.passed, 'Failed correctness/cleanup gate: ' + name);
  const canonical = JSON.stringify(Object.fromEntries(Object.entries(value.sources).sort(([a],[b]) => a.localeCompare(b))));
  const identity = hash(value.kind + '\n' + canonical);
  if (!recordedSources) { recordedSources = value.sources; sourceIdentitySha256 = identity; }
  assert(identity === sourceIdentitySha256, 'Mixed V2 source cohorts.');
  const cell = value.scene + ':' + value.seed;
  assert(!cells.has(cell), 'Duplicate V2 scene/seed.'); cells.add(cell);
  const candidate = value.results.find(item => item.method === 'least-squares');
  const others = value.results.filter(item => ['random','coordinate','candidate'].includes(item.method));
  assert(candidate && others.length === 3 && value.budget === 96, 'Incomplete V2 methods/budget.');
  if (candidate.bestLoss < Math.min(...others.map(item => item.bestLoss))) trainingWins++;
  if (candidate.finalHeldout < Math.min(...others.map(item => item.finalHeldout))) heldoutWins++;
  add('evidence/v2/' + name + '/report.json', bytes);
  add('evidence/v2/' + name + '/report.json.sha256', sidecar);
  reportRecords.push({ scene: value.scene, seed: value.seed, path: 'evidence/v2/' + name + '/report.json', sha256: hash(bytes) });
}
for (const scene of ['rover','robot','pavilion']) for (const seed of [11,29,47]) assert(cells.has(scene + ':' + seed), 'Missing V2 scene/seed.');
assert(trainingWins === 0 && heldoutWins === 0, 'The package result statement no longer matches its evidence.');

const currentSourceDifferences = [];
for (const [filename, expected] of Object.entries(recordedSources)) {
  if (filename.startsWith('node_modules/')) {
    assert(payload.has(filename) && hash(payload.get(filename)) === expected, 'Bundled Three differs from V2 evidence: ' + filename);
  } else {
    assert(filename === 'scripts/run-scene-fit.mjs' || filename.startsWith('experiments/scene-fit/'), 'Unexpected historical source path.');
    const bytes = execFileSync('git', ['show', sourceCommit + ':' + filename], { cwd: root, windowsHide: true, maxBuffer: 16e6 });
    assert(hash(bytes) === expected, 'Commit does not reproduce recorded source: ' + filename);
    add('evidence/source-v2/' + filename, bytes);
    if (payload.has(filename) && hash(payload.get(filename)) !== expected) currentSourceDifferences.push(filename);
  }
}
const analysisDirectory = 'results/development/scene-fit-analysis/2026-09-09T17-57-35.608Z';
const analysis = await readFile(path.join(root, analysisDirectory, 'summary.json'));
const analysisSidecar = await readFile(path.join(root, analysisDirectory, 'summary.json.sha256'));
assert(hash(analysis) === analysisSidecar.toString().trim() && hash(analysis) === '9cfc5a65b0ab34d0a31794c338deac05f46f5fd3a2f027b8b17e0b02c9ca5aee', 'Final analysis checksum mismatch.');
assert(!/"[A-Za-z]:[\\/]|\/Users\/|\/home\//.test(analysis.toString()), 'Analysis contains an absolute user path.');
add('evidence/analysis/summary.json', analysis);
add('evidence/analysis/summary.json.sha256', analysisSidecar);
for (const name of ['scene-fit.mp4','fitted.png','heldout.png']) await copy('deliverables/scene-fit/2026-09-09T17-57-07.628Z/' + name, 'media/' + name);

add('serve.mjs', String.raw`import { createServer } from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const prefixes = ['/experiments/scene-fit/', '/node_modules/three/build/', '/evidence/', '/media/'];
const types = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.md':'text/plain','.sha256':'text/plain','.png':'image/png','.mp4':'video/mp4'};
const inside = (base,file) => { const relative=path.relative(base,file); return relative!=='' && relative!=='..' && !relative.startsWith('..'+path.sep) && !path.isAbsolute(relative); };
export async function startServer(port=5188) {
  const packageRoot = await realpath(root);
  const server = createServer(async (request,response) => {
    try {
      if (!['GET','HEAD'].includes(request.method)) throw Error('Unsupported method');
      let name=decodeURIComponent(new URL(request.url,'http://localhost').pathname);
      if (name==='/') name='/experiments/scene-fit/index.html';
      if (name.includes(String.fromCharCode(92)) || name.includes(String.fromCharCode(0)) || name.split('/').some(part=>part==='..'||part==='.')) throw Error('Invalid path');
      const prefix=prefixes.find(value=>name.startsWith(value));
      if (!prefix) throw Error('Unknown route');
      const [directory,file]=await Promise.all([realpath(path.resolve(root,'.'+prefix)),realpath(path.resolve(root,'.'+name))]);
      if (!inside(packageRoot,directory) || !inside(directory,file) || !(await stat(file)).isFile()) throw Error('Directory escape');
      const bytes=await readFile(file);
      response.writeHead(200,{'Content-Type':types[path.extname(file)]??'application/octet-stream','Content-Length':bytes.length,'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'});
      response.end(request.method==='HEAD'?undefined:bytes);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
  return server;
}
if (process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  startServer().then(()=>console.log('Open http://127.0.0.1:5188/ in Chrome. Ctrl+C stops this local server.')).catch(error=>{console.error(error.message);process.exitCode=1;});
}
`);

add('README_RUN.md', `# Portable editable scene fitting demo

Open a terminal in this extracted folder and run:

\`\`\`sh
node serve.mjs
\`\`\`

Use Node 22.12 or later. Open http://127.0.0.1:5188/ in Google Chrome with WebGPU
support. No npm installation, model download, Python, Playwright or network service
is required. The server uses Node builtins and listens only on 127.0.0.1. Stop it
with Ctrl+C and close the demo tab when finished. If port 5188 is occupied, stop
the other local demo server first.

Select rover, robot or pavilion, choose a solver and click Fit once. The page starts
idle and uses 96 fitting renders. Inspect the unseen camera, adjust the resulting
editable dimensions, or export their JSON. Reload for another bounded fit.
These are three known scene families with fixed colors, lighting and cameras;
this does not reconstruct arbitrary scenes or photographs.

## Result and evidence

The GPU least-squares method won 0/9 training comparisons and 0/9 held-out
comparisons against the lowest-error alternative among random, coordinate and
Powell-style search in the included V2 cohort. Its recorded GPU correctness checks
passed. This is a working prototype and negative solver result, not a new algorithm,
a demonstrated speedup or solver advantage.

The nine original reports and SHA256 sidecars are in evidence/v2/. Their exact
recorded repository source is in evidence/source-v2/, recovered from commit
${sourceCommit}. The unchanged Three builds are shared with the running demo.
The final analysis is in evidence/analysis/. It also describes historical V1
cohorts whose reports are not included here; the portable evidence set contains
only the nine V2 reports. The MP4 and two screenshots are in media/.

The running demo includes later interface/editability changes. PACKAGE.json lists
runtime files that differ from the archived V2 source; the original nine-cell
comparison is not presented as a rerun of these later bytes. The archived runner
is provenance only and requires the original repository's automation dependencies.
Use serve.mjs for this standalone demo. experiments/scene-fit/README.md retains
the full research notes; its repository runner/analyzer commands and ignored local
result paths refer to the original workspace, not extra portable dependencies.

SHA256.json covers every payload file, including reports, source snapshots and
media. SHA256.json.sha256 authenticates the manifest's bytes for integrity checking;
hashes do not independently establish experimental authenticity.

## License status

The bundled Three.js ${threeVersion} files retain their MIT notices and full license
at node_modules/three/LICENSE. The original experiment source currently has no
separate license grant. This package preserves that status and does not assign
an MIT or other license to that original source. No models, private project files,
ONNX Runtime, Three Blocks or other experiments are bundled.
`);
add('PACKAGE.json', JSON.stringify({ kind: 'portable-editable-scene-fit-v1', created,
  runtimeThreeVersion: threeVersion, evidenceSourceCommit: sourceCommit,
  evidenceSourceIdentitySha256: sourceIdentitySha256, currentSourceDifferences,
  result: { v2Cells: 9, leastSquaresTrainingWins: trainingWins, leastSquaresHeldoutWins: heldoutWins },
  reports: reportRecords, packageScriptSha256: hash(await readFile(fileURLToPath(import.meta.url))),
  licenseStatus: 'Three.js MIT license retained; no new license grant for original experiment source.'
}, null, 2) + '\n');

// All input and cohort checks finish before creating the unique output directory.
await mkdir(path.dirname(output), { recursive: true });
await mkdir(output);
for (const [filename,bytes] of payload) {
  const destination = path.join(output, filename);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, bytes, { flag: 'wx' });
}
const files = Object.fromEntries([...payload].sort(([a],[b])=>a.localeCompare(b)).map(([name,bytes])=>[name,{bytes:bytes.length,sha256:hash(bytes)}]));
const manifest = JSON.stringify({ kind: 'scene-fit-package-sha256-v1', created, files,
  exclusions: ['SHA256.json','SHA256.json.sha256'] }, null, 2) + '\n';
await writeFile(path.join(output,'SHA256.json'), manifest, { flag: 'wx' });
await writeFile(path.join(output,'SHA256.json.sha256'), hash(manifest) + '\n', { flag: 'wx' });
console.log(JSON.stringify({output,files:payload.size+2,bytes:[...payload.values()].reduce((sum,bytes)=>sum+bytes.length,0),v2Cells:reports.length,sourceCommit,currentSourceDifferences},null,2));

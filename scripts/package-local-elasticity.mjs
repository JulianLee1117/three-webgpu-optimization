// Portable static build only. Never starts a browser or GPU.
import { build } from 'vite';
import { readFile, writeFile, mkdir, readdir, copyFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const experiment = 'experiments/local-elasticity';
const created = new Date().toISOString();
const output = path.join(root, 'deliverables', `local-elasticity-${created.replaceAll(':', '-')}`);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (ok, message) => { if (!ok) throw Error(message); };
const relative = filename => path.relative(root, filename).replaceAll('\\', '/');
const inputHashes = {}, captured = new Map();
async function capture(filename) {
  const absolute = path.resolve(root, filename), bytes = await readFile(absolute);
  if (captured.has(absolute)) assert(captured.get(absolute) === sha(bytes), 'Input changed during build: ' + relative(absolute));
  captured.set(absolute, sha(bytes)); inputHashes[relative(absolute)] = sha(bytes);
  return bytes;
}
async function put(name, bytes) {
  const filename = path.join(output, name); await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, bytes, { flag: 'wx' });
}
async function walk(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const name = prefix + entry.name;
    if (entry.isDirectory()) result.push(...await walk(path.join(directory, entry.name), name + '/'));
    else { assert(entry.isFile(), `Nonregular package file: ${name}`); result.push(name); }
  }
  return result.sort();
}

const threeRoot = path.join(root, 'node_modules/three-r186');
const three = JSON.parse(await capture('node_modules/three-r186/package.json'));
assert(three.version === '0.186.0', 'Requires the exact three-r186 alias at0.186.0');
const aliases = [
  { find: /^three\/webgpu$/, replacement: path.join(threeRoot, 'src/Three.WebGPU.js') },
  { find: /^three\/tsl$/, replacement: path.join(threeRoot, 'src/Three.TSL.js') },
  { find: /^three\/addons\//, replacement: path.join(threeRoot, 'examples/jsm/') },
  { find: /^three\/src\//, replacement: path.join(threeRoot, 'src/') },
  { find: /^three$/, replacement: path.join(threeRoot, 'src/Three.js') },
];
function sourceAuditPlugin() {
  return { name: 'local-elasticity-runtime-boundary',
    enforce: 'pre',
    async transform(code, id) {
      const filename = id.split('?')[0].replaceAll('\\', '/');
      if (filename.includes('/node_modules/three/')) throw Error('Mixed stock185 runtime import: ' + filename);
      if (filename.startsWith(root.replaceAll('\\', '/')) && !id.includes('?')) {
        try { if ((await stat(filename)).isFile()) await capture(filename); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
      if (filename.endsWith('/experiments/local-elasticity/app.js')) {
        // Vite requires an explicitly relative template glob. Both resolve to
        // the same source directory in the unbundled browser application.
        return code.replace('new URL(`assets/${asset}.splat`, import.meta.url)',
          'new URL(`./assets/${asset}.splat`, import.meta.url)');
      }
      return null;
    },
    transformIndexHtml: { order: 'pre', handler(html) {
      return html.replace(/<script type="importmap">[\s\S]*?<\/script>/g, '');
    } },
  };
}

// Preserve exact complete evidence, including the interaction gate failure.
const evidenceDirectory = path.join(root, experiment, 'evidence');
await mkdir(evidenceDirectory, { recursive: true });
let previousEvidence;
try {
  const bytes = await readFile(path.join(evidenceDirectory, 'manifest.json'));
  const expected = (await readFile(path.join(evidenceDirectory, 'manifest.sha256'), 'utf8')).trim().split(' ')[0];
  assert(sha(bytes) === expected, 'Existing evidence manifest checksum mismatch');
  previousEvidence = JSON.parse(bytes);
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const records = [
  ['cpu-report', 'results/development/local-elasticity-cpu/2026-09-10T00-10-38.217Z/report.json', 'passed-independent-audit'],
  ['cpu-reference-audit', 'results/development/local-elasticity-cpu/2026-09-10T00-10-38.217Z/reference-audit.json', 'passed'],
  ['interaction-report', 'results/development/local-elasticity-interaction/2026-09-10T00-13-41.554Z/report.json', 'retain-exact-status'],
  ['cpu-fixed-report', 'results/development/local-elasticity-cpu/2026-09-10T00-22-30.451Z/report.json', 'passed-independent-audit-after-roundoff-fix'],
  ['cpu-fixed-reference-audit', 'results/development/local-elasticity-cpu/2026-09-10T00-22-30.451Z/reference-audit.json', 'passed'],
  ['interaction-fixed-report', 'results/development/local-elasticity-interaction/2026-09-10T00-23-26.774Z/report.json', 'passed-124-of-124-after-roundoff-fix'],
  ['gpu-covariance-report', 'results/development/local-elasticity-gpu/2026-09-10T00-15-03.624Z-covariance/report.json', 'passed-eight-retained-gpu-cases'],
  ['gpu-covariance-reference-audit', 'results/development/local-elasticity-gpu/2026-09-10T00-15-03.624Z-covariance/reference-audit.json', 'passed-independent-numpy-audit-no-new-gpu'],
  ['ui-before-roundoff-fix', 'results/development/local-elasticity/2026-09-10T00-17-34.677Z-plant-ui/report.json', 'functional-ui-pass-with-nonconverged-solves-retained'],
];
const evidenceManifest = { kind: 'local-elasticity-complete-evidence-v1',
  note: 'Complete original JSON bytes. Earlier interaction/UI nonconvergence remains retained; fixed-version reruns are separate. No redactions.', reports: [] };
async function immutableEvidence(name, bytes) {
  const target = path.join(evidenceDirectory, name);
  try { assert(sha(await readFile(target)) === sha(bytes), 'Existing evidence differs: ' + name); }
  catch (error) { if (error.code !== 'ENOENT') throw error; await writeFile(target, bytes, { flag: 'wx' }); }
}
for (const [name, source, interpretation] of records) {
  let bytes, retained = process.argv.includes('--retained-evidence');
  if (!retained) {
    try { bytes = await readFile(path.join(root, source)); }
    catch (error) { if (error.code !== 'ENOENT') throw error; retained = true; }
  }
  if (retained) {
    const compressed = await readFile(path.join(evidenceDirectory, name + '.json.gz'));
    const entry = previousEvidence?.reports.find(item => item.file === name + '.json.gz');
    assert(entry && sha(compressed) === entry.gzipSHA256, 'Retained evidence gzip checksum mismatch: ' + name);
    bytes = gunzipSync(compressed);
    assert(sha(bytes) === entry.originalRawSHA256 && sha(bytes) === entry.publicJSONSHA256,
      'Retained evidence raw checksum mismatch: ' + name);
  }
  const digest = sha(bytes), json = JSON.parse(bytes);
  let sidecar;
  const sidecars = retained ? [`${experiment}/evidence/${name}.sha256`] : [source.replace(/\.json$/, '.sha256'), source + '.sha256'];
  for (const filename of sidecars) {
    try { sidecar = (await readFile(path.join(root, filename), 'utf8')).trim().split(/\s+/)[0]; break; }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  assert(sidecar === digest, 'Original evidence checksum mismatch: ' + source);
  assert(!/C:\\\\Users\\\\|file:\/\/\/C:\/Users\//i.test(bytes.toString()), 'Evidence contains private workspace path');
  const compressed = gzipSync(bytes, { level: 9 });
  assert(sha(gunzipSync(compressed)) === digest, 'Compression changed original JSON bytes');
  await immutableEvidence(name + '.json.gz', compressed);
  await immutableEvidence(name + '.sha256', digest + `  ${name}.json\n`);
  evidenceManifest.reports.push({ file: name + '.json.gz', originalSource: source,
    originalRawSHA256: digest, publicJSONSHA256: digest, gzipSHA256: sha(compressed),
    bytes: bytes.length, gzipBytes: compressed.length, status: json.status ?? json.passed ?? null,
    interpretation });
}
// Later package/lifecycle QA is repository evidence, appended after ZIP sealing.
// Keep those complete records when rebuilding from a fresh clone's gzip files.
for (const entry of previousEvidence?.reports ?? []) {
  if (evidenceManifest.reports.some(item => item.file === entry.file)) continue;
  assert(/^[a-zA-Z0-9._-]+\.json\.gz$/.test(entry.file), 'Invalid extra evidence filename');
  const compressed = await readFile(path.join(evidenceDirectory, entry.file));
  const bytes = gunzipSync(compressed);
  assert(sha(compressed) === entry.gzipSHA256 && sha(bytes) === entry.publicJSONSHA256 &&
    sha(bytes) === entry.originalRawSHA256, 'Additional exact evidence checksum mismatch: ' + entry.file);
  evidenceManifest.reports.push(entry);
}
const evidenceReadme = `# Complete retained records\n\nThe initial numerical canary passed. The initial interaction report retains its failed all-solves-converged gate (one plant solve reached its time budget), and the source UI record retains three near-converged line-search failures. These records were not replaced.\n\nA documented energy-roundoff fix preserves the convergence threshold and requires real projected-gradient reduction. The separate cpu-fixed-report and its independent audit passed; interaction-fixed-report passed124/124 solves under the same inputs and settings. Eight GPU covariance cases and a separate NumPy audit are included. These are bounded numerical checks, not proof of calibrated physical fidelity.\n\nEach .json.gz expands to the exact original JSON bytes; manifest.json records raw and compressed SHA256 hashes. No report fields were removed or redacted. Embedded source snapshots preserve the implementation evaluated at each stage.\n\nFrom the repository root, with Python and NumPy available:\n\n\`\`\`sh\npython experiments/local-elasticity/reference.py --report experiments/local-elasticity/evidence/cpu-fixed-report.json.gz\npython experiments/local-elasticity/reference.py --report experiments/local-elasticity/evidence/gpu-covariance-report.json.gz\n\`\`\`\n\nThe first command independently rebuilds RKPM matrices/eigenspace, energy/forces, Hessians and four steps per asset. It verifies the raw checksum and embedded source hashes. The second recomputes the full deformation Jacobian and covariance transformation from retained Float32 GPU inputs, without using a GPU. CPU PyTorch is only needed for the optional reference.py --self-test. The interaction report retains its own complete trajectory and held-out data; the four-step verifier does not certify that separate trajectory gate.\n`;
const supplementary = evidenceManifest.reports.filter(item => /bundle|lifecycle/.test(item.file));
const supplementaryNote = supplementary.length ? '\n## Packaged UI and lifecycle records\n\n' +
  supplementary.map(item => `- ${item.file}: ${item.interpretation}.`).join('\n') +
  '\n\nPackaged UI reports name the exact package manifest they tested. The sealed ZIP is not modified when later QA records are added here. Earlier passes cover their recorded cases and source version; they do not retroactively cover later boundary tests.\n' : '';
await writeFile(path.join(evidenceDirectory, 'README.md'), evidenceReadme + supplementaryNote);
const evidenceBytes = Buffer.from(JSON.stringify(evidenceManifest, null, 2) + '\n');
await writeFile(path.join(evidenceDirectory, 'manifest.json'), evidenceBytes);
await writeFile(path.join(evidenceDirectory, 'manifest.sha256'), sha(evidenceBytes) + '  manifest.json\n');
if (process.argv.includes('--evidence-only')) {
  console.log(JSON.stringify({ passed: true, retainedEvidence: process.argv.includes('--retained-evidence'),
    reports: evidenceManifest.reports.length, manifestSHA256: sha(evidenceBytes), buildOrGPU: false }));
  process.exit(0);
}

await mkdir(path.dirname(output), { recursive: true });
await mkdir(output); // New output only; Vite is forbidden from cleaning directories.
await capture(path.join(experiment, 'app.html'));
await build({ configFile: false, envFile: false, root, publicDir: false, base: './',
  resolve: { alias: aliases }, assetsInclude: ['**/*.splat'],
  plugins: [sourceAuditPlugin()], worker: { format: 'es', plugins: () => [sourceAuditPlugin()] },
  build: { outDir: output, emptyOutDir: false, target: 'es2022', sourcemap: false,
    assetsInlineLimit: 0, minify: true, chunkSizeWarningLimit: 2500,
    rollupOptions: { input: path.join(root, experiment, 'app.html') } },
});

const sourceFiles = ['app.html', 'app.js', 'assets.mjs', 'basis.mjs', 'basis.test.mjs',
  'mechanics.mjs', 'mechanics.test.mjs', 'deform.js', 'worker.mjs', 'reference.py', 'README.md'];
for (const optional of ['RESULTS.md', 'demo.png']) {
  try { await stat(path.join(root, experiment, optional)); sourceFiles.push(optional); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
for (const name of sourceFiles) {
  const bytes = await capture(`${experiment}/${name}`);
  await put(`source/${experiment}/${name}`, bytes);
  if (name.endsWith('.md') || name === 'reference.py' || name === 'demo.png') await put(`${experiment}/${name}`, bytes);
}
for (const name of await walk(evidenceDirectory)) await put(`${experiment}/evidence/${name}`, await readFile(path.join(evidenceDirectory, name)));
await put(`${experiment}/assets/manifest.json`, await capture(`${experiment}/assets/manifest.json`));
await put('scripts/check-local-elasticity-cpu.mjs', await capture('scripts/check-local-elasticity-cpu.mjs'));
await put('scripts/package-local-elasticity.mjs', await capture('scripts/package-local-elasticity.mjs'));
await put('scripts/check-local-elasticity-bundle.mjs', await capture('scripts/check-local-elasticity-bundle.mjs'));
await put('THREE-LICENSE.txt', await capture('node_modules/three-r186/LICENSE'));
const assetManifest = JSON.parse(await readFile(path.join(root, experiment, 'assets/manifest.json')));
const assetNotes = assetManifest.assets.map(item => `${item.id}: ${item.attribution}\nLicense: ${item.license}\nOriginal: ${assetManifest.source}/blob/${assetManifest.pin}/data/${item.file}\nSHA256: ${item.sha256}`).join('\n\n');
await put('ASSET-LICENSES.txt', `${assetNotes}\n\nCC BY4.0 terms: https://creativecommons.org/licenses/by/4.0/legalcode\nCC0 terms: https://creativecommons.org/publicdomain/zero/1.0/legalcode\nOriginal files are unmodified. The app transforms coordinates and covariance at runtime.\n`);
await put('THIRD-PARTY-NOTICES.txt', `Three.js0.186.0 is included under MIT; see THREE-LICENSE.txt. The app uses FreeForm's published reduced-elasticity construction, implemented locally. No Kaolin code/runtime or model weights are bundled. Asset terms are separate; see ASSET-LICENSES.txt. This package does not invent a license grant for repository-authored source: no top-level source LICENSE was present when packaged.\n`);
await put('index.html', '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=./experiments/local-elasticity/app.html"><a href="./experiments/local-elasticity/app.html">Open local elasticity</a>\n');
await put('serve.mjs', String.raw`import {createServer} from 'node:http';
import {readFile,realpath} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const base=await realpath(fileURLToPath(new URL('.',import.meta.url)));
const manifest=JSON.parse(await readFile(path.join(base,'manifest.json'),'utf8'));
const allowed=new Set([...Object.keys(manifest.files),'manifest.json','manifest.sha256']);
const argument=process.argv.find(x=>x.startsWith('--port='));
const port=Number(argument?.slice(7)??process.env.PORT??5186);
if(!Number.isInteger(port)||port<1024||port>65535)throw Error('Invalid local port');
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.json':'application/json','.gz':'application/gzip','.md':'text/plain; charset=utf-8','.txt':'text/plain; charset=utf-8','.py':'text/plain; charset=utf-8'};
const server=createServer(async(req,res)=>{
 try{
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405).end();return;}
  const raw=req.url.split('?')[0];let decoded;try{decoded=decodeURIComponent(raw);}catch{res.writeHead(400).end();return;}
  if(!decoded.startsWith('/')||decoded.includes(String.fromCharCode(92))||decoded.includes(String.fromCharCode(0))||decoded.includes('%')||decoded.split('/').some(x=>x==='.'||x==='..')){res.writeHead(400).end();return;}
  if(decoded==='/favicon.ico'){res.writeHead(204).end();return;}
  const name=decoded==='/'?'index.html':decoded.slice(1);
  if(!allowed.has(name)){res.writeHead(404).end();return;}
  const target=await realpath(path.resolve(base,name));
  if(!target.startsWith(base+path.sep)){res.writeHead(403).end();return;}
  const bytes=await readFile(target);
  res.writeHead(200,{'Content-Type':mime[path.extname(name)]??'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
  res.end(req.method==='HEAD'?undefined:bytes);
 }catch{res.writeHead(404).end();}
});
server.listen(port,'127.0.0.1',()=>console.log('Open http://127.0.0.1:'+port+'/ (idle until Prepare locally).'));
`);
await put('verify.mjs', String.raw`import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const hash=x=>createHash('sha256').update(x).digest('hex');
const raw=await readFile(new URL('manifest.json',import.meta.url));
const sidecar=(await readFile(new URL('manifest.sha256',import.meta.url),'utf8')).trim().split(' ')[0];
if(hash(raw)!==sidecar)throw Error('Manifest hash mismatch');
const manifest=JSON.parse(raw);
for(const [name,entry]of Object.entries(manifest.files)){
 if(name.startsWith('/')||name.includes('..')||name.includes(String.fromCharCode(92)))throw Error('Invalid manifest path');
 const bytes=await readFile(new URL(name,import.meta.url));
 if(bytes.length!==entry.bytes||hash(bytes)!==entry.sha256)throw Error('File mismatch: '+name);
}
console.log(JSON.stringify({passed:true,files:Object.keys(manifest.files).length,three:manifest.three}));
`);
await put('README_RUN.md', `# Local elasticity demo\n\nWith Node.js22.12 or newer, run \`node serve.mjs\` in this extracted folder and open http://127.0.0.1:5186/. Use a WebGPU-capable Chrome browser. No npm install, CDN, model download, API key or Python is needed for the demo. Choose an object and click **Prepare locally**. Drag it, or run the bounded12-second simulation. Hiding the page releases the GPU. Use \`--port=5187\` for another port.\n\nRun \`node verify.mjs\` to verify every packaged file using Node built-ins. The module worker and exact Three0.186.0 source runtime are bundled. Both original splat assets are local. Links to research and license websites are optional documentation links; the runtime makes no external requests.\n\nMethods: [README](${experiment}/README.md). CPU evidence: [complete records](${experiment}/evidence/README.md). A Python+NumPy user can run \`python ${experiment}/reference.py --report ${experiment}/evidence/cpu-fixed-report.json.gz\`. Python is an optional verifier only.\n\nThe fixed numerical canary and independent audit passed, and the unchanged interaction protocol passed124/124 solves after the documented roundoff fix. Earlier interaction/UI failures are retained separately; these bounded results do not establish unconditional convergence. Equal point weights, volume and material values are assumptions. No real material recovery, self-collision or general collision is established. This is an implementation of published FreeForm mechanics, not a new physics algorithm.\n\nThis build has static file/hash checks. A separate browser test is required before claiming this exact ZIP passed UI/GPU validation. Licenses and attribution are in THREE-LICENSE.txt, ASSET-LICENSES.txt and THIRD-PARTY-NOTICES.txt. Unbundled authored sources are under source/ for inspection; regenerating them requires the repository and its pinned dependencies.\n`);

const files = await walk(output);
const emitted = {};
for (const name of files) {
  const bytes = await readFile(path.join(output, name)); emitted[name] = { bytes: bytes.length, sha256: sha(bytes) };
  if (name.endsWith('.js') && !name.startsWith('source/') && !name.startsWith('scripts/')) {
    const text = bytes.toString();
    assert(!text.includes('/node_modules/three/'), 'Emitted bundle references stock185');
    assert(!/(?:from\s*|import\s*\()\s*["']https?:\/\//.test(text), 'External runtime module import');
  }
}
const splatFiles = files.filter(name => name.endsWith('.splat'));
assert(splatFiles.length === 2, 'Expected exactly two bundled original splat assets');
for (const asset of assetManifest.assets) assert(splatFiles.some(name => emitted[name].sha256 === asset.sha256), 'Bundled asset is not original: ' + asset.id);
assert(files.some(name => /worker-[^/]+\.js$/.test(name)), 'Bundled worker is missing');
assert(!Object.keys(inputHashes).some(name => name.startsWith('node_modules/three/')), 'Mixed runtime graph');
for (const [filename, digest] of captured) assert(sha(await readFile(filename)) === digest, 'Input changed during packaging: ' + relative(filename));
const manifest = { kind: 'local-elasticity-static-package-v1', created, three: three.version,
  buildTransforms: ['Remove development import map; normalize assets/ template URL to ./assets/ for Vite glob expansion'],
  browserQA: 'not-run-by-packager', runtime: 'Vite bundle; source-module aliases isolate Three0.186.0; worker bundled; local assets',
  inputs: inputHashes, files: emitted };
const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + '\n');
await put('manifest.json', manifestBytes); await put('manifest.sha256', sha(manifestBytes) + '  manifest.json\n');
execFileSync(process.execPath, ['--check', path.join(output, 'serve.mjs')], { windowsHide: true });
const verification = JSON.parse(execFileSync(process.execPath, [path.join(output, 'verify.mjs')], { windowsHide: true, encoding: 'utf8' }));
assert(process.platform === 'win32', 'ZIP creation currently uses Windows built-in .NET compression');
const quote = x => "'" + x.replaceAll("'", "''") + "'";
const ps = `$ErrorActionPreference='Stop'\nAdd-Type -AssemblyName System.IO.Compression\nAdd-Type -AssemblyName System.IO.Compression.FileSystem\n$base=(Resolve-Path -LiteralPath ${quote(path.join(root, 'deliverables'))}).Path\n$target=(Resolve-Path -LiteralPath ${quote(output)}).Path\nif(-not $target.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Package escaped deliverables'}\n$zip=[IO.Path]::GetFullPath($target+'.zip')\nif([IO.Path]::GetDirectoryName($zip) -ne $base){throw 'Archive escaped deliverables'}\n$stream=[IO.File]::Open($zip,[IO.FileMode]::CreateNew)
try {
 $archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
 try {
  foreach($file in (Get-ChildItem -LiteralPath $target -Recurse -File | Sort-Object FullName)) {
   if(-not $file.FullName.StartsWith($target+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive input escaped package'}
   $name=$file.FullName.Substring($target.Length+1).Replace([IO.Path]::DirectorySeparatorChar,[char]47)
   $entry=$archive.CreateEntry($name,[IO.Compression.CompressionLevel]::Optimal)
   $entry.LastWriteTime=[DateTimeOffset]::new(2000,1,1,0,0,0,[TimeSpan]::Zero)
   $input=[IO.File]::OpenRead($file.FullName)
   try { $sink=$entry.Open(); try { $input.CopyTo($sink) } finally { $sink.Dispose() } } finally { $input.Dispose() }
  }
 } finally { $archive.Dispose() }
} finally { $stream.Dispose() }\n`;
execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(ps, 'utf16le').toString('base64')], { windowsHide: true, timeout: 60000 });
const archive = await readFile(output + '.zip');
await writeFile(output + '.zip.sha256', sha(archive) + '  ' + path.basename(output) + '.zip\n', { flag: 'wx' });
console.log(JSON.stringify({ output: relative(output), zip: relative(output + '.zip'), zipBytes: archive.length,
  zipSHA256: sha(archive), manifestSHA256: sha(manifestBytes), verification,
  evidenceManifestSHA256: sha(evidenceBytes), inputModules: Object.keys(inputHashes).length,
  staticChecks: { originalAssets: 2, bundledWorker: true, mixed185Runtime: false }, gpuRun: false }, null, 2));

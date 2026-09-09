import { cp, mkdir, readFile, writeFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'deliverables/ort-surface-demo');
await mkdir(path.join(out, 'vendor/three'), { recursive: true });
await mkdir(path.join(out, 'vendor/ort'), { recursive: true });
await mkdir(path.join(out, 'licenses'), { recursive: true });
for (const file of ['three.webgpu.js', 'three.core.js', 'three.tsl.js']) await cp(path.join(root, '.generated/ort-surface', file), path.join(out, 'vendor/three', file));
for (const file of ['ort.webgpu.bundle.min.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) await cp(path.join(root, 'node_modules/onnxruntime-web-dev/dist', file), path.join(out, 'vendor/ort', file));
for (const file of ['index.html', 'surface.js']) {
  const source = await readFile(path.join(root, 'experiments/ort-surface', file), 'utf8');
  await writeFile(path.join(out, file), source.replaceAll('/.generated/ort-surface/', '/vendor/three/').replaceAll('/node_modules/onnxruntime-web-dev/dist/', '/vendor/ort/'));
}
for (const file of ['wave.onnx', 'wave.json', 'train.py', 'README.md']) await cp(path.join(root, 'experiments/ort-surface', file), path.join(out, file));
await cp(path.join(root, 'experiments/ort-surface/licenses'), path.join(out, 'licenses'), { recursive: true });
await cp(path.join(root, '.local-research/three-ort-gpu-buffer-current-dev-candidate/LICENSE'), path.join(out, 'licenses/Three-LICENSE'));
await cp(path.join(root, 'patches/three-external-storage-buffer.patch'), path.join(out, 'three-external-storage-buffer.patch'));
await cp(path.join(root, 'patches/three-external-storage-buffer.json'), path.join(out, 'three-build-manifest.json'));
for (const file of ['three-external-storage-buffer-current-dev.patch', 'three-external-storage-buffer-current-dev.json']) await cp(path.join(root,'patches',file),path.join(out,file));
await mkdir(path.join(out, 'evidence'), { recursive:true });
for (const file of ['report.json','report.json.sha256']) await cp(path.join(root,'results/development/ort-surface/2026-09-09T16-18-12.004Z-128',file),path.join(out,'evidence',file));
const server = `import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const types = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.wasm':'application/wasm'};
const server = createServer(async (req,res) => {
  const name = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const file = path.resolve(root,'.'+name+(name.endsWith('/')?'index.html':''));
  if (!file.startsWith(root+path.sep)) { res.writeHead(403).end(); return; }
  try { const bytes=await readFile(file); res.writeHead(200,{'Content-Type':types[path.extname(file)]??'application/octet-stream','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'});res.end(bytes); }
  catch { res.writeHead(404).end(); }
});
server.listen(5186,'127.0.0.1',()=>console.log('Open http://127.0.0.1:5186 · Ctrl+C stops the server.'));
`;
await writeFile(path.join(out, 'serve.mjs'), server);
const files = {};
async function inventory(dir) { for (const entry of await readdir(dir, { withFileTypes: true })) {
  const file = path.join(dir, entry.name); if (entry.isDirectory()) await inventory(file);
  else if (entry.name !== 'SHA256.json') { const bytes = await readFile(file); files[path.relative(out, file).replaceAll('\\','/')] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') }; }
} }
await inventory(out);
await writeFile(path.join(out, 'SHA256.json'), JSON.stringify(files,null,2)+'\n');
console.log(`Portable demo: ${out} (${Object.keys(files).length} files)`);

import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const root = '.generated/ort-current-dev-patch-check';
const git = (...args) => execFileSync('git', ['-c','core.autocrlf=false','-C',root,...args], { windowsHide:true, maxBuffer:32e6 });
// The scratch tree contains the original touched files fetched at this exact ref.
const upstreamCommit = 'aaf21735d5eef2321d84ec6623c500ac935cbab7';
const patch = await readFile('patches/three-external-storage-buffer.patch','utf8');
for (const file of [...patch.matchAll(/^--- a\/(.+)$/gm)].map(m=>m[1])) {
  const r = await fetch(`https://raw.githubusercontent.com/mrdoob/three.js/${upstreamCommit}/${file}`);
  if (!r.ok) throw new Error(`${file}: ${r.status}`);
  const text = await r.text();
  if (text !== await readFile(`${root}/${file}`,'utf8')) throw new Error(`Scratch source drift: ${file}`);
}
git('add','.');
git('apply','--exclude=src/renderers/common/Renderer.js','../../patches/three-external-storage-buffer.patch');
const rendererFile = `${root}/src/renderers/common/Renderer.js`;
let renderer = await readFile(rendererFile,'utf8');
const candidateRenderer = await readFile('.local-research/three-ort-gpu-buffer-current-dev-candidate/src/renderers/common/Renderer.js','utf8');
const marker = '\t * @param {BufferAttribute} attribute - The storage buffer attribute to read frm.';
const endMarker = '\t * @return {Promise<ArrayBuffer|ReadbackBuffer>}';
const start = renderer.indexOf(marker), end = renderer.indexOf(endMarker,start);
const candidateStart = candidateRenderer.indexOf('\t * @param {BufferAttribute|ExternalStorageBufferAttribute}');
const candidateEnd = candidateRenderer.indexOf(endMarker,candidateStart);
if ([start,end,candidateStart,candidateEnd].some(i=>i<0)) throw new Error('Renderer documentation anchor missing.');
renderer = renderer.slice(0,start) + candidateRenderer.slice(candidateStart,candidateEnd) + renderer.slice(end);
const disposal = '\t\t\tthis._bindings.dispose();\n';
if (renderer.split(disposal).length !== 2) throw new Error('Renderer teardown anchor is not unique.');
renderer = renderer.replace(disposal, disposal + '\t\t\tthis._attributes.dispose();\n');
await writeFile(rendererFile,renderer);
let rebased = git('diff','--','src','test').toString();
for (const file of git('ls-files','--others','--exclude-standard','src','test').toString().trim().split('\n').filter(Boolean)) {
  const lines = (await readFile(`${root}/${file}`,'utf8')).trimEnd().split('\n');
  rebased += `diff --git a/${file} b/${file}\nnew file mode 100644\n--- /dev/null\n+++ b/${file}\n@@ -0,0 +1,${lines.length} @@\n${lines.map(l=>'+'+l).join('\n')}\n`;
}
await writeFile('patches/three-external-storage-buffer-current-dev.patch',rebased);
// The index still holds the pristine upstream files: check against that baseline.
git('apply','--cached','--check','../../patches/three-external-storage-buffer-current-dev.patch');
const manifest = { upstreamCommit, status:'applies-to-exact-upstream-index',
  sha256:createHash('sha256').update(rebased).digest('hex'),
  validationBoundary:'Mechanical source rebase only. Browser tests use the separately pinned c4ffe022 base.',
  resolution:'Preserve new upstream frameBufferTargets disposal and insert attribute cleanup after binding cleanup.' };
await writeFile('patches/three-external-storage-buffer-current-dev.json',JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify(manifest,null,2));

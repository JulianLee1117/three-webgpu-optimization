// Run only after the app/source cohort is ready. Packaging performs no GPU work.
import assert from 'node:assert/strict';
import {build,version as esbuildVersion} from 'esbuild';
import {readFile,writeFile,mkdir,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const args=process.argv.slice(2);
assert.ok(args.length===0||(args.length===1&&args[0]==='--zip'),'Usage: node scripts/package-footprint-filtering.mjs [--zip]');
const root=await realpath(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'));
const experiment='experiments/footprint-filtering/';
const createdAt=new Date().toISOString();
const destinationName='footprint-filtering-demo-'+createdAt.replaceAll(':','-');
const deliverables=path.join(root,'deliverables'),output=path.join(deliverables,destinationName);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const inside=(base,file)=>{const r=path.relative(base,file);return r!==''&&r!=='..'&&!r.startsWith('..'+path.sep)&&!path.isAbsolute(r);};
assert.ok(inside(deliverables,output));
const payload=new Map(),captured=new Map(),sourceSnapshots=new Map();
const add=(name,bytes)=>{assert.match(name,/^[A-Za-z0-9._-]+$/);assert.ok(!payload.has(name));payload.set(name,Buffer.from(bytes));};
async function source(name){const bytes=await readFile(path.join(root,name));sourceSnapshots.set(name,bytes);return bytes;}
await source('scripts/package-footprint-filtering.mjs');
const three=JSON.parse(await source('node_modules/three/package.json'));
assert.equal(three.version,'0.185.1','This private-node integration is pinned to Three 0.185.1');
const threeSource=await realpath(path.join(root,'node_modules/three/src'));
const entryMap=new Map([['three','Three.js'],['three/webgpu','Three.WebGPU.js'],['three/tsl','Three.TSL.js']]);
const bundled=await build({
  absWorkingDir:root,entryPoints:[experiment+'app.js'],outfile:'app.js',bundle:true,write:false,
  format:'esm',platform:'browser',target:'es2022',minify:true,keepNames:true,legalComments:'eof',metafile:true,logLevel:'silent',
  plugins:[{name:'one-three-source-runtime',setup(builder){
    builder.onResolve({filter:/^three(?:$|\/)/},async ({path:name})=>{
      const relative=entryMap.get(name)??(name.startsWith('three/src/')?name.slice('three/src/'.length):null);
      assert.ok(relative,`Unsupported Three import: ${name}`);
      const resolved=await realpath(path.resolve(threeSource,relative));
      assert.ok(inside(threeSource,resolved),'Three source import escaped src');
      return {path:resolved};
    });
    builder.onLoad({filter:/\.(?:m?js|json)$/},async ({path:file})=>{
      const resolved=await realpath(file);assert.ok(inside(root,resolved),'Bundle input escaped repository');
      const bytes=await readFile(resolved);captured.set(resolved,bytes);
      return {contents:bytes.toString(),loader:file.endsWith('.json')?'json':'js',resolveDir:path.dirname(resolved)};
    });
  }}],
});
assert.equal(bundled.outputFiles.length,1);
assert.ok(Object.values(bundled.metafile.outputs).every(x=>x.imports.length===0),'Bundle has external runtime imports');
const inputs={};
for(const name of Object.keys(bundled.metafile.inputs).sort()){
  const file=await realpath(path.resolve(root,name)),bytes=captured.get(file);
  assert.ok(bytes,'Uncaptured source input: '+name);
  const normalized=path.relative(root,file).split(path.sep).join('/');
  assert.ok(!normalized.startsWith('node_modules/three/build/'),'Bundled Three build creates a second TSL runtime');
  assert.equal(hash(await readFile(file)),hash(bytes),'Source changed during bundling: '+name);
  inputs[normalized]={bytes:bytes.length,sha256:hash(bytes)};
}
// Plain "three" need not be imported by an app that uses only WebGPU/TSL.
for(const name of ['Three.WebGPU.js','Three.TSL.js']) assert.ok(inputs['node_modules/three/src/'+name],`Missing source entry ${name}`);
assert.equal(Object.keys(inputs).filter(n=>n.endsWith('/tsl/TSLCore.js')).length,1,'More than one TSLCore runtime');
add('app.js',bundled.outputFiles[0].contents);
add('bundle.metafile.json',JSON.stringify(bundled.metafile,null,2)+'\n');

const documents=[];
for(const name of ['README.md','PRIOR_ART.md','PROTOCOL.md','RESULTS.md']){
  try {const bytes=await source(experiment+name),destination=name==='README.md'?'RESEARCH.md':name;add(destination,bytes);documents.push({source:experiment+name,file:destination});}
  catch(error){if(error.code!=='ENOENT')throw error;}
}
const htmlSource=(await source(experiment+'app.html')).toString();
assert.equal((htmlSource.match(/<script\s+type="importmap">/g)??[]).length,1,'Unexpected import map layout');
assert.equal((htmlSource.match(/src="\/experiments\/footprint-filtering\/app\.js"/g)??[]).length,1,'Unexpected app entry');
let html=htmlSource.replace(/\s*<script\s+type="importmap">[\s\S]*?<\/script>/,'').replace('src="/experiments/footprint-filtering/app.js"','src="./app.js"');
for(const {source,file}of documents)html=html.replaceAll('href="/'+source+'"','href="./'+file+'"');
html=html.replaceAll('href="/experiments/footprint-filtering/README.md"','href="./README.md"');
assert.ok(!html.includes('importmap')&&!html.includes('/node_modules/')&&!html.includes('src="/experiments/'),'HTML still needs repository modules');
assert.ok(!/\bsrc\s*=\s*["'](?:https?:)?\/\//i.test(html)&&!/@import\b|url\(\s*["']?(?:https?:)?\/\//i.test(html),'HTML has an external runtime asset');
add('index.html',html);
add('THREE-LICENSE.txt',await source('node_modules/three/LICENSE'));
add('THIRD_PARTY_NOTICES.txt','Includes Three.js 0.185.1 under the MIT license; see THREE-LICENSE.txt. Bundle legal comments are retained. This notice applies to Three.js and does not grant a license to the complete research application. esbuild is a packaging tool, not a runtime dependency.\n');

add('serve.mjs',String.raw`import {createServer} from 'node:http';
import {readFile,realpath} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const directory=path.dirname(fileURLToPath(import.meta.url));
export async function startServer(port=5198){
  if(!Number.isInteger(port)||port<0||port>65535)throw Error('Invalid port');
  const root=await realpath(directory),manifest=JSON.parse(await readFile(path.join(root,'manifest.json'),'utf8'));
  const allowed=new Set([...Object.keys(manifest.files),'manifest.json','manifest.sha256']);
  const server=createServer(async(req,res)=>{
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'}).end();return;}
    try{
      let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);if(name==='/')name='/index.html';
      if(name==='/favicon.ico'){res.writeHead(204).end();return;}
      name=name.slice(1);
      if(!/^[A-Za-z0-9._-]+$/.test(name)||!allowed.has(name))throw Error('Unknown route');
      const file=await realpath(path.join(root,name));if(path.dirname(file)!==root)throw Error('Escaped package');
      const bytes=await readFile(file),type=name.endsWith('.html')?'text/html':name.endsWith('.js')||name.endsWith('.mjs')?'text/javascript':name.endsWith('.json')?'application/json':'text/plain';
      res.writeHead(200,{'Content-Type':type+'; charset=utf-8','Content-Length':bytes.length,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(req.method==='HEAD'?undefined:bytes);
    }catch{res.writeHead(404).end('Not found');}
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});return server;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const args=process.argv.slice(2);let value=process.env.PORT??'5198';
  if(args.length){if(args.length===1&&args[0].startsWith('--port='))value=args[0].slice(7);else if(args.length===2&&args[0]==='--port')value=args[1];else throw Error('Usage: node serve.mjs [--port=5198]');}
  if(!/^[0-9]+$/.test(value)||Number(value)<1||Number(value)>65535)throw Error('Port must be 1..65535');
  const server=await startServer(Number(value));console.log('Open http://127.0.0.1:'+server.address().port+'/');
  console.log('The page starts idle. Open the experiment to allocate the GPU. Ctrl+C stops this server.');
}
`);
add('verify.mjs',String.raw`import assert from 'node:assert/strict';
import {readFile,readdir,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=await realpath(path.dirname(fileURLToPath(import.meta.url))),hash=b=>createHash('sha256').update(b).digest('hex');
const bytes=await readFile(path.join(root,'manifest.json'));assert.equal(hash(bytes),(await readFile(path.join(root,'manifest.sha256'),'utf8')).trim());
const manifest=JSON.parse(bytes),expected=[...Object.keys(manifest.files),'manifest.json','manifest.sha256'].sort();
assert.deepEqual((await readdir(root)).sort(),expected);
for(const [name,record]of Object.entries(manifest.files)){
  assert.match(name,/^[A-Za-z0-9._-]+$/);const file=await realpath(path.join(root,name));assert.equal(path.dirname(file),root);
  const data=await readFile(file);assert.equal(data.length,record.bytes);assert.equal(hash(data),record.sha256,name);
}
console.log(JSON.stringify({verified:true,files:expected.length,scope:'File integrity only; no browser or GPU execution'}));
`);
add('README.md',`# Signal Loom: native Three procedural filtering

With Node.js 22.12 or newer, run \`node serve.mjs\` in this extracted folder and open http://127.0.0.1:5198/ in a browser with WebGPU. No npm install, CDN, model download or API key is needed by this bundled runtime. The app remains idle until **Open demo** allocates the GPU. Use **Release GPU** to release it; hiding the page also releases it. A preview stops after twelve seconds.

Use \`node serve.mjs --port=5199\` or the PORT environment variable to select another loopback port. Press Ctrl+C to stop the server. No browser opens automatically.

This prototype transforms an authored native TSL trigonometric color graph into a joint pixel-and-pattern-exposure box average, expanding products before filtering to retain correlated terms. It supports a bounded finite sin/cos algebra with constant amplitudes. Established Fourier integration and automatic shader filtering are prior art; the investigated contribution is a reusable native TSL adapter.

The mathematical integral is exact for affine phases over the declared box. This folded, lit demonstration uses local shader derivatives and integrates albedo before lighting and tone mapping; it is an approximation, and does not integrate geometry, visibility or camera motion. It is not a general material antialiasing system or an ideal Nyquist low-pass filter. No performance advantage is claimed.

The bundle keeps exactly one Three.js 0.185.1 source-module runtime, including private node representations used by the compiler. That version-specific integration is not a stable public Three.js API contract.

${documents.map(d=>`- [${d.file}](${d.file}) is a snapshot of ${d.source}.`).join('\n')}

Those documents may reference raw research files outside this portable runtime. Packaging does not rerun or certify a scientific result. The manifest records the exact runtime and document source hashes; it does not replace their experiment reports. A separate packaged-browser UI check is required before calling this package tested on a GPU.

Run \`node verify.mjs\` to verify all packaged file hashes using only Node built-ins. [bundle.metafile.json](bundle.metafile.json) and [manifest.json](manifest.json) record the resolved source modules and input hashes. Three retains its [MIT license](THREE-LICENSE.txt); this dependency notice does not relicense the research application.
`);

// Refuse a mixed build if any captured source or copied document changed while
// the bundle was being prepared. Output directories are new and never replaced.
for(const [file,bytes]of captured)assert.equal(hash(await readFile(file)),hash(bytes),'Source changed before seal: '+file);
for(const [name,bytes]of sourceSnapshots)assert.equal(hash(await readFile(path.join(root,name))),hash(bytes),'Copied source changed before seal: '+name);
const manifest={kind:'footprint-filtering-portable-v1',createdAt,three:three.version,esbuild:esbuildVersion,
  runtime:'All Three entry points and three/src/* resolve to node_modules/three/src; no bundled build runtime',
  inputHashes:inputs,sourceSnapshots:Object.fromEntries([...sourceSnapshots].map(([name,b])=>[name,{bytes:b.length,sha256:hash(b)}])),
  documents,files:Object.fromEntries([...payload].sort(([a],[b])=>a.localeCompare(b)).map(([name,b])=>[name,{bytes:b.length,sha256:hash(b)}]))};
const manifestBytes=Buffer.from(JSON.stringify(manifest,null,2)+'\n');add('manifest.json',manifestBytes);add('manifest.sha256',hash(manifestBytes)+'\n');
await mkdir(deliverables,{recursive:true});await mkdir(output);
for(const [name,bytes]of payload)await writeFile(path.join(output,name),bytes,{flag:'wx'});
execFileSync(process.execPath,['--check',path.join(output,'app.js')],{windowsHide:true});
execFileSync(process.execPath,['--check',path.join(output,'serve.mjs')],{windowsHide:true});
const verification=JSON.parse(execFileSync(process.execPath,[path.join(output,'verify.mjs')],{encoding:'utf8',windowsHide:true}));
let zip=null;
if(args.includes('--zip')){
  assert.equal(process.platform,'win32','The optional ZIP path uses Windows PowerShell');
  const literal=value=>"'"+value.replaceAll("'","''")+"'";
  const script=`$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression
$base=(Resolve-Path -LiteralPath ${literal(deliverables)}).Path
$target=(Resolve-Path -LiteralPath ${literal(output)}).Path
if (-not $target.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Package escaped deliverables' }
$archivePath=[IO.Path]::GetFullPath($target+'.zip')
if ([IO.Path]::GetDirectoryName($archivePath) -ne $base) { throw 'ZIP escaped deliverables' }
$stream=[IO.File]::Open($archivePath,[IO.FileMode]::CreateNew)
try {
  $archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create,$false)
  try {
    [string[]]$names=Get-ChildItem -LiteralPath $target -File | ForEach-Object { $_.Name }
    [Array]::Sort($names,[StringComparer]::Ordinal)
    foreach($name in $names) {
      $entry=$archive.CreateEntry($name,[IO.Compression.CompressionLevel]::Optimal)
      $entry.LastWriteTime=[DateTimeOffset]::new(2000,1,1,0,0,0,[TimeSpan]::Zero)
      $input=[IO.File]::OpenRead([IO.Path]::Combine($target,$name))
      try { $sink=$entry.Open(); try { $input.CopyTo($sink) } finally { $sink.Dispose() } } finally { $input.Dispose() }
    }
  } finally { $archive.Dispose() }
} finally { $stream.Dispose() }
`;
  execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-EncodedCommand',Buffer.from(script,'utf16le').toString('base64')],{windowsHide:true,timeout:60000});
  const bytes=await readFile(output+'.zip');zip={file:output+'.zip',bytes:bytes.length,sha256:hash(bytes),fixedEntryTimestamp:'2000-01-01T00:00:00Z'};
}
console.log(JSON.stringify({out:output,manifestSha256:hash(manifestBytes),files:payload.size,inputModules:Object.keys(inputs).length,verification,zip,
  gpuOrBrowserRun:false},null,2));

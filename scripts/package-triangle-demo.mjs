import {build, version as esbuildVersion} from 'esbuild';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {mkdir, readFile, realpath, writeFile} from 'node:fs/promises';
import {request} from 'node:http';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const experiment='experiments/gpu-deform-collider/';
const created=new Date().toISOString();
const deliverables=path.join(root,'deliverables');
const output=path.join(deliverables,'triangle-query-demo-'+created.replaceAll(':','-'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const assert=(value,message)=>{if(!value)throw Error(message);};
const inside=(base,file)=>{const rel=path.relative(base,file);return rel!==''&&rel!=='..'&&!rel.startsWith('..'+path.sep)&&!path.isAbsolute(rel);};
const payload=new Map(),inputBytes=new Map();
const add=(name,bytes)=>{assert(!payload.has(name)&&!name.includes('\\')&&!name.split('/').includes('..'),'Invalid package path: '+name);payload.set(name,Buffer.isBuffer(bytes)?bytes:Buffer.from(bytes));};
async function copy(source,destination=source){const bytes=await readFile(path.join(root,source));add(destination,bytes);return bytes;}
async function verifiedReport(directory,expected,sidecarName='report.json.sha256'){
  const bytes=await readFile(path.join(root,directory,'report.json'));
  const sidecar=await readFile(path.join(root,directory,sidecarName));
  const checksum=sidecar.toString().trim().split(/\s+/)[0];
  assert(hash(bytes)===checksum&&(!expected||checksum===expected),'Report checksum mismatch: '+directory);
  return {bytes,sidecar,value:JSON.parse(bytes),sha256:checksum};
}

const three=JSON.parse(await readFile(path.join(root,'node_modules/three/package.json')));
const bvh=JSON.parse(await readFile(path.join(root,'node_modules/three-mesh-bvh/package.json')));
assert(three.version==='0.185.1'&&bvh.version==='0.9.15','Unexpected dependency versions.');
const original=await readFile(path.join(root,'node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js'));
assert(hash(original)==='ddd53d39cca6b095df625448c17e43826087614fab2a56f308cf4c21cb2b7b66','Upstream helper differs from the verified source.');
const patch=await readFile(path.join(root,'patches/three-mesh-bvh-closest-point.patch'));
assert(hash(patch)==='5b4ed0117362211565e3cb49567c454d0bece8ce34da30b0f418cd48f7fabc42','Final patch changed; review evidence before packaging.');
const correction=await readFile(path.join(root,experiment+'closest-point-correction.mjs'));
assert(hash(correction)==='bd635cfc5a5902fce26ed6f655619a2bae03a708653cf1d6183f9e10916159d4','Final correction changed; review evidence before packaging.');

const reportSpecs=[
  {name:'original',directory:'2026-09-09T19-18-49.653Z-wave-16-128',sha256:'3121e489aea3c620ed5f49d3ce73b76f577642499e39c3cc413eddd37c05a4f7',mismatches:10},
  {name:'corrected',directory:'2026-09-09T19-18-51.659Z-wave-16-128',sha256:'75c1b1c87a57dad99a2ff59d8137e56b64f403bd528d2754747925d15a265ce0',mismatches:0},
];
const evidence=[];
for(const spec of reportSpecs){
  const record=await verifiedReport('results/development/gpu-deform-collider/'+spec.directory,spec.sha256);
  const r=record.value,p=r.triangleProbe;
  assert(r.kind==='upstream-triangle-probe'&&p.kind==='isolated-upstream-triangle-probe-v2'&&p.caseCount===200&&p.mismatchCount===spec.mismatches&&p.cpuOracleMismatchCount===0,'Unexpected probe result.');
  assert(!r.errors.length&&!p.errors.length&&!r.changedSources.length,'Probe has GPU errors or changed sources.');
  add('evidence/'+spec.name+'/report.json',record.bytes);
  add('evidence/'+spec.name+'/report.json.sha256',record.sidecar);
  evidence.push({...spec,record:r,path:'evidence/'+spec.name+'/report.json'});
}
const [before,after]=evidence.map(x=>x.record);
const fixtures=r=>r.triangleProbe.cases.map(({id,p,a,b,c,expected,threeCpu})=>({id,p,a,b,c,expected,threeCpu}));
assert(JSON.stringify(fixtures(before))===JSON.stringify(fixtures(after)),'Original/corrected probe inputs differ.');
assert(JSON.stringify(before.triangleProbe.tolerances)===JSON.stringify(after.triangleProbe.tolerances),'Probe tolerances differ.');
const servedKeys=[...new Set([...Object.keys(before.servedSources),...Object.keys(after.servedSources)])];
assert(JSON.stringify(servedKeys.filter(k=>before.servedSources[k]!==after.servedSources[k]))===JSON.stringify(['/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js']),'Unexpected changed dependency in the probe pair.');

const screenshotDirectory='deliverables/triangle-query/2026-09-09T19-28-20.909Z';
const ui=await verifiedReport(screenshotDirectory,null,'report.sha256');
assert(ui.value.status==='passed'&&!ui.value.errors.length&&!ui.value.mobileOverflow,'UI validation did not pass.');
assert(ui.value.checks.length===18&&ui.value.checks.every(x=>x.status.state==='ready'&&!x.status.gpuAllocated&&!x.status.busy),'Incomplete UI fixture/cleanup checks.');
add('evidence/ui/report.json',ui.bytes);add('evidence/ui/report.sha256',ui.sidecar);
await copy(screenshotDirectory+'/desktop.png','media/desktop.png');
await copy(screenshotDirectory+'/mobile.png','media/mobile.png');
const videoDirectory='deliverables/triangle-query-video/2026-09-09T19-32-11.529Z';
const recording=await readFile(path.join(root,videoDirectory,'recording.json'));
const clip=JSON.parse(recording);
assert(clip.frames.length===3&&!clip.errors.length&&clip.frames.map(x=>x.order).join(',')==='ABC,ACB,ABC','Unexpected slide clip evidence.');
add('evidence/ui/recording.json',recording);
await copy(videoDirectory+'/triangle-distance-error.mp4','media/triangle-distance-error.mp4');
await copy('node_modules/three/LICENSE','licenses/three-MIT.txt');
await copy('node_modules/three-mesh-bvh/LICENSE','licenses/three-mesh-bvh-MIT.txt');
add('patches/three-mesh-bvh-closest-point.patch',patch);
add('source/upstream-fns.js',original);add('source/closest-point-correction.mjs',correction);
const demo=await copy(experiment+'triangle-demo.js','source/triangle-demo.js');
await copy(experiment+'triangle-probe.js','source/triangle-probe.js');
await copy(experiment+'triangle-query-counterexample.mjs','source/triangle-query-counterexample.mjs');

const bundled=await build({
  absWorkingDir:root,entryPoints:[experiment+'triangle-demo.js'],outfile:'app.js',
  bundle:true,write:false,format:'esm',platform:'browser',target:'es2022',metafile:true,
  minify:false,treeShaking:true,legalComments:'inline',logLevel:'silent',
  plugins:[{name:'capture-exact-source-bytes',setup(builder){
    builder.onLoad({filter:/\.(?:m?js|json)$/},async args=>{
      assert(inside(root,args.path),'Bundle input escaped repository.');
      const bytes=await readFile(args.path);inputBytes.set(args.path,bytes);
      return {contents:bytes.toString(),loader:args.path.endsWith('.json')?'json':'js',resolveDir:path.dirname(args.path)};
    });
  }}],
});
assert(bundled.outputFiles.length===1,'Expected one JavaScript bundle.');
assert(Object.values(bundled.metafile.outputs).every(x=>x.imports.length===0),'Bundle still requires external runtime imports.');
add('app.js',bundled.outputFiles[0].contents);
add('bundle.metafile.json',JSON.stringify(bundled.metafile,null,2)+'\n');
const bundleInputs={};
for(const name of Object.keys(bundled.metafile.inputs).sort()){
  const absolute=path.resolve(root,name),bytes=inputBytes.get(absolute);
  assert(bytes,'Uncaptured esbuild input: '+name);
  assert(hash(await readFile(absolute))===hash(bytes),'Bundle source changed during packaging: '+name);
  bundleInputs[name]={sha256:hash(bytes),bytes:bytes.length};
}
assert(hash(inputBytes.get(path.join(root,experiment+'triangle-demo.js')))===hash(demo),'Demo source changed during packaging.');
assert(hash(inputBytes.get(path.join(root,experiment+'closest-point-correction.mjs')))===hash(correction),'Correction changed while bundling.');
assert(hash(inputBytes.get(path.join(root,'node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js')))===hash(original),'Upstream helper changed while bundling.');

const sourceHtml=await readFile(path.join(root,experiment+'triangle-demo.html'),'utf8');
assert((sourceHtml.match(/<script type="importmap">/g)??[]).length===1,'Unexpected import map layout.');
assert(sourceHtml.includes('<script type="module" src="./triangle-demo.js"></script>'),'Demo script tag changed.');
const html=sourceHtml.replace(/\s*<script type="importmap">[\s\S]*?<\/script>/,'')
  .replace('<script type="module" src="./triangle-demo.js"></script>','<script type="module" src="./app.js"></script>')
  .replace('href="./TRIANGLE-QUERY-FINDING.md"','href="./FINDING.md"');
assert(!html.includes('node_modules')&&!html.includes('importmap'),'HTML still depends on repository imports.');
add('index.html',html);
add('source/triangle-demo.html',sourceHtml);

add('serve.mjs',String.raw`import {createServer} from 'node:http';
import {readFile,realpath,stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const directory=path.dirname(fileURLToPath(import.meta.url));
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.json':'application/json','.md':'text/plain; charset=utf-8','.txt':'text/plain; charset=utf-8','.sha256':'text/plain; charset=utf-8','.patch':'text/plain; charset=utf-8','.png':'image/png','.mp4':'video/mp4'};
const inside=(base,file)=>{const r=path.relative(base,file);return r!==''&&r!=='..'&&!r.startsWith('..'+path.sep)&&!path.isAbsolute(r);};
export async function startServer(port=5191){
  if(!Number.isInteger(port)||port<0||port>65535)throw Error('Invalid port.');
  const root=await realpath(directory),manifest=JSON.parse(await readFile(path.join(root,'manifest.json'),'utf8'));
  const allowed=new Set([...Object.keys(manifest.files),'manifest.json','manifest.json.sha256']);
  const server=createServer(async(req,res)=>{
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'}).end();return;}
    try{
      let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);
      if(name==='/')name='/index.html';
      if(name.includes(String.fromCharCode(92))||[...name].some(x=>x.charCodeAt(0)<32)||name.split('/').some(x=>x==='.'||x==='..'))throw Error('Invalid path.');
      const relative=name.slice(1);
      if(!allowed.has(relative))throw Error('Unknown path.');
      const file=await realpath(path.resolve(root,relative));
      if(!inside(root,file)||!(await stat(file)).isFile())throw Error('Path escaped package.');
      const bytes=await readFile(file);
      res.writeHead(200,{'Content-Type':types[path.extname(file)]??'application/octet-stream','Content-Length':bytes.length,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
      res.end(req.method==='HEAD'?undefined:bytes);
    }catch{res.writeHead(404).end('Not found.');}
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
  return server;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const args=process.argv.slice(2);let value=process.env.PORT??'5191';
  if(args.length){if(args.length===1&&args[0].startsWith('--port='))value=args[0].slice(7);else if(args.length===2&&args[0]==='--port')value=args[1];else throw Error('Usage: node serve.mjs [--port=5191]');}
  if(!/^[0-9]+$/.test(value)||Number(value)<1||Number(value)>65535)throw Error('Port must be 1..65535.');
  const server=await startServer(Number(value));
  console.log('Triangle query demo: http://127.0.0.1:'+server.address().port+'/');
  console.log('Press Ctrl+C to stop. No GPU work starts until Compare is pressed.');
}
`);

add('verify.mjs',String.raw`import {readFile,readdir,realpath,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';import path from 'node:path';import {fileURLToPath} from 'node:url';
const root=await realpath(path.dirname(fileURLToPath(import.meta.url)));
const hash=b=>createHash('sha256').update(b).digest('hex');
const manifestBytes=await readFile(path.join(root,'manifest.json'));
if(hash(manifestBytes)!==(await readFile(path.join(root,'manifest.json.sha256'),'utf8')).trim())throw Error('Manifest checksum mismatch.');
const manifest=JSON.parse(manifestBytes),seen=[];
async function walk(directory){for(const entry of await readdir(directory,{withFileTypes:true})){const file=path.join(directory,entry.name);if(entry.isSymbolicLink())throw Error('Unexpected symbolic link.');if(entry.isDirectory())await walk(file);else seen.push(path.relative(root,file).split(path.sep).join('/'));}}
await walk(root);
const expected=[...Object.keys(manifest.files),'manifest.json','manifest.json.sha256'].sort();
if(JSON.stringify(seen.sort())!==JSON.stringify(expected))throw Error('Package file inventory differs from manifest.');
for(const [name,record] of Object.entries(manifest.files)){const file=await realpath(path.resolve(root,name)),relative=path.relative(root,file);if(relative==='..'||relative.startsWith('..'+path.sep)||path.isAbsolute(relative))throw Error('Manifest path escaped package.');const bytes=await readFile(file);if(bytes.length!==record.bytes||hash(bytes)!==record.sha256)throw Error('Checksum mismatch: '+name);}
console.log(JSON.stringify({passed:true,files:expected.length,scope:'File integrity only; no GPU or browser work.'}));
`);

add('README.md',`# Triangle closest-point correctness demo

This package runs locally with Node.js 22.12 or newer and a browser with WebGPU support. All JavaScript is already bundled: no npm install, build, CDN, or network download is needed.

1. Extract the zip and open a terminal in this folder.
2. Run \`node serve.mjs\`.
3. Open http://127.0.0.1:5191/ and press **Compare on my GPU**.
4. Try height 0.001 with orders ABC and ACB. The same triangle exposes an order-dependent closest-point bug.

Use \`node serve.mjs --port=5192\` or the PORT environment variable to change the loopback port. Press Ctrl+C to stop the server. The page starts idle; each comparison executes the original and corrected helpers in one bounded GPU dispatch and releases its GPU resources. No browser is launched automatically.

The approximately 500× result is a **distance overestimate**, not a speedup. Contact badges apply the displayed radius threshold to those measured distances. The vertical diagram is magnified for thin triangles; computations use actual coordinates.

Read [FINDING.md](FINDING.md) for the exact counterexample, upstream source, patch, and scope. [Original](evidence/original/report.json) and [corrected](evidence/corrected/report.json) 200-case GPU reports are included unchanged. The corrected helper passed this finite suite; that is not a proof for every geometry, numerical scale, or GPU.

Run \`node verify.mjs\` to check all packaged file hashes without starting a browser or GPU. [manifest.json](manifest.json) records every payload hash and the exact esbuild input hashes; [bundle.metafile.json](bundle.metafile.json) records bundling dependencies. The manifest has its own SHA-256 sidecar. These checks detect changes, not provide a cryptographic signature from a trusted publisher.

[Desktop](media/desktop.png) and [mobile](media/mobile.png) screenshots came from the preceding repository UI validation, whose [raw report](evidence/ui/report.json) is included. The packaging script verifies the static server and artifact integrity without GPU execution; a packaged-browser check is a separate step.

The [six-second slide clip](media/triangle-distance-error.mp4) holds three screenshots of actual GPU comparisons for two seconds each: ABC, ACB, ABC. It is not a frame-rate recording. Its [recording metadata](evidence/ui/recording.json) contains the measured values.

The files under source/ are readable source snapshots for review; their original repository imports are not a second standalone runtime. Run the bundled index.html through serve.mjs. Three.js and three-mesh-bvh retain their MIT notices under [licenses/](licenses/three-MIT.txt); the final patch is [here](patches/three-mesh-bvh-closest-point.patch).
`);
add('FINDING.md',`# Wrong boundary edge in a closest-point query

Confirmed on September 9, 2026 in three-mesh-bvh 0.9.15, pinned upstream commit \`8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab\`. The affected [TSL/WebGPU primitive](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/tsl/fns.js#L33) is called by the library's ordinary closest-point traversal.

For A=(0,0,0), B=(2,0,0), C=(-1,1,0), and P=(1,-2,0), the correct closest point is (1,0,0), distance squared 4. The original GPU helper returns A, distance squared 5. Two projected barycentric coordinates are negative; the first branch selects CA and skips the nearer AB edge. The same triangle fails in 3 of 6 vertex orders (ABC, CAB, CBA).

With C=(-1,0.001,0) and P=(1,-0.002,0), uploaded as f32, the original squared distance is 1.0000040531158447, whereas the CPU answer is 0.0000040000003799796195. The distance is approximately 500 times too large. The corrected GPU answer is (1,0,0), squared distance 0.000004000000444648322. This is geometric error, not speed.

The [original raw GPU report](evidence/original/report.json) has 10 mismatches among 200 cases; the [final corrected report](evidence/corrected/report.json) passes 200/200. The inputs, CPU answers, tolerances, and probe implementation are identical. Only the served primitive module differs. The probe uses one dispatch without BVH traversal, refitting, deformation, or rendering, and includes 72 analytic cases plus 128 deterministic random cases. Both CPU oracles agree; both runs have no GPU errors or changed-source warnings.

- Original report SHA-256: \`${reportSpecs[0].sha256}\`.
- Corrected report SHA-256: \`${reportSpecs[1].sha256}\`.
- Original helper SHA-256: \`${hash(original)}\`.
- Corrected helper SHA-256: \`4b81a1348d16857dc29639ea823c21c7ffa89e80eed03ea19ce28609d2cbe462\`.

The [final patch](patches/three-mesh-bvh-closest-point.patch) uses standard seven-region Voronoi tests and a boundary fallback for zero area. It changes only src/webgpu/tsl/fns.js and was checked against the pinned upstream file. [Original helper](source/upstream-fns.js), [local correction](source/closest-point-correction.mjs), [GPU probe](source/triangle-probe.js), and [CPU reproduction](source/triangle-query-counterexample.mjs) are included for inspection. The installed library was not modified for these experiments. Earlier failed local corrections are retained in the research repository, not substituted into these final reports.

This differs from [issue #914](https://github.com/gkjohnson/three-mesh-bvh/issues/914) and [PR #915](https://github.com/gkjohnson/three-mesh-bvh/pull/915), which corrected barycentric output ordering in August 2026. Related GLSL code merits review, but no WebGL reproduction or fix is claimed here.

The finite tests establish a useful correctness finding and a tested local correction, not a new closest-point algorithm, general floating-point proof, performance improvement or full collision engine. Random near-degenerate triangles are excluded; two explicit thin analytic triangles are included. Other scales and drivers remain unverified. The demo's contact hit/miss applies its chosen radius to the returned unsigned distance.
`);
add('THIRD_PARTY_NOTICES.md',`# Third-party notices

The browser bundle includes Three.js ${three.version} and three-mesh-bvh ${bvh.version}, both MIT licensed. Complete notices are in [licenses/three-MIT.txt](licenses/three-MIT.txt) and [licenses/three-mesh-bvh-MIT.txt](licenses/three-mesh-bvh-MIT.txt). esbuild ${esbuildVersion} was used at packaging time and is not required to run the package. Bundle legal comments are retained.
`);

await mkdir(deliverables,{recursive:true});
assert(inside(await realpath(deliverables),output),'Output escaped deliverables.');
await mkdir(output);
for(const [name,bytes] of payload){const target=path.join(output,name);assert(inside(output,target),'Unsafe output file.');await mkdir(path.dirname(target),{recursive:true});await writeFile(target,bytes,{flag:'wx'});}
const manifest={kind:'triangle-query-static-demo-v1',created,
  provenance:{upstreamCommit:'8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab',three:three.version,threeMeshBvh:bvh.version,
    packageScriptSha256:hash(await readFile(fileURLToPath(import.meta.url))),sourceHtmlSha256:hash(sourceHtml),uiSourceDirectory:screenshotDirectory,
    gpuEvidence:evidence.map(({name,directory,sha256,path})=>({name,directory,sha256,path}))},
  bundle:{esbuildVersion,metafile:'bundle.metafile.json',inputs:bundleInputs},
  integrityScope:'files covers every payload file; manifest.json is covered by manifest.json.sha256. Both manifest files are excluded from files to avoid self-reference.',
  files:Object.fromEntries([...payload.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([name,bytes])=>[name,{sha256:hash(bytes),bytes:bytes.length}]))};
const manifestBytes=JSON.stringify(manifest,null,2)+'\n';
await writeFile(path.join(output,'manifest.json'),manifestBytes,{flag:'wx'});
await writeFile(path.join(output,'manifest.json.sha256'),hash(manifestBytes)+'\n',{flag:'wx'});
for(const name of ['app.js','serve.mjs','verify.mjs'])execFileSync(process.execPath,['--check',path.join(output,name)],{windowsHide:true});
const integrity=JSON.parse(execFileSync(process.execPath,[path.join(output,'verify.mjs')],{encoding:'utf8',windowsHide:true}));
const {startServer}=await import(pathToFileURL(path.join(output,'serve.mjs')));
const server=await startServer(0),port=server.address().port;
const fetchRaw=(url,method='GET')=>new Promise((resolve,reject)=>{const req=request({hostname:'127.0.0.1',port,path:url,method},res=>{const chunks=[];res.on('data',b=>chunks.push(b));res.on('end',()=>resolve({status:res.statusCode,headers:res.headers,bytes:Buffer.concat(chunks)}));});req.on('error',reject);req.end();});
let staticChecks=0;
try{
  for(const name of ['index.html','app.js','FINDING.md','manifest.json','media/desktop.png','evidence/original/report.json']){
    const result=await fetchRaw(name==='index.html'?'/':'/'+name);assert(result.status===200&&hash(result.bytes)===hash(await readFile(path.join(output,name))),'Served bytes mismatch: '+name);staticChecks++;
  }
  const head=await fetchRaw('/app.js','HEAD');assert(head.status===200&&head.bytes.length===0&&Number(head.headers['content-length'])===payload.get('app.js').length,'HEAD failed.');staticChecks++;
  for(const url of ['/missing','/%2e%2e%2fpackage.json','/..%5cpackage.json','/%00','/source/%2e%2e%2f%2e%2e%2fpackage.json','/manifest.json/']){assert((await fetchRaw(url)).status===404,'Containment check failed: '+url);staticChecks++;}
  assert((await fetchRaw('/','POST')).status===405,'Method restriction failed.');staticChecks++;
}finally{await new Promise((resolve,reject)=>server.close(e=>e?reject(e):resolve()));}
for(const [absolute,bytes] of inputBytes)assert(hash(await readFile(absolute))===hash(bytes),'Source changed before sealing: '+path.relative(root,absolute));
assert(hash(await readFile(path.join(root,experiment+'triangle-demo.html')))===hash(sourceHtml),'HTML changed before sealing.');

const zip=output+'.zip';
assert(inside(deliverables,zip),'Archive escaped deliverables.');
const archiveCheck=execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',String.raw`
$ErrorActionPreference='Stop'
$baseRoot=(Resolve-Path -LiteralPath $env:TRIANGLE_PACKAGE_BASE).ProviderPath.TrimEnd([IO.Path]::DirectorySeparatorChar)+[IO.Path]::DirectorySeparatorChar
$bundleRoot=(Resolve-Path -LiteralPath $env:TRIANGLE_PACKAGE_FOLDER).ProviderPath
$archivePath=[IO.Path]::GetFullPath($env:TRIANGLE_PACKAGE_ZIP)
if(-not $bundleRoot.StartsWith($baseRoot,[StringComparison]::OrdinalIgnoreCase) -or -not $archivePath.StartsWith($baseRoot,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive paths escaped deliverables.'}
if(Test-Path -LiteralPath $archivePath){throw 'Archive already exists.'}
Compress-Archive -LiteralPath $bundleRoot -DestinationPath $archivePath
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive=[IO.Compression.ZipFile]::OpenRead($archivePath)
$entryPrefix=[IO.Path]::GetFileName($bundleRoot)+'/'
$checked=0
try{
  foreach($entry in $archive.Entries){
    $entryName=$entry.FullName.Replace([char]92,[char]47)
    if($entryName.EndsWith('/')){continue}
    if(-not $entryName.StartsWith($entryPrefix,[StringComparison]::Ordinal) -or $entryName.Split('/') -contains '..'){throw 'Unsafe archive entry.'}
    $relative=$entryName.Substring($entryPrefix.Length)
    $diskFile=[IO.Path]::GetFullPath((Join-Path $bundleRoot $relative))
    if(-not $diskFile.StartsWith($bundleRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive verification escaped package.'}
    $stream=$entry.Open()
    $algorithm=[Security.Cryptography.SHA256]::Create()
    try{$digest=[BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','')}finally{$stream.Dispose();$algorithm.Dispose()}
    $diskStream=[IO.File]::OpenRead($diskFile)
    $diskAlgorithm=[Security.Cryptography.SHA256]::Create()
    try{$diskDigest=[BitConverter]::ToString($diskAlgorithm.ComputeHash($diskStream)).Replace('-','')}finally{$diskStream.Dispose();$diskAlgorithm.Dispose()}
    if($digest -ne $diskDigest){throw 'Archive entry checksum mismatch.'}
    $checked++
  }
}finally{$archive.Dispose()}
$expected=(Get-ChildItem -LiteralPath $bundleRoot -Recurse -File).Count
if($checked -ne $expected){throw 'Archive omitted package files.'}
Write-Output $checked
`],{encoding:'utf8',windowsHide:true,env:{...process.env,TRIANGLE_PACKAGE_BASE:deliverables,TRIANGLE_PACKAGE_FOLDER:output,TRIANGLE_PACKAGE_ZIP:zip}}).trim();
const zipBytes=await readFile(zip),zipSha256=hash(zipBytes);
await writeFile(zip+'.sha256',zipSha256+'\n',{flag:'wx'});
console.log(JSON.stringify({output,zip,zipBytes:zipBytes.length,zipSha256,files:integrity.files,zipFilesVerified:Number(archiveCheck),staticChecks,bundleBytes:payload.get('app.js').length,bundleInputs:Object.keys(bundleInputs).length,gpuExecuted:false,browserLaunched:false},null,2));

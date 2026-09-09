import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

// This audit imports the independent numeric fixture, never the GPU probe or
// generated compiler. Reading retained pixels does not create a GPU device.
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const ORACLE='experiments/footprint-filtering/fixtures.mjs';
const KINDS=['weave','energy','interference'];
const LANES=['point','compiled','manual','supersample64'];
const WIDTH=96,HEIGHT=64,PIXELS=WIDTH*HEIGHT,ABSOLUTE=5e-4,ALPHA=2e-6;
const f32Case=c=>({...c,scale:c.scale.map(Math.fround),origin:c.origin.map(Math.fround),
  frequency:Math.fround(c.frequency),time:Math.fround(c.time),shutter:Math.fround(c.shutter)});
const CASES=[
  {id:'magnified',scale:[.05,.05],origin:[-.31,.17],frequency:1,time:.137,shutter:0},
  {id:'minified',scale:[2,2],origin:[-.31,.17],frequency:8,time:.137,shutter:0},
  {id:'anisotropic',scale:[3,.4],origin:[-.31,.17],frequency:6,time:.137,shutter:0},
  {id:'combined-shutter',scale:[1.5,1],origin:[-.31,.17],frequency:6,time:.137,shutter:.18},
].map(f32Case);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const insist=(condition,message)=>{if(!condition)throw Error(message);};
const own=(object,key)=>Object.hasOwn(object??{},key);
function close(actual,expected,label,tolerance=1e-12) {
  insist(Number.isFinite(actual)&&Number.isFinite(expected)&&Math.abs(actual-expected)<=tolerance+Math.abs(expected)*1e-9,
    `${label}: retained value disagrees with independent recomputation`);
}
function equal(actual,expected,label) {
  insist(JSON.stringify(actual)===JSON.stringify(expected),`${label}: unexpected protocol value`);
}
function array(value,length,label,{f32=false}={}) {
  insist(Array.isArray(value)&&value.length===length,`${label}: expected ${length} retained values`);
  insist(value.every(n=>typeof n==='number'&&Number.isFinite(n)&&(!f32||Math.fround(n)===n)),`${label}: invalid ${f32?'Float32 ':''}value`);
}
function relativeName(name) {
  insist(typeof name==='string'&&name.length>0&&!name.startsWith('/')&&!name.includes('\\')&&!name.includes(':')&&!name.includes('\0')&&
    name.split('/').every(part=>part!==''&&part!=='.'&&part!=='..'),'Unsafe source/manifest path');
  return name;
}
function digest(value,label) {insist(typeof value==='string'&&/^[0-9a-f]{64}$/.test(value),`${label}: invalid SHA256`);return value;}
async function optional(file) {try{return await readFile(file);}catch(error){if(error.code==='ENOENT')return null;throw error;}}

export async function loadReport(file,{manifestFile}={}) {
  const stored=await readFile(file);
  insist(stored.length<=256*1024*1024,'Report exceeds the 256 MiB input bound');
  const json=file.endsWith('.gz')?gunzipSync(stored,{maxOutputLength:256*1024*1024}):stored;
  const checks={jsonSHA256:sha(json),storedSHA256:sha(stored),rawSidecarVerified:false,manifestVerified:false};
  const sidecarName=path.basename(file).replace(/\.json(?:\.gz)?$/,'.sha256');
  const sidecar=await optional(path.join(path.dirname(file),sidecarName));
  if(sidecar){const expected=digest(sidecar.toString('utf8').trim().split(/\s+/)[0],'report.sha256');
    insist(expected===checks.jsonSHA256,'Raw report SHA256 mismatch');checks.rawSidecarVerified=true;}
  const manifestBytes=manifestFile?await readFile(manifestFile):await optional(path.join(path.dirname(file),'manifest.json'));
  if(manifestBytes){
    const container=JSON.parse(manifestBytes.toString('utf8'));
    insist(container.kind==='footprint-filtering-evidence-v1','Unexpected evidence manifest kind');
    const entries=[container,...(container.relatedReports??[])];
    for(const entry of entries)relativeName(entry.file);
    const manifest=entries.find(entry=>path.resolve(path.dirname(manifestFile??file),entry.file)===path.resolve(file));
    insist(manifest,'Manifest does not describe the selected report');
    relativeName(manifest.file);
    insist(path.resolve(path.dirname(manifestFile??file),manifest.file)===path.resolve(file),'Manifest selects a different report');
    insist(digest(manifest.publicJSONSHA256,'publicJSONSHA256')===checks.jsonSHA256,'Public JSON SHA256 mismatch');
    digest(manifest.originalRawSHA256,'originalRawSHA256');
    if(file.endsWith('.gz'))insist(digest(manifest.gzipSHA256,'gzipSHA256')===checks.storedSHA256,'Compressed report SHA256 mismatch');
    // Redactions require disclosure. The original cannot be verified without
    // the original bytes, even when its hash is retained in a manifest.
    if(manifest.originalRawSHA256!==manifest.publicJSONSHA256)
      insist(Array.isArray(manifest.redactions)&&manifest.redactions.length>0,'Changed public bytes have no disclosed redactions');
    checks.manifestVerified=true;checks.originalBytesUnchanged=manifest.originalRawSHA256===checks.jsonSHA256;
  }
  checks.externallyRecordedChecksumPresent=checks.rawSidecarVerified||checks.manifestVerified;
  return {report:JSON.parse(json.toString('utf8')),checks};
}

export async function auditSources(report) {
  const hashes=report.sourceHashes,snapshots=report.sourceSnapshots,served=report.servedHashes;
  insist(hashes&&snapshots&&served&&typeof hashes==='object'&&typeof snapshots==='object'&&typeof served==='object','Missing source records');
  const required=['scripts/run-footprint-filtering.mjs',ORACLE,'experiments/footprint-filtering/compiler.js',
    'experiments/footprint-filtering/gpu-probe.js','experiments/footprint-filtering/probe.html',
    'experiments/constraint-editing/rebind.js','experiments/material-mips/autograd.js'];
  for(const name of required)insist(own(hashes,name)&&own(snapshots,name),`Missing required source snapshot: ${name}`);
  let verifiedSnapshots=0,matchingLiveSources=0,matchingInstalledDependencies=0,unavailableInstalledDependencies=0;
  const historicalSources=[];
  for(const [name,text]of Object.entries(snapshots)){
    relativeName(name);insist(typeof text==='string'&&own(hashes,name),'Snapshot missing its source hash');
    insist(sha(Buffer.from(text,'utf8'))===digest(hashes[name],name),`Snapshot SHA256 mismatch: ${name}`);verifiedSnapshots++;
  }
  for(const [name,hash]of Object.entries(hashes)){
    relativeName(name);digest(hash,name);
    if(/\.(?:js|mjs|html)$/.test(name))insist(own(snapshots,name),`Missing executable snapshot: ${name}`);
    const live=await optional(path.join(ROOT,name));
    if(live&&sha(live)===hash)matchingLiveSources++;
    else historicalSources.push(name);
  }
  insist(!historicalSources.includes(ORACLE),'Live independent numeric oracle differs from captured fixture source');
  for(const name of required.filter(n=>n!=='scripts/run-footprint-filtering.mjs'))
    insist(served['/'+name]===hashes[name],`Required served/captured source mismatch: ${name}`);
  for(const [route,hash]of Object.entries(served)){
    insist(typeof route==='string'&&route.startsWith('/')&&!route.startsWith('//'),'Unsafe served source path');
    const name=relativeName(route.slice(1));digest(hash,route);
    insist(name.startsWith('experiments/footprint-filtering/')||name.startsWith('node_modules/three/src/')||
      ['experiments/material-mips/autograd.js','experiments/constraint-editing/rebind.js'].includes(name),'Served path outside recorded allowlist');
    if(own(hashes,name))insist(hash===hashes[name],`Served/captured hash mismatch: ${name}`);
    else {
      insist(name.startsWith('node_modules/three/src/'),'Served experiment source lacks a capture');
      const live=await optional(path.join(ROOT,name));
      if(live&&sha(live)===hash)matchingInstalledDependencies++;
      else unavailableInstalledDependencies++;
    }
  }
  insist(matchingInstalledDependencies+unavailableInstalledDependencies>0,'No served Three.js dependency hashes');
  return {passed:true,verifiedSnapshots,matchingLiveSources,historicalSources,matchingInstalledDependencies,unavailableInstalledDependencies,
    liveNumericOracleVerified:true,interpretation:'Hashes establish retained-byte and source-record consistency. They are not independent proof that the GPU run occurred. Non-oracle snapshots may describe historical source; installed dependency mismatches are reported, not silently replaced.'};
}

function comparison(rgba,expected,tolerance=null) {
  let squared=0,maxAbsolute=0,maximumIndex=-1,mismatches=0,maxAlphaError=0;
  const failureExamples=[];
  for(let i=0;i<expected.length;i++){
    const actual=rgba[Math.floor(i/3)*4+i%3],difference=Math.abs(actual-expected[i]);
    squared+=difference*difference;
    if(difference>maxAbsolute){maxAbsolute=difference;maximumIndex=i;}
    if(tolerance!==null&&difference>tolerance){mismatches++;if(failureExamples.length<12)failureExamples.push({pixel:Math.floor(i/3),channel:i%3,actual,expected:expected[i],difference});}
  }
  for(let i=3;i<rgba.length;i+=4)maxAlphaError=Math.max(maxAlphaError,Math.abs(rgba[i]-1));
  return {rmse:Math.sqrt(squared/expected.length),maxAbsolute,maximumIndex,mismatches,nonfinite:0,maxAlphaError,failureExamples};
}
function retainedComparison(actual,expected,label) {
  insist(actual&&typeof actual==='object',`${label}: missing comparison`);
  for(const key of ['rmse','maxAbsolute','maxAlphaError'])close(actual[key],expected[key],`${label}.${key}`);
  for(const key of ['maximumIndex','mismatches','nonfinite'])insist(actual[key]===expected[key],`${label}.${key}: recomputation mismatch`);
  insist(Array.isArray(actual.failureExamples)&&actual.failureExamples.length===expected.failureExamples.length,`${label}: wrong failure examples`);
  for(let i=0;i<expected.failureExamples.length;i++)for(const key of Object.keys(expected.failureExamples[i]))
    close(actual.failureExamples[i][key],expected.failureExamples[i][key],`${label}.failureExamples`);
}
function checkReference(actual,expected,label) {
  array(actual,expected.length,label);
  for(let i=0;i<actual.length;i++)close(actual[i],expected[i],`${label}[${i}]`,1e-13);
}
function settings(actual,expected,label) {
  insist(actual&&Object.keys(actual).length===Object.keys(expected).length,`${label}: missing/extra setting`);
  for(const key of Object.keys(expected))equal(actual[key],expected[key],`${label}.${key}`);
}
function references(kind,s,oracle) {
  const exact=[],point=[],du=[s.scale[0]/WIDTH,0],dv=[0,s.scale[1]/HEIGHT];
  for(let y=0;y<HEIGHT;y++)for(let x=0;x<WIDTH;x++){
    const u=(x+.5)/WIDTH*s.scale[0]+s.origin[0],v=(1-(y+.5)/HEIGHT)*s.scale[1]+s.origin[1];
    point.push(...oracle.numericFixture(kind,u,v,s.time,{frequency:s.frequency}));
    exact.push(...oracle.integrateBox(kind,{u,v,t:s.time,du,dv,dt:s.shutter,frequency:s.frequency}));
  }
  return {exact,point,du,dv};
}

export function auditPixels(report,oracle) {
  insist(report.kind==='footprint-filtering-gpu-report-v1','Unexpected outer report kind');
  const probe=report.probe;insist(probe?.kind==='native-tsl-footprint-gpu-v1','Missing complete GPU probe');
  for(const [key,value]of Object.entries({width:WIDTH,height:HEIGHT,maximumRenders:64,timeoutMs:40000,
    pixelAbsolute:ABSOLUTE,coordinateAbsolute:2e-6,alphaAbsolute:ALPHA}))equal(probe.protocol?.[key],value,`protocol.${key}`);
  insist(Array.isArray(probe.cases)&&probe.cases.length===CASES.length,'Missing/extra protocol case');
  CASES.forEach((s,i)=>settings(probe.cases[i],s,`cases[${i}]`));
  insist(Array.isArray(probe.cells)&&probe.cells.length===KINDS.length*CASES.length,'Missing/extra matrix cell');
  const calibration=probe.calibration,expectedCalibration=[];
  array(calibration?.rgba,PIXELS*4,'calibration.rgba',{f32:true});
  for(let y=0;y<HEIGHT;y++)for(let x=0;x<WIDTH;x++)expectedCalibration.push((x+.5)/WIDTH,1-(y+.5)/HEIGHT,0);
  const calibrationCheck=comparison(calibration.rgba,expectedCalibration,2e-6);
  retainedComparison(calibration.comparison,calibrationCheck,'calibration');
  const cells=[];
  for(const kind of KINDS)for(const s of CASES){
    const cell=probe.cells[cells.length],label=`${kind}/${s.id}`;
    insist(cell?.kind===kind&&cell.id===s.id,`${label}: omitted, duplicated or reordered cell`);
    settings(cell.settings,s,label);const reference=references(kind,s,oracle);
    equal(cell.pixelEdges,{du:reference.du,dv:reference.dv},`${label}.pixelEdges`);
    checkReference(cell.referenceRGB,reference.exact,`${label}.referenceRGB`);
    checkReference(cell.pointReferenceRGB,reference.point,`${label}.pointReferenceRGB`);
    equal(Object.keys(cell.lanes).sort(),[...LANES].sort(),`${label}.lanes`);
    const lanes={};let passed=true;
    for(const lane of LANES){
      const data=cell.lanes[lane];array(data.rgba,PIXELS*4,`${label}.${lane}.rgba`,{f32:true});
      const result=comparison(data.rgba,reference.exact,['compiled','manual'].includes(lane)?ABSOLUTE:null);
      retainedComparison(data.comparison,result,`${label}.${lane}.comparison`);
      if(lane==='point'){
        result.pointParity=comparison(data.rgba,reference.point,ABSOLUTE);
        retainedComparison(data.pointParity,result.pointParity,`${label}.pointParity`);
        passed&&=result.pointParity.mismatches===0;
      }
      passed&&=result.mismatches===0&&result.maxAlphaError<=ALPHA;
      lanes[lane]=result;
    }
    const quality={pointRMSE:lanes.point.rmse,compiledRMSE:lanes.compiled.rmse,manualRMSE:lanes.manual.rmse,supersample64RMSE:lanes.supersample64.rmse,
      reductionVersusPoint:lanes.point.rmse>0?1-lanes.compiled.rmse/lanes.point.rmse:null,
      reductionVersusSupersample64:lanes.supersample64.rmse>0?1-lanes.compiled.rmse/lanes.supersample64.rmse:null};
    for(const [key,value]of Object.entries(quality))value===null?equal(cell.quality?.[key],null,`${label}.quality.${key}`):close(cell.quality?.[key],value,`${label}.quality.${key}`);
    insist(cell.passed===passed,`${label}: retained pass flag disagrees with numeric checks`);
    const utilityRequired=s.id!=='magnified',utilityPassed=!utilityRequired||lanes.compiled.rmse<=lanes.point.rmse*.5;
    cells.push({kind,id:s.id,passed,utilityRequired,utilityPassed,lanes,quality});
  }
  const passedCells=cells.filter(c=>c.passed).length;
  insist(probe.passedCells===passedCells,'Retained passedCells disagrees with recomputation');
  return {numerical:{passed:passedCells===12&&calibrationCheck.mismatches===0&&calibrationCheck.maxAlphaError<=ALPHA,
    passedCells,totalCells:12,calibration:calibrationCheck,comparedRGBValues:12*4*PIXELS*3,referenceRGBValues:12*2*PIXELS*3},
    utility:{passed:cells.every(c=>c.utilityPassed),passedCells:cells.filter(c=>c.utilityRequired&&c.utilityPassed).length,totalCells:9,
      criterion:'Compiled RMSE <= point RMSE / 2 separately in every minified, anisotropic and combined-shutter cell. No timing or universal supersampling advantage claim.'},cells};
}

export function auditExecution(report,numericalPassed) {
  const p=report.probe,l=p?.lifecycle,problems=[];
  const gate=(ok,message)=>{if(!ok)problems.push(message);};
  gate(report.status==='passed','Outer run did not pass');gate(p?.status==='passed'&&numericalPassed,'Probe did not pass independently recomputed numerical gates');
  for(const [label,value]of [['browser errors',report.errors],['request failures',report.requestFailures],['GPU errors',p?.errors]])
    gate(Array.isArray(value)&&value.length===0,`Nonempty or missing ${label}`);
  gate(!report.failure&&!p?.failure,'Recorded execution failure');gate(report.watchdogExpired===false,'Browser watchdog expired or missing');
  gate(l?.disposed===true&&l.busy===false,'Owned device disposal/busy record failed');
  gate(l?.renders===49&&l.maximumRenders===64&&l.width===WIDTH&&l.height===HEIGHT,'Unexpected render count, budget or dimensions');
  gate(typeof report.browser==='string'&&report.browser.length>0,'Missing browser version');
  gate(Number.isFinite(p?.completedWallMs)&&p.completedWallMs>=0,'Invalid descriptive wall time');
  gate(Array.isArray(p?.shaders)&&p.shaders.length>0&&p.shaders.every(s=>typeof s==='string'&&s.length>0),'Missing actual shader text');
  return {passed:problems.length===0,problems,errors:Array.isArray(report.errors)?report.errors.length:null,
    requestFailures:Array.isArray(report.requestFailures)?report.requestFailures.length:null,gpuErrors:Array.isArray(p?.errors)?p.errors.length:null,
    lifecycle:l??null,watchdogExpired:report.watchdogExpired??null,browser:report.browser??null,shaderCount:p?.shaders?.length??0,
    completedWallMs:p?.completedWallMs??null,
    interpretation:'Audits retained execution and cleanup records, not independent hardware attestation. Total wall time includes compilation, CPU references and readback; it is not a lane speed comparison.'};
}

export function selfTest(report,oracle) {
  const tests=[
    ['corrupted compiled pixel',r=>{r.probe.cells[0].lanes.compiled.rgba[0]=Math.fround(r.probe.cells[0].lanes.compiled.rgba[0]+.1);}],
    ['wrong retained box reference',r=>{r.probe.cells[0].referenceRGB[0]+=.1;}],
    ['omitted matrix cell',r=>{r.probe.cells.pop();}],
  ];
  const results=[];
  for(const [name,mutate]of tests){
    const copy=structuredClone(report);mutate(copy);let rejected=false;
    try{const a=auditPixels(copy,oracle);rejected=!a.numerical.passed;}catch{rejected=true;}
    insist(rejected,`Self-test accepted ${name}`);results.push({name,rejected});
  }
  return results;
}

async function main() {
  const args=process.argv.slice(2);
  insist(args.every(a=>a==='--self-test'||a.startsWith('--report=')||a.startsWith('--manifest=')),'Usage: node scripts/analyze-footprint-filtering.mjs [--report=file.json[.gz]] [--manifest=file.json] [--self-test]');
  for(const prefix of ['--report=','--manifest='])insist(args.filter(a=>a.startsWith(prefix)).length<=1,'Duplicate CLI option');
  const value=prefix=>args.find(a=>a.startsWith(prefix))?.slice(prefix.length);
  const file=path.resolve(ROOT,value('--report=')??'experiments/footprint-filtering/evidence/report.json.gz');
  const manifestFile=value('--manifest=')?path.resolve(ROOT,value('--manifest=')):undefined;
  const {report,checks}=await loadReport(file,{manifestFile}),sourceAudit=await auditSources(report);
  const oracle=await import(pathToFileURL(path.join(ROOT,ORACLE)).href),pixels=auditPixels(report,oracle);
  const executionAudit=auditExecution(report,pixels.numerical.passed);
  const tests=args.includes('--self-test')?selfTest(report,oracle):[];
  const summary={kind:'footprint-filtering-cpu-audit-v1',createdAt:new Date().toISOString(),
    passed:pixels.numerical.passed&&pixels.utility.passed&&executionAudit.passed&&sourceAudit.passed,
    inputBasename:path.basename(file),integrity:checks,sourceAudit,executionAudit,...pixels,selfTests:tests,
    scope:'A single retained browser matrix of three affine trigonometric RGB fixtures. Exact box/shutter reference, independently reconstructed from fixed settings. No general shader, geometry, perspective, nonlinear warp or antialiasing guarantee; no speed claim.'};
  const directory=path.join(ROOT,'results/development/footprint-filtering-analysis');await mkdir(directory,{recursive:true});
  const name=summary.createdAt.replaceAll(':','-')+'.json',output=path.join(directory,name),bytes=JSON.stringify(summary,null,2)+'\n';
  await writeFile(output,bytes,{flag:'wx'});await writeFile(output+'.sha256',sha(bytes)+'\n',{flag:'wx'});
  for(const cell of pixels.cells){const q=cell.quality;
    console.log(`${cell.kind}/${cell.id}: point=${q.pointRMSE.toExponential(3)} compiled=${q.compiledRMSE.toExponential(3)} manual=${q.manualRMSE.toExponential(3)} ss64=${q.supersample64RMSE.toExponential(3)} compiled-max=${cell.lanes.compiled.maxAbsolute.toExponential(3)} parity=${cell.passed?'pass':'FAIL'} utility=${cell.utilityRequired?(cell.utilityPassed?'pass':'FAIL'):'n/a'}`);}
  console.log(JSON.stringify({passed:summary.passed,numerical:`${pixels.numerical.passedCells}/12`,utility:`${pixels.utility.passedCells}/9`,
    executionAudit,sourceAudit,integrity:checks,selfTests:tests,summary:path.relative(ROOT,output).replaceAll('\\','/'),sha256:sha(bytes)},null,2));
  if(!summary.passed)process.exitCode=1;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
  main().catch(error=>{console.error('Footprint audit rejected: '+String(error.message).replace(/[A-Z]:[\\/][^\n]*/gi,'<local path>'));process.exitCode=1;});

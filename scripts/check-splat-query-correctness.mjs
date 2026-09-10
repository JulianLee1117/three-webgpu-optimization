// CPU-only invocation of the actual pinned GaussianSplat.raycast implementation.
// Node24 hooks redirect the addon's bare Three imports to the SAME r186 source
// runtime. Variants alter in-memory source only; node_modules is never edited.
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {FIXTURES,ellipsoidIntersection} from '../experiments/splat-query-correctness/reference.mjs';

const root=fileURLToPath(new URL('../',import.meta.url)),packageRoot=path.join(root,'node_modules/three-r186'),
  addon=pathToFileURL(path.join(packageRoot,'examples/jsm/objects/GaussianSplat.js')).href,
  sha=bytes=>createHash('sha256').update(bytes).digest('hex'),source=await readFile(fileURLToPath(addon),'utf8'),
  expectedSourceHash='ea3f148d39413ef31781cdbb1c97a8972b2dce01ef5eebe6ef800c2b1ba7ab42';
assert.equal(sha(source),expectedSourceHash,'Upstream source changed; do not bless a new revision silently');
const oldRadius='SPLAT_KERNEL_CUTOFF * Math.sqrt( Math.max( c00, c11, c22 ) )',
  conservativeRadius='SPLAT_KERNEL_CUTOFF * Math.sqrt( c00 + c11 + c22 + 3 * Math.max( c00, c11, c22 ) * COVARIANCE_FLATNESS )',
  oldPointRadius='SPLAT_KERNEL_CUTOFF * Math.sqrt( maxVariance )',
  conservativePointRadius='SPLAT_KERNEL_CUTOFF * Math.sqrt( c00 + c11 + c22 + 3 * maxVariance * COVARIANCE_FLATNESS )',
  sphereTest='if ( raycaster.ray.intersectsSphere( _sphere ) === false ) return;',
  pointTest='if ( _ray.distanceSqToPoint( center ) > boundingRadius * boundingRadius )';
assert.equal(source.split(oldRadius).length-1,2);assert.equal(source.split(oldPointRadius).length-1,1);
assert.equal(source.split(sphereTest).length-1,1);assert.equal(source.split(pointTest).length-1,1);
const variants={
  original:source,
  geometryBoundsOnly:source.replaceAll(oldRadius,conservativeRadius),
  pointBoundOnly:source.replace(oldPointRadius,conservativePointRadius),
  conservativeBounds:source.replaceAll(oldRadius,conservativeRadius).replace(oldPointRadius,conservativePointRadius),
  bypassSphereRejections:source.replace(sphereTest,'// CPU diagnostic: outer sphere rejection disabled.').replace(pointTest,'if ( false )')
};
const loaded=new Set();
const hook=registerHooks({
  resolve(specifier,context,nextResolve){
    const target=specifier==='three'?'src/Three.js':specifier==='three/webgpu'?'src/Three.WebGPU.js':specifier==='three/tsl'?'src/Three.TSL.js':
      specifier.startsWith('three/addons/')?'examples/jsm/'+specifier.slice('three/addons/'.length):null;
    const result=nextResolve(target?pathToFileURL(path.join(packageRoot,target)).href:specifier,context);
    if(result.url.startsWith(pathToFileURL(packageRoot+path.sep).href))loaded.add(result.url.split('?')[0]);
    return result;
  },
  load(url,context,nextLoad){
    if(url.startsWith(addon+'?variant=')){
      const name=new URL(url).searchParams.get('variant');assert.ok(name in variants);
      return {format:'module',source:variants[name],shortCircuit:true};
    }
    return nextLoad(url,context);
  }
});
const THREE=await import('three/webgpu'),{createGaussianSplatGeometry}=await import(pathToFileURL(path.join(packageRoot,'examples/jsm/utils/GaussianSplatUtils.js')).href);
assert.equal(THREE.REVISION,'186');
const classes={};for(const name of Object.keys(variants))classes[name]=(await import(name==='original'?addon:addon+'?variant='+name)).GaussianSplat;
function query(GaussianSplat,input,{outerSphereOverride=false}={}){
  const geometry=createGaussianSplatGeometry(new Float32Array([0,0,0]),Float32Array.from(input.covariance),new Uint8Array([255,255,255,input.opacity])),
    object=new GaussianSplat(geometry),raycaster=new THREE.Raycaster(new THREE.Vector3(...input.origin),new THREE.Vector3(...input.direction),0,10),hits=[];
  object.updateMatrixWorld(true);object.computeBoundingSphere();
  const computedSphere={center:object.boundingSphere.center.toArray(),radius:object.boundingSphere.radius};
  if(outerSphereOverride)object.boundingSphere=new THREE.Sphere(new THREE.Vector3(),100);
  object.raycast(raycaster,hits);
  const output={computedSphere,outerSphereOverride,hits:hits.map(hit=>({distance:hit.distance,point:hit.point.toArray(),index:hit.index})),
    geometryBounds:{min:object.boundingBox.min.toArray(),max:object.boundingBox.max.toArray()}};
  geometry.dispose();object.geometry.dispose();object.material.dispose();
  return output;
}
const report={kind:'three-r186-gaussian-raycast-sphere-bound-cpu-v1',created:new Date().toISOString(),
  node:process.version,scope:'Actual upstream CPU raycast; no renderer, GPU adapter, device, browser, or visual-quality assertion.',
  upstream:{version:'0.186.0',tag:'r186',annotatedTagSha:'819fadd6b663b74d828c6af72a543024f74d3877',
    commit:'148ef33ecb6d2502ff796d4554abd1549c95d519',tagDate:'2026-09-08T19:12:48Z',
    url:'https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js',
    sha256:expectedSourceHash,source},
  constants:{SPLAT_KERNEL_CUTOFF:2,COVARIANCE_FLATNESS:1e-4,MIN_RAYCAST_OPACITY:.2},
  comparisonTolerance:{distanceAbsolute:1e-10,pointAbsolute:1e-10,oracleSurfaceResidual:1e-10},
  variants:Object.fromEntries(Object.entries(variants).map(([name,text])=>[name,{sha256:sha(text),bytes:Buffer.byteLength(text)}])),cases:[],sources:{},resolvedSourceHashes:{}};
try{
  for(const input of FIXTURES){
    const oracle=ellipsoidIntersection(input.covariance,input.origin,input.direction),expectedHit=oracle.hit&&input.opacity/255>=.2,
      runs=Object.fromEntries(Object.entries(classes).map(([name,Class])=>[name,query(Class,input)]));
    runs.originalWithOuterSphereOverride=query(classes.original,input,{outerSphereOverride:true});
    for(const name of ['conservativeBounds','bypassSphereRejections']){
      const hits=runs[name].hits;assert.equal(hits.length,expectedHit?1:0,input.name+' '+name+' hit status');
      if(expectedHit){assert.ok(Math.abs(hits[0].distance-oracle.distance)<=1e-10);for(let j=0;j<3;j++)assert.ok(Math.abs(hits[0].point[j]-oracle.point[j])<=1e-10);}
    }
    if(oracle.hit)assert.ok(oracle.residual<=1e-10);
    report.cases.push({input,oracle,expectedHit,runs});
  }
  const defective=report.cases.filter(c=>c.expectedHit&&c.runs.original.hits.length===0);
  assert.deepEqual(defective.map(c=>c.input.name),['rotated-45deg-true-hit','rotated-three-axes-true-hit']);
  for(const c of defective)for(const name of ['geometryBoundsOnly','pointBoundOnly','originalWithOuterSphereOverride'])assert.equal(c.runs[name].hits.length,0);
  report.status='defect-confirmed';report.summary={cases:report.cases.length,originalFalseNegatives:defective.length,
    conservativeVariantMismatches:0,narrowPhaseDiagnosticMismatches:0,
    inference:'Both the outer object sphere and per-splat sphere rejection underbound rotated anisotropic covariance ellipsoids. Enlarging either one alone is insufficient.',
    limits:'Six fixed CPU fixtures, identity object transforms, opaque/faint controls. The trace correction is a conservative diagnostic, not a performance-optimal or universally validated upstream patch.'};
}catch(error){report.status='verification-failed';report.failure=String(error.stack??error).replaceAll(root,'<workspace>/');}
hook.deregister();
for(const file of ['scripts/check-splat-query-correctness.mjs','experiments/splat-query-correctness/reference.mjs']){
  const bytes=await readFile(path.join(root,file));report.sources[file]={sha256:sha(bytes),text:bytes.toString('utf8')};
}
for(const url of [...loaded].sort())report.resolvedSourceHashes[path.relative(root,fileURLToPath(url)).replaceAll('\\','/')]=sha(readFileSync(fileURLToPath(url)));
assert.ok(Object.keys(report.resolvedSourceHashes).every(file=>file.startsWith('node_modules/three-r186/')),'Mixed Three runtimes');
const directory=path.join(root,'experiments/splat-query-correctness/evidence',report.created.replaceAll(':','-'));
await mkdir(directory,{recursive:true});
const encoded=JSON.stringify(report,null,2)+'\n';
await writeFile(path.join(directory,'report.json'),encoded,{flag:'wx'});
await writeFile(path.join(directory,'report.sha256'),`${sha(encoded)}  report.json\n`,{flag:'wx'});
console.log(JSON.stringify({status:report.status,summary:report.summary,report:path.relative(root,path.join(directory,'report.json')),sha256:sha(encoded),
  cases:report.cases.map(c=>({name:c.input.name,oracle:c.oracle.distance??null,original:c.runs.original.hits.length,corrected:c.runs.conservativeBounds.hits.length}))}));
if(report.status!=='defect-confirmed')process.exitCode=1;

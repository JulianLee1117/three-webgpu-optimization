// CPU-only verification of the sealed evidence. Never starts a browser or GPU.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';

const root = new URL('../', import.meta.url);
const base = new URL('experiments/material-mips/evidence/', root);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const close = (a,b,label,t=1e-11) => assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=t*(1+Math.abs(b)), label);
const finiteArray = (a,n) => { assert.equal(a.length,n); assert.ok(a.every(Number.isFinite)); };
const read = name => {
  assert.match(name,/^[a-zA-Z0-9._-]+$/);
  return readFile(new URL(name,base));
};
assert.ok(process.argv.slice(2).every(x=>x==='--numeric'), 'Only --numeric is supported');
const numeric = process.argv.includes('--numeric');
const manifestBytes = await read('manifest.json');
assert.equal(sha(manifestBytes),(await read('manifest.sha256')).toString().trim());
const manifest = JSON.parse(manifestBytes);
assert.equal(manifest.format,'material-mips-portable-evidence-v1');
for(const entry of manifest.files) {
  const bytes=await read(entry.file);
  assert.equal(bytes.length,entry.bytes); assert.equal(sha(bytes),entry.sha256,entry.file);
}
assert.equal(manifest.verifier.file,'scripts/analyze-material-mips.mjs');
assert.equal(sha(await readFile(new URL(import.meta.url))),manifest.verifier.sha256,'Verifier source hash');
const pins = {
  'load-failure':'04ec2f793dd544cf78621f91a2b2d65415f0bfc0f5a9772df203143ea7870426',
  'forward-parity':'bfc341e58792f801a324b520ac3771a0ce319f8f83800a1646e12fbbf75ee24e',
  'hardware-fd':'d258a7e46b12c3ff14d6555c20c5aec1eb4a74f53c34517f7265ac2beca82902',
  'cpu-utility':'048d849e153d036c0f1dff59974d13415bf242a47e2689a322d9c9dc7b2fa72b',
};
assert.equal(manifest.reports.length,4);
const reports = {};
for (const entry of manifest.reports) {
  assert.equal(entry.rawSha256,pins[entry.id]);
  assert.ok(!reports[entry.id]);
  const compressed = await read(entry.archive);
  assert.equal(sha(compressed),entry.archiveSha256); assert.equal(compressed.length,entry.archiveBytes);
  const bytes = gunzipSync(compressed,{maxOutputLength:16_000_000});
  assert.equal(sha(bytes),entry.portableSha256); assert.equal(bytes.length,entry.portableBytes);
  if (!entry.sanitization.changes.length) assert.equal(entry.portableSha256,entry.rawSha256);
  const text = bytes.toString();
  assert.ok(!/(?:[A-Z]:[\\/]Users[\\/]|\/Users\/|\/home\/|PRIVATE KEY|(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,})/i.test(text), 'Unexpected private path or credential-like string');
  reports[entry.id] = JSON.parse(text);
}
const summary = {kind:'material-mips-evidence-analysis-v1',verified:true,gpuRerun:false,numericRecomputation:numeric,
  originalStatuses:Object.fromEntries(Object.entries(reports).map(([k,v])=>[k,v.status])),gpu:[],cpu:[],failedCaseDiagnostics:[]};
const load = reports['load-failure'];
assert.equal(load.status,'failed'); assert.equal(load.probe,undefined);
assert.equal(Object.keys(load.servedHashes).length,1); assert.match(load.failure,/Timeout 20000ms/);

// Reproduce the frozen classifier, then independently audit the failed cases
// with normalized light and all lower AND upper dot-product clamp distances.
const normalize = a => { const d=Math.hypot(...a); return a.map(v=>v/d); };
function cell(point,L,fullClampMargin=false) {
  const N=normalize([point[0],point[1],1]);
  const nl=N.reduce((s,v,i)=>s+v*L[i],0),nv=N[2];
  const clamp=v=>Math.min(1,Math.max(0,v)),r=Math.min(1,Math.max(.0525,point[2]));
  const uv=[r,clamp(nv),r,clamp(nl)];
  const phase=v=>v<=0?'low':v>=1?'high':'inside';
  const signature=[point[2]<=.0525?'rough-low':point[2]>=1?'rough-high':'rough-inside',phase(nl),phase(nv),...uv.map(v=>Math.floor(16*v-.5))].join('/');
  const distances=[...uv.map(v=>Math.abs(16*v-.5-Math.round(16*v-.5))/16),Math.abs(point[2]-.0525),Math.abs(point[2]-1),Math.abs(nl),Math.abs(nv)];
  if(fullClampMargin) distances.push(Math.abs(1-nl),Math.abs(1-nv));
  return {signature,margin:Math.min(...distances)};
}

for (const id of ['forward-parity','hardware-fd']) {
  const outer=reports[id],r=outer.probe,t=r.tolerances;
  assert.deepEqual(outer.errors,[]); assert.deepEqual(r.errors,[]); assert.equal(r.disposed,true);
  finiteArray(r.parameters,512); assert.equal(r.lights.length,4);
  const out={id,cases:128,lights:4,stockPixelComponents:0,stockPixelMismatches:0,cpuForwardMismatches:0,cpuGradientMismatches:0,
    maxStockPixelError:0,fdComparisons:0,finestSmoothChecked:0,finestBoundaryExcluded:0,fdFailures:0,nonfinite:0,perLight:[]};
  for(const [li,l] of r.lights.entries()) {
    finiteArray(l.compute,512); finiteArray(l.derivatives,512); finiteArray(l.pixelCenters,384); finiteArray(l.cpu,384); finiteArray(l.cpuGradients,384);
    let pixelMax=0,cpuMax=0,gradientMax=0;
    for(let i=0;i<128;i++) for(let k=0;k<3;k++) {
      const actual=l.compute[4*i+k],pixel=l.pixelCenters[3*i+k],cpu=l.cpu[3*i+k],ad=l.derivatives[4*i+k],ref=l.cpuGradients[3*i+k];
      const pe=Math.abs(actual-pixel),ce=Math.abs(actual-cpu),ge=Math.abs(ad-ref);
      pixelMax=Math.max(pixelMax,pe); cpuMax=Math.max(cpuMax,ce); gradientMax=Math.max(gradientMax,ge);
      out.stockPixelComponents++; out.stockPixelMismatches+=Number(pe>t.pixelAbsolute+t.pixelRelative*Math.abs(pixel));
      out.cpuForwardMismatches+=Number(ce>t.cpuAbsolute+t.cpuRelative*Math.abs(cpu));
      out.cpuGradientMismatches+=Number(ge>t.gradientAbsolute+t.gradientRelative*Math.abs(ref));
    }
    close(pixelMax,l.maxPixelError,'pixel maximum'); close(cpuMax,l.maxCPUError,'CPU maximum'); close(gradientMax,l.maxGradientError,'gradient maximum');
    out.maxStockPixelError=Math.max(out.maxStockPixelError,pixelMax);
    if(id==='forward-parity') { assert.equal(l.gpuFiniteDifferences,undefined); assert.equal(l.mismatches.length,0); continue; }
    assert.deepEqual(r.gpuFiniteDifferenceProtocol.steps,[.002,.001,.0005]);
    const p=r.gpuFiniteDifferenceProtocol; assert.equal(p.acceptedStep,.0005); assert.equal(p.absolute,.002); assert.equal(p.relative,.025); assert.equal(p.boundaryMargin,5e-6);
    assert.equal(l.gpuFiniteDifferences.length,9);
    const local={finestSmoothChecked:0,finestCrossedOrNearBoundary:0,nonfinite:0,failures:0,maxSmoothError:0};
    for(const [ei,e] of l.gpuFiniteDifferences.entries()) {
      assert.equal(e.h,p.steps[Math.floor(ei/3)]); assert.equal(e.parameter,ei%3);
      finiteArray(e.plus,512); finiteArray(e.minus,512); finiteArray(e.plusParameters,384); finiteArray(e.minusParameters,384); assert.equal(e.comparisons.length,128);
      for(let i=0;i<128;i++) {
        const center=r.parameters.slice(4*i,4*i+3),a=[...center],b=[...center];
        a[e.parameter]=Math.fround(a[e.parameter]+Math.fround(e.h)); b[e.parameter]=Math.fround(b[e.parameter]-Math.fround(e.h));
        assert.deepEqual(e.plusParameters.slice(3*i,3*i+3),a); assert.deepEqual(e.minusParameters.slice(3*i,3*i+3),b);
        const states=[center,a,b].map(q=>cell(q,l.light)),crossed=states.some(s=>s.signature!==states[0].signature),near=states.some(s=>s.margin<5e-6);
        const denominator=a[e.parameter]-b[e.parameter],fd=(e.plus[4*i+3]-e.minus[4*i+3])/denominator,ad=l.derivatives[4*i+e.parameter];
        const error=Math.abs(fd-ad),tolerance=.002+.025*Math.abs(ad),checked=e.h===.0005&&!crossed&&!near;
        const c=e.comparisons[i]; assert.equal(c.case,i); assert.equal(c.finite,true); close(c.denominator,denominator,'FD denominator'); close(c.finiteDifference,fd,'FD value'); close(c.ad,ad,'AD value');
        close(c.error,error,'FD error'); close(c.tolerance,tolerance,'FD tolerance');
        assert.deepEqual(c.cellSignatures,states.map(s=>s.signature)); close(c.minimumBoundaryMargin,Math.min(...states.map(s=>s.margin)),'cell margin');
        assert.equal(c.crossedCellOrClamp,crossed); assert.equal(c.nearBoundary,near); assert.equal(c.checked,checked); assert.equal(c.passed,checked?error<=tolerance:null);
        out.fdComparisons++;
        if(e.h===.0005) {
          if(!checked) local.finestCrossedOrNearBoundary++;
          else { local.finestSmoothChecked++; local.maxSmoothError=Math.max(local.maxSmoothError,error); if(error>tolerance) {
            local.failures++;
            const audit=[center,a,b].map(q=>cell(q,normalize(l.light),true));
            assert.ok(audit.every(s=>s.signature===states[0].signature&&s.margin>5e-6),'Independent classifier audit');
            summary.failedCaseDiagnostics.push({light:li,case:i,parameter:e.parameter,point:center,ad,cpuIdealDerivative:l.cpuGradients[3*i+e.parameter],
              independentClassifierMinimumMargin:Math.min(...audit.map(s=>s.margin)),series:l.gpuFiniteDifferences.filter(x=>x.parameter===e.parameter).map(x=>({h:x.h,...x.comparisons[i]}))});
          } }
        }
      }
    }
    assert.deepEqual(local,l.gpuFiniteDifferenceSummary); assert.equal(l.mismatches.length,local.failures);
    assert.ok(l.mismatches.every(m=>m.kind==='gpu-forward-fd'));
    out.finestSmoothChecked+=local.finestSmoothChecked;out.finestBoundaryExcluded+=local.finestCrossedOrNearBoundary;out.fdFailures+=local.failures;out.perLight.push(local);
  }
  assert.equal(out.stockPixelMismatches,0); assert.equal(out.cpuForwardMismatches,0); assert.equal(out.cpuGradientMismatches,0);
  assert.equal(r.mismatchCount,id==='hardware-fd'?5:0); assert.equal(out.fdFailures,id==='hardware-fd'?5:0);
  if(id==='hardware-fd') { assert.equal(out.fdComparisons,4608); assert.equal(out.finestSmoothChecked,1521); assert.equal(out.finestBoundaryExcluded,15); }
  assert.equal(outer.status,id==='hardware-fd'?'failed':'passed'); summary.gpu.push(out);
}
assert.deepEqual(reports['forward-parity'].probe.parameters,reports['hardware-fd'].probe.parameters);
assert.deepEqual(reports['forward-parity'].probe.lights.map(l=>l.light),reports['hardware-fd'].probe.lights.map(l=>l.light));

let numericBRDF;
if(numeric) {
  // Pin the oracle to the retained GPU cohort. The earlier CPU report recorded
  // a different complete shading.js file; report this explicitly, then verify
  // every retained CPU metric instead of asserting historical source identity.
  const cpu=reports['cpu-utility'];
  for(const [name,expected] of Object.entries(cpu.sourceHashes)) {
    if(name==='scripts/check-material-mips.mjs'||name==='experiments/material-mips/fixtures.mjs') continue;
    const current=sha(await readFile(new URL(name,root)));
    if(name==='experiments/material-mips/shading.js') {
      const gpuHash=reports['hardware-fd'].sourceHashes[name];
      assert.equal(current,gpuHash,`Numeric source changed since GPU cohort: ${name}`);
      summary.numericOracleSource={file:name,currentSha256:current,cpuReportSha256:expected,gpuReportSha256:gpuHash,
        matchesCPUReport:current===expected,note:'A differing full bridge source hash is not historical source identity; all retained CPU aggregate and per-texel metrics are recomputed below.'};
    } else assert.equal(current,expected,`Numeric dependency changed: ${name}`);
  }
  ({numericBRDF}=await import('../experiments/material-mips/shading.js'));
  const scalar=(q,L)=>numericBRDF([q[0],q[1],1],q[2],L,[0,0,1]).reduce((s,v,k)=>s+v*[.3,.5,.2][k],0);
  for(const d of summary.failedCaseDiagnostics) {
    const l=reports['hardware-fd'].probe.lights[d.light];
    for(const s of d.series) {
      const e=l.gpuFiniteDifferences.find(x=>x.h===s.h&&x.parameter===d.parameter),i=d.case;
      const a=e.plusParameters.slice(3*i,3*i+3),b=e.minusParameters.slice(3*i,3*i+3),cp=scalar(a,l.light),cm=scalar(b,l.light);
      s.cpuPlus=cp; s.cpuMinus=cm; s.gpuPlus=e.plus[4*i+3]; s.gpuMinus=e.minus[4*i+3];
      s.cpuFiniteDifference=(cp-cm)/s.denominator; s.cpuFiniteDifferenceADError=Math.abs(s.cpuFiniteDifference-d.ad);
      s.plusForwardResidual=s.gpuPlus-cp; s.minusForwardResidual=s.gpuMinus-cm;
      s.residualAmplification=(s.plusForwardResidual-s.minusForwardResidual)/s.denominator;
      close(s.finiteDifference-s.cpuFiniteDifference,s.residualAmplification,'FD residual decomposition');
    }
  }
}
const cpu=reports['cpu-utility']; assert.equal(cpu.status,'failed'); assert.equal(cpu.sourceUnchanged,true); assert.equal(cpu.configuration.minimumHeldoutMSEImprovement,.3);
assert.deepEqual(cpu.texels,[56,67,89,96,99,192,195,223]); assert.equal(cpu.fixtures.length,2);
for(const f of cpu.fixtures) {
  assert.equal(f.inputs.trainPairs.length,32); assert.equal(f.inputs.heldoutPairs.length,64);
  assert.equal(f.inputs.trainTargets.length,8*32*4); assert.equal(f.inputs.heldoutTargets.length,8*64*4);
  assert.ok(f.optimizations.every(o=>o.completed===128&&o.trace.length===128));
  const variants={...f.baselines,learned:f.learned};
  for(const [name,v] of Object.entries(variants)) for(const split of ['train','heldout']) {
    const metrics=v[split]; assert.equal(metrics.perTexelMSE.length,8);
    close(metrics.perTexelMSE.reduce((a,b)=>a+b,0)/8,metrics.mse,`${name} aggregate`); close(Math.sqrt(metrics.mse),metrics.rmse,`${name} RMSE`);
    if(numeric) {
      const pairs=f.inputs[split+'Pairs'],targets=f.inputs[split+'Targets'],perTexelMSE=[]; let total=0,peak=0;
      for(let i=0;i<8;i++) {
        let local=0; const w=v.parameters.slice(4*i,4*i+3),N=normalize([w[0],w[1],1]);
        for(const [j,pair] of pairs.entries()) {
          const rgb=numericBRDF(N,w[2],pair.light,pair.view);
          for(let k=0;k<3;k++){const error=rgb[k]-targets[4*(i*pairs.length+j)+k];local+=error*error;peak=Math.max(peak,Math.abs(error));}
        }
        total+=local; perTexelMSE.push(local/(pairs.length*3));
      }
      close(total/(8*pairs.length*3),metrics.mse,`${f.kind}/${name}/${split} numeric MSE`); close(peak,metrics.maxAbsolute,'numeric peak');
      perTexelMSE.forEach((v,i)=>close(v,metrics.perTexelMSE[i],'numeric per-texel MSE'));
    }
  }
  const [bestName,best]=Object.entries(f.baselines).sort((a,b)=>a[1].heldout.mse-b[1].heldout.mse)[0];
  const improvement=1-f.learned.heldout.mse/best.heldout.mse;
  assert.equal(bestName,f.strongestBaseline); close(improvement,f.heldoutMSEImprovement,'utility improvement');
  assert.equal(f.passed,improvement>=.3); assert.equal(f.passed,false);
  summary.cpu.push({fixture:f.kind,texels:8,bestBaseline:bestName,baselineHeldoutMSE:best.heldout.mse,learnedHeldoutMSE:f.learned.heldout.mse,heldoutMSEImprovement:improvement,passed:false});
}
summary.verdict='Stock forward parity passed; the ideal-bilinear AD path passed CPU checks but failed five fixed hardware-forward FD checks. Both material-mip CPU utility fixtures failed. No universal or exact-hardware-gradient claim.';
console.log(JSON.stringify(summary,null,2));

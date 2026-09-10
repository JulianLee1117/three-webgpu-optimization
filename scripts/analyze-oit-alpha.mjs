// SPDX-License-Identifier: MIT
// CPU-only verification of retained GPU evidence; this never creates a browser.
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const sha=raw=>createHash('sha256').update(raw).digest('hex');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const EXPECTED=[['one-half-transparent',0,[.5]],['empty-transparent',0,[]],['one-half-opaque-background',1,[.5]],['two-half-transparent',0,[.5,.5]],['one-half-quarter-background',.25,[.5]]];

export function auditReport(report){
  assert.equal(report.kind,'oit-alpha-correctness-report-v1');assert.equal(report.status,'passed');
  assert.equal(report.probe.revision,'186');assert.equal(report.probe.status,'passed');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.requestFailures,[]);assert.deepEqual(report.probe.validationErrors,[]);assert.equal(report.watchdogExpired,false);
  assert.deepEqual(report.probe.lifecycle,{rendererCount:1,fixtureLaneRenders:15,rendererRenderCalls:35,disposed:true,deviceLossReason:'destroyed'});
  assert.deepEqual(report.probe.fixtures.map(f=>[f.id,f.backgroundAlpha,f.opacities]),EXPECTED);
  assert.equal(report.probe.cells.length,15);
  const original=report.sourceSnapshots[report.correction.source];
  assert.equal(sha(original),report.correction.originalSHA256);
  assert.equal(original.split(report.correction.before).length,2);
  assert.equal(report.correction.before,'return vec4( mix( accumColor, beautyNode.rgb, revealageNode ), beautyNode.a );');
  assert.equal(report.correction.after,'return vec4( mix( accumColor, beautyNode.rgb, revealageNode ), float( 1 ).sub( revealageNode ).add( beautyNode.a.mul( revealageNode ) ) );');
  assert.equal(sha(original.replace(report.correction.before,report.correction.after)),report.correction.correctedSHA256);
  assert.equal(report.servedHashes['/'+report.correction.source],report.correction.originalSHA256);
  assert.equal(report.servedHashes[report.correction.servedPath],report.correction.correctedSHA256);
  for(const[name,digest]of Object.entries(report.sourceHashes))assert.equal(sha(report.sourceSnapshots[name]),digest,`Source snapshot ${name}`);
  const rows=[];let pixelsChecked=0;
  for(const[id,backgroundAlpha,opacities]of EXPECTED){
    let transmittance=1-backgroundAlpha;for(const opacity of opacities)transmittance*=1-opacity;
    const expected=1-transmittance,packed={};
    for(const lane of ['direct','stock','corrected']){
      const cells=report.probe.cells.filter(c=>c.fixture===id&&c.lane===lane);assert.equal(cells.length,1);const cell=cells[0];
      const bytes=Buffer.from(cell.pixelsBase64,'base64'),raw=Buffer.from(cell.rawReadback.base64,'base64');
      assert.equal(bytes.length,4096);assert.equal(raw.length,8064);assert.equal(cell.rawReadback.bytesPerRow,256);assert.equal(cell.rawReadback.byteLength,8064);
      for(let y=0;y<32;y++)assert.deepEqual(bytes.subarray(y*128,(y+1)*128),raw.subarray(y*256,y*256+128),'Padded readback packing');
      let error=0;for(let i=3;i<bytes.length;i+=4){error=Math.max(error,Math.abs(bytes[i]-expected*255));pixelsChecked++;}
      assert.equal(cell.expectedAlpha,expected);assert.equal(cell.matchesPorterDuff,error<=2);
      if(lane!=='stock')assert.ok(error<=2,`${lane} alpha oracle failed`);
      packed[lane]=bytes;rows.push({fixture:id,lane,expectedAlpha:expected,actualAlphaByte:bytes[3],allPixelMaxErrorBytes:error});
    }
    for(let i=0;i<4096;i++)if(i%4!==3)assert.equal(packed.stock[i],packed.corrected[i],'Correction changed RGB');
    assert.deepEqual(packed.direct,packed.corrected,'Corrected RGBA differs from normal blending');
  }
  const failures=rows.filter(c=>c.lane==='stock'&&c.allPixelMaxErrorBytes>2).map(c=>c.fixture);
  assert.deepEqual(failures,['one-half-transparent','two-half-transparent','one-half-quarter-background']);
  return {status:'passed',pixelsChecked,capturedSourceCount:Object.keys(report.sourceHashes).length,recordedServedSourceCount:Object.keys(report.servedHashes).length,stockFailures:failures,correctedRGBAExactlyEqualsDirect:true,correctedRGBExactlyEqualsStock:true,rows};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(process.argv.slice(2).some(a=>a!=='--self-test'))throw Error('Only --self-test is supported. No GPU is used.');
  const evidence=path.join(root,'experiments/oit-alpha-correctness/evidence'),manifestRaw=await readFile(path.join(evidence,'manifest.json'));
  assert.equal(sha(manifestRaw),(await readFile(path.join(evidence,'manifest.sha256'),'utf8')).trim());
  const manifest=JSON.parse(manifestRaw);assert.equal(manifest.kind,'oit-alpha-public-evidence-v1');assert.equal(path.basename(manifest.file),manifest.file);assert.deepEqual(manifest.redactions,[]);
  const compressed=await readFile(path.join(evidence,manifest.file));assert.equal(compressed.length,manifest.gzipBytes);assert.equal(sha(compressed),manifest.gzipSHA256);
  const raw=gunzipSync(compressed);assert.equal(raw.length,manifest.rawBytes);assert.equal(sha(raw),manifest.reportSHA256);
  const report=JSON.parse(raw),analysis=auditReport(report);
  if(process.argv.includes('--self-test')){
    const corruptions=[r=>{r.probe.cells[0].pixelsBase64=Buffer.alloc(4096).toString('base64');},r=>{r.sourceSnapshots[r.correction.source]+=' ';},r=>{r.correction.after=r.correction.before;},r=>{r.probe.lifecycle.disposed=false;}];
    for(const corrupt of corruptions){const clone=structuredClone(report);corrupt(clone);assert.throws(()=>auditReport(clone));}
    analysis.corruptionTests=corruptions.length;
  }
  console.log(JSON.stringify({reportSHA256:manifest.reportSHA256,...analysis},null,2));
}

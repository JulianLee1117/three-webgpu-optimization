import {createServer} from 'node:http';
import {readFile,realpath,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright-core';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),serve=process.argv.includes('--serve');
if(process.argv.slice(2).some(v=>v!=='--serve'))throw Error('Only option: --serve');
const prefix='/experiments/light-vault/',allowed=[prefix,'/node_modules/three-r186/','/experiments/caustic-sketch/'];
const servedHashes={},routes=[],sha=v=>createHash('sha256').update(v).digest('hex');
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.css':'text/css','.png':'image/png','.json':'application/json','.bin':'application/octet-stream'};
const server=createServer(async(req,res)=>{try{
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405).end();return;}
  let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);if(name==='/')name=prefix+'index.html';if(name==='/favicon.ico'){res.writeHead(204).end();return;}
  const match=allowed.find(a=>name.startsWith(a));if(!match||name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid route');
  const base=await realpath(path.join(root,match)),file=await realpath(path.join(root,name)),relative=path.relative(base,file);if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Outside allowlist');
  const raw=await readFile(file),digest=sha(raw);if(!serve&&servedHashes[name]&&servedHashes[name]!==digest)throw Error('Source changed during test');servedHashes[name]=digest;
  res.writeHead(200,{'Content-Type':mime[path.extname(name)]??'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(req.method==='HEAD'?undefined:raw);
}catch(error){routes.push({url:req.url,error:error.message});res.writeHead(404).end();}});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(serve?5205:0,'127.0.0.1',resolve);});const url=`http://127.0.0.1:${server.address().port}/`;
if(serve)console.log(JSON.stringify({url,state:'On-demand rendering; fixed phase plate; no optimization on entry.'}));
else{
  const began=Date.now(),out=path.join(root,'results/development/light-vault-browser',new Date().toISOString().replaceAll(':','-'));await mkdir(out,{recursive:true});
  const report={status:'failed',startedAt:new Date().toISOString(),viewport:{width:1440,height:1050},errors:[],warnings:[],routes,servedHashes,checks:[],screenshots:{}};
  // Browser requests are recorded separately. Explicit source snapshots include
  // workers and their imports even when a failed run never reaches that branch.
  const sourceFiles=[
    'scripts/run-light-vault.mjs',
    ...['index.html','style.css','main.mjs','gpu.mjs','scene.mjs','wave.mjs','targets.mjs','seal.mjs','worker.mjs','volume-worker.mjs','assets/initial-phase.bin','assets/material-atlas.png'].map(file=>'experiments/light-vault/'+file),
    'experiments/caustic-sketch/solver.mjs',
    ...['build/three.webgpu.js','build/three.core.js','build/three.tsl.js','examples/jsm/environments/RoomEnvironment.js'].map(file=>'node_modules/three-r186/'+file),
  ];
  async function sourceSnapshot(){return Object.fromEntries(await Promise.all(sourceFiles.map(async file=>[file,sha(await readFile(path.join(root,file)))])));}
  let browser,page;const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),90000);
  function check(name,passed,detail){report.checks.push({name,passed:!!passed,detail});if(!passed)throw Error(`Check failed: ${name}`);}
  async function screenshot(name){const file=path.join(out,name+'.png');await page.screenshot({path:file});report.screenshots[name]={path:name+'.png',sha256:sha(await readFile(file))};}
  async function verify(label){const v=await page.evaluate(()=>window.lightVault.verify());report[label]=v;check(label+' GPU complex field agrees with CPU',v.relativeComplexError<.0001,v);check(label+' actual texture agrees with CPU',v.relativeTextureError<.0002,v.relativeTextureError);return v;}
  async function dragTo(distance){const current=await page.evaluate(()=>window.lightVault.state.distance),b=await page.locator('#card').boundingBox();await page.mouse.move(b.x+b.width*.45,b.y+b.height*.5);await page.mouse.down();await page.mouse.move(b.x+b.width*.45+(distance-current)/.00007,b.y+b.height*.5,{steps:15});await page.mouse.up();await page.waitForTimeout(120);}
  async function settle(){await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));}
  async function ready(label){
    await page.waitForFunction(()=>window.lightVault?.state.ready||window.lightVault?.evidence.errors.length,null,{timeout:35000});
    check(label+' experiment ready',await page.evaluate(()=>window.lightVault.state.ready),await page.evaluate(()=>window.lightVault.evidence.errors));
    await settle();
  }
  async function volumeReady(label){
    await page.waitForFunction(()=>{const app=window.lightVault,v=app.evidence.volume;return v?.status==='ready'&&v.sourceRevision===app.evidence.sourceUpdates;},null,{timeout:6500});
    const volume=await page.evaluate(()=>({...window.lightVault.evidence.volume,sourceUpdates:window.lightVault.evidence.sourceUpdates}));
    report[label]=volume;
    check(label+' is current and bounded',volume.points>0&&volume.points<=32768&&volume.slices===32&&volume.sourceRevision===volume.sourceUpdates,volume);
    await settle();return volume;
  }
  async function measureInteraction(){
    await dragTo(.010);
    const before=await page.evaluate(()=>({phase:window.lightVault.phase(),updates:window.lightVault.evidence.sourceUpdates}));
    const box=await page.locator('#card').boundingBox(),startX=box.x+box.width*.2,y=box.y+box.height*.5,travel=(.032-.010)/.00007;
    check('Measurement pointer sweep stays in the viewport',startX>=0&&startX+travel<report.viewport.width&&y>=0&&y<report.viewport.height,{startX,y,travel});
    await page.mouse.move(startX,y);await page.mouse.down();await page.evaluate(()=>window.lightVault.beginInteraction());
    const started=performance.now();
    // Forty real Playwright pointer moves paced over two seconds. Readbacks,
    // screenshots and verification are outside this measurement interval.
    for(let step=1;step<=40;step++){
      const wait=started+step*50-performance.now();if(wait>0)await new Promise(resolve=>setTimeout(resolve,wait));
      await page.mouse.move(startX+travel*step/40,y);
    }
    await page.mouse.up();await settle();
    const measured=await page.evaluate(()=>window.lightVault.endInteraction());
    report.interaction={...measured,protocol:{scheduledDurationMs:2000,steps:40,fromMetres:.010,toMetres:.032,meaning:'Input to matching render GPU submission completion; neither presentation latency nor a frame-rate claim.'}};
    check('Pointer interaction records finite latency samples',measured.inputs===40&&measured.samples.length>0&&measured.samples.length<=40&&measured.lastSubmitted===40&&Number.isFinite(measured.p50)&&Number.isFinite(measured.p95)&&measured.p50>=0&&measured.p95>=measured.p50&&measured.samples.every(s=>Number.isFinite(s.inputToGpuCompletionMs)&&s.inputToGpuCompletionMs>=0),{inputs:measured.inputs,samples:measured.samples.length,p50:measured.p50,p95:measured.p95,elapsedMs:measured.elapsedMs,lastSubmitted:measured.lastSubmitted});
    const after=await page.evaluate(()=>({phase:window.lightVault.phase(),updates:window.lightVault.evidence.sourceUpdates,distance:window.lightVault.state.distance}));
    check('Pointer measurement preserves the source exactly',sha(JSON.stringify(before.phase))===sha(JSON.stringify(after.phase))&&before.updates===after.updates,{beforeUpdates:before.updates,afterUpdates:after.updates});
    check('Pointer measurement reaches the requested final depth',Math.abs(after.distance-.032)<.00015,after.distance);
    await verify('afterInteraction');
  }
  try{
    report.sourceHashesBefore=await sourceSnapshot();
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage({viewport:report.viewport});page.setDefaultTimeout(18000);
    page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());if(m.type()==='warning')report.warnings.push(m.text());});
    await page.goto(url);await ready('Initial');await volumeReady('initialVolume');await screenshot('start');
    const phase=await page.evaluate(()=>window.lightVault.phase());report.phaseHash=sha(JSON.stringify(phase));
    await verify('middle');await page.click('#capture');await page.waitForFunction(()=>!window.lightVault.state.capturing);check('Defocused capture cannot unlock a memory',await page.evaluate(()=>window.lightVault.state.found.length===0));

    await dragTo(.012);
    report.captureMove=await page.evaluate(async()=>{
      const app=window.lightVault,errors=app.evidence.errors.length,at=app.state.distance,captures=app.evidence.captures.length;
      const pending=app.capture();await Promise.resolve();const pendingWhenMoved=app.state.capturing;
      app.setDistance(.030);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));await pending;
      return {pendingWhenMoved,at,distance:app.state.distance,errorDelta:app.evidence.errors.length-errors,captures:app.evidence.captures.slice(captures)};
    });
    check('Distance can change while capture is pending',report.captureMove.pendingWhenMoved&&report.captureMove.errorDelta===0&&report.captureMove.distance===.030&&report.captureMove.captures.length===1&&Math.abs(report.captureMove.captures[0].distance-report.captureMove.at)<1e-12,report.captureMove);
    const moved=await verify('afterCaptureMove');check('Queued depth reaches the actual GPU field',moved.match.clear&&moved.match.best===1,moved.match);
    await page.click('#reset');await dragTo(.012);
    report.captureReset=await page.evaluate(async()=>{
      const app=window.lightVault,errors=app.evidence.errors.length,captures=app.evidence.captures.length;
      const pending=app.capture();await Promise.resolve();const pendingWhenReset=app.state.capturing;
      app.reset();await pending;await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      return {pendingWhenReset,found:[...app.state.found],won:app.state.won,capturing:app.state.capturing,errorDelta:app.evidence.errors.length-errors,newCaptures:app.evidence.captures.length-captures};
    });
    check('Reset invalidates a pending capture without restoring a memory',report.captureReset.pendingWhenReset&&!report.captureReset.won&&!report.captureReset.capturing&&report.captureReset.found.length===0&&report.captureReset.newCaptures===0&&report.captureReset.errorDelta===0,report.captureReset);

    await dragTo(.012);const near=await verify('near');check('Near image is the heart',near.match.clear&&near.match.best===0,near.match);await screenshot('near-heart');
    await page.click('#capture');await page.waitForFunction(()=>window.lightVault.state.found.length===1);check('Actual clear capture discovers one memory',true);await screenshot('first-capture');
    await dragTo(.030);const far=await verify('far');check('Far image is the other memory',far.match.clear&&far.match.best===1,far.match);await screenshot('far-a');
    check('Phase plate is byte-for-byte unchanged while moving',sha(JSON.stringify(await page.evaluate(()=>window.lightVault.phase())))===report.phaseHash);
    await page.click('#capture');await page.waitForFunction(()=>window.lightVault.state.won&&window.lightVault.state.opening>.99);check('Two optical captures open the vault',await page.evaluate(()=>window.lightVault.state.found.length===2));await screenshot('open');
    await page.click('#keep-exploring');await volumeReady('settledInitialVolume');const before=await page.evaluate(()=>window.lightVault.evidence.frames);await page.waitForTimeout(350);check('Idle vault does not keep rendering',before===await page.evaluate(()=>window.lightVault.evidence.frames));
    await page.click('#reset');check('Reset clears the discovered memories',await page.evaluate(()=>!window.lightVault.state.won&&window.lightVault.state.found.length===0));
    await measureInteraction();
    await page.click('#create');await page.fill('#text-0','HI');await page.click('[data-text="0"]');await page.fill('#text-1','BY');await page.click('[data-text="1"]');await page.click('#fit');
    await page.waitForFunction(()=>!document.getElementById('maker').open||(window.lightVault.evidence.fits.length===1&&!window.lightVault.state.fitting&&!window.lightVault.state.changing),null,{timeout:18000});report.custom=await page.evaluate(()=>({fits:window.lightVault.evidence.fits,feedback:document.getElementById('fit-status').textContent,modal:document.getElementById('maker').open}));
    check('Two newly typed memories produce an accepted phase plate',!report.custom.modal&&report.custom.fits.length===1,report.custom);
    await volumeReady('customVolume');
    check('A new seal changes the phase pattern',sha(JSON.stringify(await page.evaluate(()=>window.lightVault.phase())))!==report.phaseHash);
    await dragTo(.012);await verify('customNear');await screenshot('custom-hi');
    await dragTo(.030);await verify('customFar');await screenshot('custom-by');
    await page.click('#create');const downloadPromise=page.waitForEvent('download');await page.click('#export');const download=await downloadPromise;
    check('Portable seal uses the expected download name',download.suggestedFilename()==='light-vault-seal.json',download.suggestedFilename());
    const filename=path.join(out,'light-vault-seal.json');await download.saveAs(filename);const savedText=await readFile(filename,'utf8'),saved=JSON.parse(savedText);
    check('Export contains phase data, optical units and capture references',saved.format==='light-vault-seal-v1'&&saved.n===256&&saved.phaseRadians.length===65536&&saved.pitch===1e-5&&saved.wavelength===532e-9&&saved.references.length===2&&saved.references.every(r=>r.length===65536),{format:saved.format,n:saved.n,pitch:saved.pitch,wavelength:saved.wavelength,referenceLengths:saved.references.map(r=>r.length)});
    const savedPhaseHash=sha(JSON.stringify(Array.from(Float32Array.from(saved.phaseRadians))));
    report.export={file:'light-vault-seal.json',sha256:sha(savedText),bytes:Buffer.byteLength(savedText),savedPhaseHash};
    await page.reload();await ready('Reloaded');await volumeReady('reloadedVolume');
    check('Reload restores the original phase before file import',sha(JSON.stringify(await page.evaluate(()=>window.lightVault.phase())))===report.phaseHash);
    await page.click('#create');await page.locator('#import-file').setInputFiles(filename);
    await page.waitForFunction(()=>document.getElementById('feedback').textContent.startsWith('Seal loaded.')&&!window.lightVault.state.changing,null,{timeout:18000});
    check('File input restores the exported Float32 phase exactly',sha(JSON.stringify(await page.evaluate(()=>window.lightVault.phase())))===savedPhaseHash);
    await volumeReady('importedVolume');
    await dragTo(.012);const importedNear=await verify('importedNear');check('Imported near image remains capturable',importedNear.match.clear&&importedNear.match.best===0,importedNear.match);
    await page.click('#capture');await page.waitForFunction(()=>window.lightVault.state.found.length===1&&!window.lightVault.state.capturing);
    await dragTo(.030);const importedFar=await verify('importedFar');check('Imported far image remains capturable',importedFar.match.clear&&importedFar.match.best===1,importedFar.match);
    await page.click('#capture');await page.waitForFunction(()=>window.lightVault.state.won&&window.lightVault.state.opening>.99);
    check('Two actual captures unlock the imported seal',await page.evaluate(()=>window.lightVault.state.found.length===2));await screenshot('imported-seal-open');
    check('Imported phase stays fixed while discovering it',sha(JSON.stringify(await page.evaluate(()=>window.lightVault.phase())))===savedPhaseHash);
    await page.click('#keep-exploring');await page.click('#reset');

    await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));await page.waitForTimeout(250);await screenshot('mobile');
    check('Mobile does not overflow horizontally',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    const captureBox=await page.locator('#capture').boundingBox();
    check('Mobile capture button is visible above the fold',captureBox&&captureBox.x>=0&&captureBox.y>=0&&captureBox.x+captureBox.width<=390&&captureBox.y+captureBox.height<=844,captureBox);
    const sliderBox=await page.locator('#depth').boundingBox();
    check('Mobile distance slider is visible above the fold',sliderBox&&sliderBox.x>=0&&sliderBox.y>=0&&sliderBox.x+sliderBox.width<=390&&sliderBox.y+sliderBox.height<=844,sliderBox);
    const mobileBefore=await page.evaluate(()=>window.lightVault.state.distance);
    await page.locator('#depth').click({position:{x:sliderBox.width*.8,y:sliderBox.height*.5}});await settle();
    const mobileAfter=await page.evaluate(()=>window.lightVault.state.distance);
    check('Mobile slider accepts a real pointer interaction',Math.abs(mobileAfter-mobileBefore)>.002,{before:mobileBefore,after:mobileAfter});
    await verify('mobilePointer');
    report.final=await page.evaluate(()=>window.lightVault.summary());
    report.pendingDisposal=await page.evaluate(async()=>{
      const app=window.lightVault,errors=app.evidence.errors.length,found=[...app.state.found];
      const pending=app.capture();await Promise.resolve();const pendingWhenDisposed=app.state.capturing;
      const disposal=app.dispose();await Promise.all([pending,disposal]);await new Promise(resolve=>setTimeout(resolve,50));
      return {pendingWhenDisposed,disposed:app.state.disposed,capturing:app.state.capturing,errorDelta:app.evidence.errors.length-errors,foundBefore:found,foundAfter:[...app.state.found]};
    });
    check('Disposal waits for pending capture and suppresses stale completion',report.pendingDisposal.pendingWhenDisposed&&report.pendingDisposal.disposed&&!report.pendingDisposal.capturing&&report.pendingDisposal.errorDelta===0&&JSON.stringify(report.pendingDisposal.foundBefore)===JSON.stringify(report.pendingDisposal.foundAfter),report.pendingDisposal);
    check('Fit and volume worker modules and dependencies were actually served',['worker.mjs','volume-worker.mjs','wave.mjs','targets.mjs','seal.mjs'].every(file=>Object.hasOwn(servedHashes,prefix+file))&&Object.hasOwn(servedHashes,'/experiments/caustic-sketch/solver.mjs'));
    check('No browser errors',report.errors.length===0,report.errors);check('No rejected routes',routes.length===0,routes);
    report.sourceHashesAfter=await sourceSnapshot();check('All recorded sources remain unchanged during the run',JSON.stringify(report.sourceHashesBefore)===JSON.stringify(report.sourceHashesAfter));report.status='passed';
  }catch(error){report.failure=error.stack;if(page)await screenshot('failure').catch(()=>{});}
  finally{clearTimeout(watchdog);await browser?.close();server.close();report.elapsedMs=Date.now()-began;await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({status:report.status,report:path.join(out,'report.json'),failure:report.failure,errors:report.errors,checks:report.checks},null,2));if(report.status!=='passed')process.exitCode=1;}
}

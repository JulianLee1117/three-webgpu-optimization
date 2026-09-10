import { createServer } from 'node:http';
import { readFile, realpath, mkdir, writeFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright-core';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const serve=process.argv.includes('--serve');
if(process.argv.slice(2).some(v=>v!=='--serve'))throw Error('Only option: --serve');
const prefix='/experiments/caustic-sketch/';
const allowed=[prefix,'/node_modules/three-r186/'];
const servedHashes={},routes=[];
const sha=v=>createHash('sha256').update(v).digest('hex');
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.css':'text/css','.png':'image/png','.json':'application/json'};
const server=createServer(async(req,res)=>{
  try{
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405).end();return;}
    let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);
    if(name==='/')name=prefix+'index.html';
    if(name==='/favicon.ico'){res.writeHead(204).end();return;}
    const match=allowed.find(a=>name.startsWith(a));
    if(!match||name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid route');
    const base=await realpath(path.join(root,match)),file=await realpath(path.join(root,name)),relative=path.relative(base,file);
    if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Outside allowlist');
    const raw=await readFile(file),digest=sha(raw);
    if(!serve&&servedHashes[name]&&servedHashes[name]!==digest)throw Error('Source changed during test');
    servedHashes[name]=digest;res.writeHead(200,{'Content-Type':mime[path.extname(name)]??'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(req.method==='HEAD'?undefined:raw);
  }catch(error){routes.push({url:req.url,error:error.message});res.writeHead(404).end();}
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(serve?5204:0,'127.0.0.1',resolve);});
const url=`http://127.0.0.1:${server.address().port}/`;
if(serve){console.log(JSON.stringify({url,state:'On-demand rendering; one bounded CPU fit on entry.'}));}
else{
  const began=Date.now(),out=path.join(root,'results/development/sunprint',new Date().toISOString().replaceAll(':','-'));
  await mkdir(out,{recursive:true});
  const report={status:'failed',startedAt:new Date().toISOString(),viewport:{width:1440,height:1050},errors:[],warnings:[],routes,servedHashes,checks:[],screenshots:{}};
  let browser,page;
  const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),90000);
  function check(name,passed,detail){report.checks.push({name,passed:!!passed,detail});if(!passed)throw Error(`Check failed: ${name}`);}
  async function screenshot(name){const file=path.join(out,name+'.png');await page.screenshot({path:file});report.screenshots[name]={path:name+'.png',sha256:sha(await readFile(file))};}
  async function idle(){await page.waitForFunction(()=>window.sunprint?.evidence.ready&&!window.sunprint.state.solving,null,{timeout:20000});await page.waitForTimeout(200);}
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();
    page=await browser.newPage({viewport:report.viewport});page.setDefaultTimeout(15000);
    page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());if(m.type()==='warning')report.warnings.push(m.text());});
    await page.goto(url);await page.waitForFunction(()=>window.sunprint?.evidence.ready||window.sunprint?.evidence.errors.length,null,{timeout:35000});
    report.initial=await page.evaluate(()=>window.sunprint.summary());check('Initial fit ready',report.initial.evidence.ready,report.initial);
    await idle();await screenshot('light-heart');
    report.gpu=await page.evaluate(()=>window.sunprint.verify());check('Actual GPU Snell hits match independent vector trace',report.gpu.maxError<.0001&&report.gpu.lost===0,report.gpu.maxError);
    report.raster=await page.evaluate(()=>window.sunprint.rasterEvidence());
    check('Actual raster preserves finite transmitted flux',report.raster.finite&&report.raster.relativeFluxError<.05,{flux:report.raster.flux,cpuFlux:report.raster.cpuFlux,relativeError:report.raster.relativeFluxError});
    check('Actual raster follows independent ray distribution and orientation',report.raster.cosine>.97&&report.raster.cosine>report.raster.flippedCosine+.02,{cosine:report.raster.cosine,flippedCosine:report.raster.flippedCosine});
    const height=await page.evaluate(()=>window.sunprint.surface());
    await page.locator('#distance').fill('6');await page.locator('#distance').dispatchEvent('input');await page.waitForTimeout(200);await screenshot('defocus');
    const near=await page.evaluate(()=>window.sunprint.verify());
    check('Screen moves without changing glass',JSON.stringify(height)===JSON.stringify(await page.evaluate(()=>window.sunprint.surface())));
    check('Screen distance changes optical hits',near.gpuHits.some((v,i)=>Math.abs(v-report.gpu.gpuHits[i])>.05));
    await page.click('#refocus');await page.click('#flatten');await page.waitForTimeout(200);await screenshot('flat');
    report.flat=await page.evaluate(()=>window.sunprint.verify());check('Flat lens verified on GPU',report.flat.maxError<.0001);
    check('Flat glass removes fitted geometry',(await page.evaluate(()=>window.sunprint.surface())).every(v=>v===0));await page.click('#flatten');
    check('Restore returns exact lens',JSON.stringify(height)===JSON.stringify(await page.evaluate(()=>window.sunprint.surface())));
    await page.click('[data-view=glass]');await page.waitForTimeout(300);await screenshot('glass');
    await page.click('#rays');await page.waitForTimeout(300);await screenshot('studio-rays');
    await page.click('#rays');await page.click('[data-view=light]');
    await page.click('#clear');const box=await page.locator('#drawing').boundingBox();
    // Real pointer drawing: a broad, closed diagonal diamond not in the preset library.
    const points=[[.5,.2],[.8,.5],[.5,.8],[.2,.5],[.5,.2]];
    await page.mouse.move(box.x+box.width*points[0][0],box.y+box.height*points[0][1]);await page.mouse.down();
    for(const [x,y]of points.slice(1))await page.mouse.move(box.x+box.width*x,box.y+box.height*y,{steps:12});await page.mouse.up();
    await page.waitForTimeout(700);await idle();await screenshot('freehand-diamond');
    report.freehand=await page.evaluate(()=>window.sunprint.summary());check('Real freehand triggers a new fit',report.freehand.evidence.solves.length>=2);
    check('Freehand inverse loss improves',report.freehand.result.finalLoss<report.freehand.result.initialLoss*.6,report.freehand.result);
    report.freehandGpu=await page.evaluate(()=>window.sunprint.verify());check('Freehand GPU matches independent trace',report.freehandGpu.maxError<.0001,report.freehandGpu.maxError);
    await page.fill('#word','HI');await page.click('#type');await idle();await screenshot('letters');
    report.letters=await page.evaluate(()=>window.sunprint.summary());check('Letter fit improves',report.letters.result.finalLoss<report.letters.result.initialLoss*.6,report.letters.result);
    const downloadPromise=page.waitForEvent('download');await page.click('#export');const download=await downloadPromise;const exportPath=path.join(out,download.suggestedFilename());await download.saveAs(exportPath);const stl=await readFile(exportPath);
    check('Export is a complete binary STL',stl.length===84+stl.readUInt32LE(80)*50,{bytes:stl.length,triangles:stl.readUInt32LE(80)});
    const before=await page.evaluate(()=>window.sunprint.evidence.frames);await page.waitForTimeout(500);const after=await page.evaluate(()=>window.sunprint.evidence.frames);check('Stationary view does not keep rendering',after===before,{before,after});
    await page.click('[data-preset=heart]');await idle();
    await page.setViewportSize({width:390,height:844});await page.waitForTimeout(250);await screenshot('mobile');check('Mobile has no horizontal overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    check('No browser errors',report.errors.length===0,report.errors);check('No rejected routes',routes.length===0,routes);
    await page.evaluate(()=>window.sunprint.dispose());check('Disposal completed',await page.evaluate(()=>window.sunprint.state.disposed));report.status='passed';
  }catch(error){report.failure=error.stack;if(page)await screenshot('failure').catch(()=>{});}
  finally{clearTimeout(watchdog);await browser?.close();server.close();report.elapsedMs=Date.now()-began;await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({status:report.status,report:path.join(out,'report.json'),failure:report.failure,errors:report.errors,checks:report.checks},null,2));if(report.status!=='passed')process.exitCode=1;}
}

import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),serve=args.includes('--serve'),ui=args.includes('--ui'),fit=args.includes('--fit'),kind=args.find(a=>a.startsWith('--kind='))?.slice(7)??'tentacle',variant=args.find(a=>a.startsWith('--variant='))?.slice(10)??'original';
if(!['ribbon','tentacle'].includes(kind)||!['original','late-target'].includes(variant)||args.some(a=>!['--serve','--ui','--fit',`--kind=${kind}`,`--variant=${variant}`].includes(a)))throw Error('Unknown bounded runner option');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),served={};
const prefixes=['/experiments/constraint-editing/','/experiments/material-mips/','/node_modules/three/src/'];
const inside=(base,file)=>{const r=path.relative(base,file);return r!==''&&!r.startsWith('..')&&!path.isAbsolute(r);};
const server=createServer(async(req,res)=>{try{
  let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(name==='/')name='/experiments/constraint-editing/index.html';
  if(name==='/favicon.ico'){res.writeHead(204).end();return;}
  if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
  const prefix=prefixes.find(p=>name.startsWith(p));if(!prefix)throw Error('Invalid route');
  const [base,file]=await Promise.all([realpath(path.join(root,prefix)),realpath(path.join(root,name))]);if(!inside(base,file))throw Error('Route escape');
  const bytes=await readFile(file),digest=hash(bytes);if(!serve&&served[name]&&served[name]!==digest)throw Error('Source changed');served[name]=digest;
  res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.json')?'application/json':name.endsWith('.md')?'text/plain':'text/javascript','Cache-Control':'no-store'});res.end(bytes);
}catch(error){res.writeHead(404).end(String(error.message));}});
await new Promise(resolve=>server.listen(serve?5195:0,'127.0.0.1',resolve));const url=`http://127.0.0.1:${server.address().port}/`;
if(serve)console.log(JSON.stringify({url,mode:'idle-until-click'}));
else{
  const out=path.join(root,'results/development/constraint-editing-gpu',new Date().toISOString().replaceAll(':','-')+`-${kind}-${variant}-${ui?'ui':fit?'fit':'parity'}`);await mkdir(out,{recursive:true});
  const report={kind:'constraint-editing-gpu-v2',field:kind,variant,fit,ui,status:'failed',errors:[],sourceHashes:{},sourceSnapshots:{},servedHashes:served};
  for(const name of ['scripts/run-constraint-editing.mjs','experiments/material-mips/autograd.js',...(await readdir(path.join(root,'experiments/constraint-editing'))).filter(n=>/\.(js|mjs|html|md)$/.test(n)).map(n=>'experiments/constraint-editing/'+n)]){
    const bytes=await readFile(path.join(root,name));report.sourceHashes[name]=hash(bytes);if(/\.(js|mjs|html)$/.test(name))report.sourceSnapshots[name]=bytes.toString('utf8');
  }
  const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}};
  let browser,page;report.preGpu=telemetry();const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage({viewport:{width:1360,height:1040}});page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    await page.goto(url);
    if(ui){
      await page.waitForFunction(()=>window.constraintLab);
      report.initialState=await page.evaluate(()=>window.constraintLab.status);
      if(report.initialState.allocated)throw Error('GPU allocated on initial load');
      await page.selectOption('#program',kind);await page.selectOption('#phase',variant);await page.click('#start');await page.waitForFunction(()=>window.constraintLab.status.allocated&&!window.constraintLab.status.busy,null,{timeout:20000});
      await page.click('#fit');await page.waitForFunction(()=>!window.constraintLab.status.busy,null,{timeout:20000});
      await page.evaluate(()=>window.constraintLab.pause());
      report.uiResult=await page.evaluate(()=>({result:window.constraintLab.result,status:window.constraintLab.status,message:document.querySelector('#status').textContent}));
      if(!report.uiResult.result)throw Error('UI fit did not produce a result');
      await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
      await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
      report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(report.mobileOverflow)throw Error('Mobile overflow');
      await page.evaluate(()=>window.constraintLab.close());report.finalState=await page.evaluate(()=>window.constraintLab.status);if(report.finalState.allocated)throw Error('Close did not release GPU');
    }else{
      report.probe=await page.evaluate(async({kind,fit,variant})=>{const{runProbe}=await import('/experiments/constraint-editing/probe.js');return runProbe(kind,{fit,variant});},{kind,fit,variant});
      if(report.probe.status!=='passed')throw Error(report.probe.failure??'Probe failed');
    }
    if(report.errors.length)throw Error('Browser errors');
    for(const [name,digest]of Object.entries(served))if(hash(await readFile(path.join(root,name)))!==digest)throw Error('Served source changed during run');report.status='passed';
  }catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.postGpu=telemetry();
    const bytes=JSON.stringify(report,(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),hash(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,failure:report.failure,validation:report.probe?.validations.map(v=>({passed:v.passed,maxPrimal:v.maxPrimal,maxAD:v.maxAD,maxFD:v.maxFD,failures:v.failures.slice(0,3)})),fits:report.probe?.fits.map(f=>({method:f.method,validation:f.validation})),errors:report.errors,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
  }
}

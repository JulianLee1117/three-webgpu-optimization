import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),serve=process.argv.includes('--serve');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),served={};
const inside=(base,file)=>{const r=path.relative(base,file);return r!==''&&!r.startsWith('..')&&!path.isAbsolute(r);};
const server=createServer(async(req,res)=>{try{
  let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(name==='/')name='/experiments/material-mips/index.html';
  if(name==='/favicon.ico'){res.writeHead(204).end();return;}
  if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
  const prefix=['/experiments/material-mips/','/node_modules/three/src/'].find(p=>name.startsWith(p));if(!prefix)throw Error('Invalid route');
  const [base,file]=await Promise.all([realpath(path.join(root,prefix)),realpath(path.join(root,name))]);if(!inside(base,file))throw Error('Route escape');
  const bytes=await readFile(file),digest=hash(bytes);if(!serve&&served[name]&&served[name]!==digest)throw Error('Source changed');served[name]=digest;
  res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.json')?'application/json':name.endsWith('.md')?'text/plain':'text/javascript','Cache-Control':'no-store'});res.end(bytes);
}catch{res.writeHead(404).end();}});
await new Promise(resolve=>server.listen(serve?5194:0,'127.0.0.1',resolve));const url=`http://127.0.0.1:${server.address().port}/`;
if(serve)console.log(JSON.stringify({url,mode:'idle'}));
else{
  const out=path.join(root,'results/development/material-mips',new Date().toISOString().replaceAll(':','-')+'-parity');await mkdir(out,{recursive:true});
  const report={kind:'material-mips-gpu-gate-v1',status:'failed',errors:[],sourceHashes:{},servedHashes:served};
  for(const name of ['scripts/run-material-mips.mjs',...(await readdir(path.join(root,'experiments/material-mips'))).filter(n=>/\.(js|mjs|html|md)$/.test(n)).map(n=>'experiments/material-mips/'+n)])report.sourceHashes[name]=hash(await readFile(path.join(root,name)));
  const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}};
  let browser,page;report.preGpu=telemetry();const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    await page.goto(url);await page.waitForFunction(()=>window.materialMips,null,{timeout:20000});
    report.probe=await page.evaluate(()=>window.materialMips.runParity());
    if(report.probe.status!=='passed'||report.errors.length)throw Error('Parity probe failed; inspect retained evidence');
    for(const [name,digest]of Object.entries(served))if(hash(await readFile(path.join(root,name)))!==digest)throw Error('Served source changed during run');report.status='passed';
  }catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.postGpu=telemetry();
    const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),hash(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,failure:report.probe?.failure??report.failure,mismatchCount:report.probe?.mismatchCount,lights:report.probe?.lights.map(l=>({maxPixelError:l.maxPixelError,maxCPUError:l.maxCPUError,maxGradientError:l.maxGradientError,mismatches:l.mismatches.slice(0,3)})),errors:report.errors,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
  }
}

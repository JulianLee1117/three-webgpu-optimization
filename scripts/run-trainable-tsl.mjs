import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {chromium} from 'playwright-core';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),serve=args.includes('--serve'),kind=args.find(a=>a.startsWith('--field='))?.slice(8)??'neural',seed=Number(args.find(a=>a.startsWith('--seed='))?.slice(7)??11),ui=args.includes('--ui');
if(args.some(a=>!['--serve','--ui',`--field=${kind}`,`--seed=${seed}`].includes(a))||!['neural','islands'].includes(kind)||![11,29].includes(seed))throw Error('Unknown bounded runner option');
const hash=data=>createHash('sha256').update(data).digest('hex'),served={};
const inside=(base,target)=>{const rel=path.relative(base,target);return rel!==''&&!rel.startsWith('..')&&!path.isAbsolute(rel);};
const server=createServer(async(req,res)=>{try{
  let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(name==='/')name='/experiments/trainable-tsl/index.html';
  if(name==='/favicon.ico'){res.writeHead(204).end();return;}
  if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
  const prefix=['/experiments/trainable-tsl/','/node_modules/three/build/'].find(p=>name.startsWith(p));if(!prefix)throw Error('Unknown route');
  const [base,file]=await Promise.all([realpath(path.join(root,prefix)),realpath(path.join(root,name))]);if(!inside(base,file))throw Error('Route escape');
  const bytes=await readFile(file),digest=hash(bytes);if(!serve&&served[name]&&served[name]!==digest)throw Error('Source changed during run');served[name]=digest;
  res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.md')?'text/plain':name.endsWith('.json')?'application/json':'text/javascript','Cache-Control':'no-store'});res.end(bytes);
}catch{res.writeHead(404).end();}});
await new Promise(resolve=>server.listen(serve?5192:0,'127.0.0.1',resolve));const url=`http://127.0.0.1:${server.address().port}/`;
if(serve){console.log(JSON.stringify({url,mode:'idle-until-click'}));}
else{
  const out=path.join(root,'results/development/trainable-tsl',new Date().toISOString().replaceAll(':','-')+`-${kind}-${seed}${ui?'-ui':''}`);await mkdir(out,{recursive:true});
  const report={kind:'native-trainable-tsl-v1',field:kind,seed,status:'failed',errors:[],sourceHashes:{},servedHashes:served};
  for(const file of ['scripts/run-trainable-tsl.mjs',...(await readdir(path.join(root,'experiments/trainable-tsl'))).filter(f=>/\.(js|html|md)$/.test(f)).map(f=>'experiments/trainable-tsl/'+f)])report.sourceHashes[file]=hash(await readFile(path.join(root,file)));
  const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}};
  let browser,page;report.preGpu=telemetry();const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage({viewport:{width:1360,height:1000}});
    page.on('pageerror',error=>report.errors.push(error.message));page.on('console',message=>{if(message.type()==='error')report.errors.push(message.text());});
    await page.goto(url);await page.waitForFunction(()=>window.trainableLab);
    report.initialState=await page.evaluate(()=>window.trainableLab.status);if(report.initialState.allocated||report.initialState.running)throw Error('Not idle on load');
    if(ui){
      await page.selectOption('#field',kind);await page.click('#train');await page.waitForFunction(()=>!window.trainableLab.status.running,null,{timeout:40000});
      report.uiResult=await page.evaluate(()=>({result:window.trainableLab.result,status:window.trainableLab.status,message:document.querySelector('#status').textContent}));
      if(!report.uiResult.result||report.uiResult.result.completed!==160)throw Error('UI did not finish: '+report.uiResult.message);
      await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
      await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
      report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(report.mobileOverflow)throw Error('Mobile overflow');
      await page.click('#restore');await page.locator('#examples').click({position:{x:110,y:90}});
      report.paint=await page.evaluate(()=>window.trainableLab.status);if(!report.paint.custom)throw Error('Paint did not edit examples');
      report.stop=await page.evaluate(async()=>{const p=window.trainableLab.train();window.trainableLab.stop();await p;return window.trainableLab.status;});if(report.stop.running||report.stop.allocated)throw Error('Early stop did not release GPU');
    }else{
      report.probe=await page.evaluate(async({kind,seed})=>{
        const {createTrainer}=await import('/experiments/trainable-tsl/engine.js');const trainer=await createTrainer({kind,seed});
        const result={adapter:trainer.adapter,stats:trainer.stats,results:[]};
        try{result.validation=await trainer.validate();if(!result.validation.passed)throw Error('Gradient correctness failed');
          for(const method of seed===11?['ad','fd']:['fd','ad']){
            const fit=await trainer.train({method});result.results.push(fit);
            fit.finalValidation=await trainer.validate({resetFirst:false,checkUpdate:false});
            if(!fit.finalValidation.passed)throw Error('Learned-weight gradient correctness failed');
            const reloaded=await trainer.loadWeights(JSON.parse(JSON.stringify(fit.weights)));
            fit.reloadExact=reloaded.weights.every((v,i)=>v===fit.weights[i])&&reloaded.gradients.every((v,i)=>v===fit.finalValidation.adGradients[i]);
            if(!fit.reloadExact)throw Error('Export/reload changed weights or generated gradients');
          }
        }catch(error){result.failure=String(error.stack??error);}
        finally{result.errors=[...trainer.errors];result.shaderSources=[...trainer.shaders];trainer.dispose();result.finalState=trainer.status;}
        return result;
      },{kind,seed});
      if(report.probe.failure)throw Error(report.probe.failure);
      if(report.probe.results.some(r=>r.completed!==160||!r.utilityPassed))throw Error('Declared utility gate failed');
    }
    if(report.errors.length)throw Error('Browser errors');
    for(const [name,digest]of Object.entries(served))if(hash(await readFile(path.join(root,name)))!==digest)throw Error('Served source changed');
    report.status='passed';
  }catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.postGpu=telemetry();
    const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),hash(bytes)+'\n',{flag:'wx'});
    const v=report.probe?.validation;
    console.log(JSON.stringify({out,status:report.status,failure:report.failure,validation:v&&{passed:v.passed,count:v.count,maxADAbsolute:v.maxADAbsolute,maxFDAbsolute:v.maxFDAbsolute,adam:v.adam?.passed},results:report.probe?.results.map(r=>({method:r.method,completed:r.completed,initial:r.initial.heldout.rmse,final:r.final.heldout.rmse,utilityPassed:r.utilityPassed,completedWallMs:r.completedWallMs})),errors:report.errors,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
  }
}

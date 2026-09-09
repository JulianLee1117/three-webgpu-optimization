import {createExperiment} from './model.js';
import {createDisplay} from './display.js';
import {solveConstrained,CONSTRAINED_LIMITS} from './solver-constrained.mjs';
import {validate} from './fixtures.mjs';
import {runProbe} from './probe.js';
const $=id=>document.getElementById(id);
let experiment=null,display=null,busy=false,cancelled=false,active=null,lastResult=null,custom=false,presetTarget=null,failureMessage=null;
function controls(){
  $('start').disabled=busy||!!experiment;$('program').disabled=busy;$('phase').disabled=busy;
  for(const id of ['fit','play','reset'])$(id).disabled=busy||!experiment;
  $('export').disabled=busy||!lastResult;$('stop').disabled=!experiment&&!busy;
}
function metrics(params){
  const m=validate(experiment.fixture,params);
  $('target-error').textContent=m.targetError.toFixed(4);
  $('clearance').textContent=(m.minClearance>=0?'+':'')+m.minClearance.toFixed(4);
  $('motion').textContent=(100*m.velocityNRMSE).toFixed(1)+'%';
  $('clearance').style.color=m.minClearance<0?'#ffb58b':'#b6f7d6';return m;
}
function release(){display?.dispose();display=null;experiment?.evaluator.dispose();experiment=null;controls();}
async function open(){
  if(busy||experiment)return;busy=true;cancelled=false;custom=false;failureMessage=null;lastResult=null;controls();$('status').textContent='Binding the native shader graph…';
  try{
    experiment=await createExperiment($('program').value,$('phase').value);
    presetTarget=[...experiment.fixture.target.position];
    if(cancelled){release();return;}
    $('empty')?.remove();
    display=createDisplay({...experiment,container:$('viewport'),onError:error=>{failureMessage=error.message;cancelled=true;$('status').textContent=failureMessage;if(!busy)release();},onTarget:position=>{
      if(busy)return;experiment.fixture.target.position=position;custom=true;lastResult=null;metrics(display.parameters);controls();$('status').textContent='Target moved. Apply the edit; some targets may be outside this shader’s controls.';
    }});
    if(cancelled)return;
    metrics(experiment.fixture.initial);$('status').textContent='Ready. Drag the target or apply the preset edit.';
  }catch(error){$('status').textContent=error.message;release();}
  finally{busy=false;if(cancelled)release();controls();}
}
async function fit(){
  if(busy||!experiment)return;busy=true;cancelled=false;failureMessage=null;lastResult=null;display.pause();controls();$('progress').value=0;
  $('status').textContent='Fitting the existing shader parameters with generated GPU derivatives…';
  try{
    const deadline=performance.now()+12000;
    const result=await solveConstrained(experiment.fixture,{evaluate:experiment.evaluator.evaluate,shouldStop:()=>cancelled||performance.now()>deadline,
      onProgress:async({outer,inner,params})=>{
        const iteration=outer*CONSTRAINED_LIMITS.inner+inner;
        $('progress').value=(iteration+1)/(CONSTRAINED_LIMITS.outer*CONSTRAINED_LIMITS.inner);
        if(iteration%4===0&&!cancelled){display.setParameters(params);await new Promise(resolve=>setTimeout(resolve,0));}
      }});
    if(cancelled)return;
    result.exportedParameters=result.params.map(Math.fround);display.setParameters(result.exportedParameters);result.validation=metrics(result.exportedParameters);result.custom=custom;lastResult=result;
    $('progress').value=1;
    $('status').textContent=result.status==='stopped'?'Twelve-second fitting limit reached. Partial parameters are available.':result.validation.pass&&result.diagnostics.maximumViolation<=.001?'Edit complete. The original shader now runs with eight updated parameters.':
      'Fit completed with unsatisfied constraints. The metrics show where this edit falls short.';
    display.play();
  }catch(error){$('status').textContent=error.message;release();}
  finally{busy=false;if(cancelled){release();$('status').textContent=failureMessage??'Stopped. GPU released.';}controls();}
}
async function close(){cancelled=true;display?.pause();await active?.catch(()=>{});release();$('status').textContent='Stopped. GPU released. Open the experiment to continue.';}
function reset(){if(busy||!experiment)return;display.pause();experiment.fixture.target.position=[...presetTarget];custom=false;lastResult=null;display.setParameters(experiment.fixture.initial);metrics(experiment.fixture.initial);$('progress').value=0;$('status').textContent='Original parameters restored.';controls();}
$('start').addEventListener('click',()=>active=open());$('fit').addEventListener('click',()=>active=fit());$('play').addEventListener('click',()=>display?.play());$('stop').addEventListener('click',close);$('reset').addEventListener('click',reset);
for(const id of ['program','phase'])$(id).addEventListener('change',async()=>{await close();lastResult=null;custom=false;$('status').textContent='Ready. Open the selected experiment.';controls();});
$('export').addEventListener('click',()=>{
  if(!lastResult||!experiment)return;
  const data={format:'native-tsl-motion-parameters-v1',three:'0.185.1',kind:experiment.fixture.kind,variant:experiment.fixture.variant,parameters:lastResult.exportedParameters,
    target:experiment.fixture.target,obstacle:experiment.fixture.obstacle,validation:lastResult.validation,custom};
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=data.kind+'-motion.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
addEventListener('resize',()=>display?.resize());document.addEventListener('visibilitychange',()=>{if(document.hidden)close();});
window.constraintLab={open:()=>active=open(),fit:()=>active=fit(),close,reset,pause:()=>display?.pause(),runProbe,
  get result(){return lastResult;},get targetScreen(){return display?.targetScreen;},get status(){return {allocated:!!experiment,busy,custom,playing:display?.playing??false};}};
controls();

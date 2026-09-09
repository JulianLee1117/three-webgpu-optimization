import { SCENE_SPECS } from './scenes.js';
import { initialVector, optimize, METHOD_LABELS } from './optimizers.mjs';
import { optimizeLeastSquares } from './least-squares.mjs';
import { createEngine, physicalValues } from './engine.js';
const state={phase:'idle',errors:[],disposed:false,view:'train'};let engine;
const LABELS={...METHOD_LABELS,'least-squares':'GPU finite differences / damped least squares'};
const selector=document.querySelector('#scene'),status=document.querySelector('#status'),metrics=document.querySelector('#metrics');
for(const spec of SCENE_SPECS){const option=document.createElement('option');option.value=spec.id;option.textContent=spec.title;selector.appendChild(option);}
selector.value='robot';
for(const option of document.querySelector('#method').options)option.textContent=LABELS[option.value];
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function initialize(scene,seed){
  if(state.phase!=='idle')throw Error('One initialization per page load');
  const spec=SCENE_SPECS.find(s=>s.id===scene);if(!spec)throw Error('Unknown scene');
  for(const id of ['run','scene','method','seed'])document.querySelector('#'+id).disabled=true;
  selector.value=scene;document.querySelector('#seed').value=String(seed);
  state.phase='initializing';status.textContent='Preparing GPU loss…';
  engine=await createEngine(spec,document.querySelector('#view'));
  state.initial=initialVector(spec.parameters.length,seed);state.scene=scene;state.seed=seed;
  const values=physicalValues(spec,state.initial);
  for(const type of ['out-of-memory','internal','validation'])engine.device.pushErrorScope(type);
  try{
    state.parity=await engine.validate(values);
    state.initialTrain=(await engine.evaluate(values)).loss;
    state.initialHeldout=(await engine.evaluate(values,true)).loss;
    await engine.evaluate(values);engine.saveCenter();
    state.equationParity=(await engine.captureEquations(state.initial,Array(8).fill(.06),{validate:true})).parity;
  }finally{for(let i=0;i<3;i++){const e=await engine.device.popErrorScope();if(e)state.errors.push(e.message);}}
  if(state.errors.length)throw Error(state.errors.join('; '));
  engine.display(values,values);state.phase='ready';state.results=[];
  return {environment:engine.environment,scene,seed,parameters:spec.parameters,initial:state.initial,initialPhysical:values,target:spec.target,initialTrain:state.initialTrain,initialHeldout:state.initialHeldout,parity:state.parity,equationParity:state.equationParity};
}
async function fit(method,budget=96,preview=false){
  if(!['ready','finished'].includes(state.phase))throw Error('Fitter is not ready');
  if(!['random','coordinate','candidate','least-squares'].includes(method)||budget!==96)throw Error('Unbounded method or budget');
  if(state.results.some(r=>r.method===method)||state.results.length>=4)throw Error('Each method runs at most once per page');
  state.phase='fitting';status.textContent=LABELS[method];let evaluation=0,best=state.initial,bestLoss=Infinity,completedMs=0,scalarQueries=0,equationBatches=0;
  const renderHistory=[];
  const start=performance.now();
  for(const type of ['out-of-memory','internal','validation'])engine.device.pushErrorScope(type);
  let output;
  try{
    const evaluate=async vector=>{
      const callbackStart=performance.now();
      if(state.errors.length||engine.errors.length)throw Error([...state.errors,...engine.errors].join('; '));
      if(evaluation>=budget)throw Error('Render budget exhausted');
      const result=await engine.evaluate(physicalValues(engine.spec,vector));evaluation++;scalarQueries++;
      renderHistory.push({evaluation,kind:'scalar',parameters:[...vector],loss:result.loss});
      if(result.loss<bestLoss){bestLoss=result.loss;best=[...vector];engine.saveCenter();}
      if(preview && evaluation%8===0)engine.display(physicalValues(engine.spec,state.initial),physicalValues(engine.spec,best));
      metrics.textContent=`${LABELS[method]} · ${evaluation}/${budget} fitting renders\nBest training image MSE: ${bestLoss.toExponential(4)}`;
      await wait(4);completedMs+=performance.now()-callbackStart;return result.loss;
    };
    const captureEquations=async(center,steps)=>{
      if(center.some((v,i)=>v!==best[i]))throw Error('Equations must use the cached best image');
      if(evaluation+16>budget)throw Error('Equation batch exceeds render budget');
      const start=performance.now();const result=await engine.captureEquations(center,steps,{onRender:async vector=>{
        evaluation++;renderHistory.push({evaluation,kind:'finite-difference',parameters:[...vector]});await wait(4);
      }});completedMs+=performance.now()-start;equationBatches++;return result;
    };
    output=method==='least-squares'?await optimizeLeastSquares({initial:state.initial,budget,evaluate,captureEquations}):await optimize({method,initial:state.initial,budget,seed:state.seed,evaluate});
    if(evaluation!==budget||output.evaluations!==budget)throw Error('Evaluation budget mismatch');
    output.finalHeldout=(await engine.evaluate(physicalValues(engine.spec,output.best),true)).loss;
  }finally{for(let i=0;i<3;i++){const e=await engine.device.popErrorScope();if(e)state.errors.push(e.message);}}
  if(state.errors.length||engine.errors.length)throw Error([...state.errors,...engine.errors].join('; '));
  const record={...output,method,renderHistory,scalarQueries,equationBatches,wallMs:performance.now()-start,serializedEvaluationMs:completedMs,timingBoundary:'Serialized callback wall time, including deliberate pauses and UI work in all methods; descriptive only.',lossReadbackBytes:scalarQueries*16+equationBatches*176,physical:physicalValues(engine.spec,output.best)};
  state.results.push(record);state.phase='finished';status.textContent='Finished · GPU idle';
  engine.display(physicalValues(engine.spec,state.initial),record.physical);
  metrics.textContent=`${LABELS[method]} · ${evaluation} fitting renders\nTraining image MSE: ${state.initialTrain.toExponential(4)} → ${record.bestLoss.toExponential(4)}\nHeld-out view MSE: ${state.initialHeldout.toExponential(4)} → ${record.finalHeldout.toExponential(4)}\nFitting loop readback: ${record.lossReadbackBytes} bytes. No color, light or camera parameters fitted.`;
  state.parameterPatch={scene:state.scene,parameters:Object.fromEntries(engine.spec.parameters.map((p,i)=>[p.name,record.physical[i]]))};
  document.querySelector('#parameters').textContent=JSON.stringify(state.parameterPatch,null,2);
  document.querySelector('#save').disabled=false;return record;
}
async function dispose(){if(engine&&!state.disposed){await engine.dispose();state.errors.push(...engine.errors);state.disposed=true;state.phase='disposed';}}
function show(method){
  const record=state.results.find(r=>r.method===method);if(!record)throw Error('No fitted result for '+method);
  engine.display(physicalValues(engine.spec,state.initial),record.physical);
  status.textContent=LABELS[method]+' · completed';
  metrics.textContent=`${LABELS[method]} · ${record.evaluations} fitting renders\nTraining image MSE: ${state.initialTrain.toExponential(4)} → ${record.bestLoss.toExponential(4)}\nHeld-out view MSE: ${state.initialHeldout.toExponential(4)} → ${record.finalHeldout.toExponential(4)}\nFitting loop readback: ${record.lossReadbackBytes} bytes. No color, light or camera parameters fitted.`;
  state.parameterPatch={scene:state.scene,parameters:Object.fromEntries(engine.spec.parameters.map((p,i)=>[p.name,record.physical[i]]))};
  document.querySelector('#parameters').textContent=JSON.stringify(state.parameterPatch,null,2);
  state.displayPhysical=[...record.physical];
  const tuning=document.querySelector('#tuning');tuning.replaceChildren();tuning.disabled=false;
  for(const [i,p]of engine.spec.parameters.entries()){
    const label=document.createElement('label'),caption=document.createElement('span'),slider=document.createElement('input');
    caption.textContent=p.name;slider.type='range';slider.min=p.min;slider.max=p.max;slider.step=(p.max-p.min)/200;slider.value=record.physical[i];slider.setAttribute('aria-label',p.name);
    slider.addEventListener('input',()=>{
      state.displayPhysical[i]=Number(slider.value);engine.display(physicalValues(engine.spec,state.initial),state.displayPhysical,state.view);
      state.parameterPatch={scene:state.scene,parameters:Object.fromEntries(engine.spec.parameters.map((p,i)=>[p.name,state.displayPhysical[i]]))};
      document.querySelector('#parameters').textContent=JSON.stringify(state.parameterPatch,null,2);
      status.textContent='Manual edit · GPU idle between changes';metrics.textContent='Exported parameters now match your manual edits. The fitting measurements apply to the original fitted result.';
    });label.append(caption,slider);tuning.appendChild(label);
  }
  document.querySelector('#view-mode').disabled=false;
}
document.querySelector('#run').addEventListener('click',async()=>{
  document.querySelector('#run').disabled=true;selector.disabled=true;document.querySelector('#method').disabled=true;
  try{await initialize(selector.value,Number(document.querySelector('#seed').value));await fit(document.querySelector('#method').value,96,true);show(document.querySelector('#method').value);}catch(error){state.errors.push(error.message);status.textContent='Failed: '+error.message;await dispose();}
});
document.querySelector('#view-mode').addEventListener('change',event=>{state.view=event.target.value;engine.display(physicalValues(engine.spec,state.initial),state.displayPhysical,state.view);});
document.querySelector('#save').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(state.parameterPatch,null,2)+'\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='scene-parameters.json';link.click();URL.revokeObjectURL(url);
});
window.sceneFit={initialize,fit,show,dispose,state};

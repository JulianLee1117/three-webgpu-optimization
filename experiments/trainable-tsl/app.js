import {createTrainer} from './engine.js';
import {makeSamples,FIELDS} from './fields.js';
const $=id=>document.getElementById(id),canvas=$('examples'),ctx=canvas.getContext('2d');
let trainer=null,samples=makeSamples('neural'),custom=false,running=false,cancelled=false,lastResult=null,active=null;
function draw(){ctx.fillStyle='#08151e';ctx.fillRect(0,0,240,240);ctx.strokeStyle='#18333f';for(let i=0;i<=8;i++){ctx.beginPath();ctx.moveTo(i*30,0);ctx.lineTo(i*30,240);ctx.moveTo(0,i*30);ctx.lineTo(240,i*30);ctx.stroke();}for(let i=0;i<samples.length;i+=4){const t=Math.max(0,Math.min(1,(samples[i+2]+.5)/1.5));ctx.fillStyle=`hsl(${205-155*t} 70% ${38+32*t}%)`;ctx.beginPath();ctx.arc((samples[i]+1)*120,(samples[i+1]+1)*120,4,0,7);ctx.fill();}}
function controls(){for(const id of ['train','restore','field','height'])$(id).disabled=running;$('stop').disabled=!running;$('export').disabled=running||!lastResult;}
function restore(){trainer?.dispose();trainer=null;samples=makeSamples($('field').value);custom=false;lastResult=null;draw();controls();$('status').textContent='Ready · 160 bounded learning steps';$('viewport').innerHTML='<div id="empty">Choose a field and train it.<br>The GPU stays idle until you start.</div>';$('progress').value=0;$('error').textContent=$('heldout').textContent='—';$('steps').textContent='0 / 160';}
function paint(event){if(running)return;const r=canvas.getBoundingClientRect(),x=2*(event.clientX-r.left)/r.width-1,z=2*(event.clientY-r.top)/r.height-1;for(let i=0;i<samples.length;i+=4)if((samples[i]-x)**2+(samples[i+1]-z)**2<.085)samples[i+2]=Number($('height').value);custom=true;lastResult=null;draw();controls();$('status').textContent='Edited examples · train to update the surface';}
let painting=false;canvas.addEventListener('pointerdown',event=>{painting=true;canvas.setPointerCapture(event.pointerId);paint(event);});canvas.addEventListener('pointermove',event=>{if(painting)paint(event);});canvas.addEventListener('pointerup',()=>painting=false);canvas.addEventListener('pointercancel',()=>painting=false);
$('height').addEventListener('input',()=>{$('height-value').textContent=Number($('height').value).toFixed(2);});$('field').addEventListener('change',restore);$('restore').addEventListener('click',restore);
async function train(){
  if(running)return;running=true;cancelled=false;lastResult=null;controls();$('progress').value=0;$('status').textContent='Compiling the forward graph and generated gradients…';
  try{
    trainer?.dispose();trainer=null;$('empty')?.remove();
    trainer=await createTrainer({kind:$('field').value,container:$('viewport'),customSamples:custom?samples:null});
    if(cancelled){trainer.dispose();trainer=null;return;}
    trainer.resize($('viewport').clientWidth,Math.max(300,Math.round($('viewport').clientWidth*.56)));
    const result=await trainer.train({onProgress:({iteration,steps})=>{$('progress').value=iteration;$('steps').textContent=`${iteration} / ${steps}`;$('status').textContent='Learning locally on WebGPU…';}});
    if(!cancelled){lastResult=result;$('progress').value=result.completed;$('steps').textContent=`${result.completed} / 160`;$('error').textContent=`${result.initial.trainingRMSE.toFixed(3)} → ${result.final.trainingRMSE.toFixed(3)}`;$('heldout').textContent=result.final.heldout?`${result.initial.heldout.rmse.toFixed(3)} → ${result.final.heldout.rmse.toFixed(3)}`:'Painted examples';$('status').textContent=result.stopped?'Stopped · partial weights available':'Finished · GPU idle · weights stay editable';}
  }catch(error){$('status').textContent=error.message;trainer?.dispose();trainer=null;}
  finally{running=false;if(cancelled){trainer?.dispose();trainer=null;$('status').textContent='Stopped · GPU released';}controls();}
}
function stop(){cancelled=true;trainer?.stop();}
$('train').addEventListener('click',()=>{active=train();});$('stop').addEventListener('click',stop);
$('export').addEventListener('click',()=>{if(!lastResult)return;const data={format:'trainable-tsl-weights-v1',three:'0.185.1',field:lastResult.kind,weights:lastResult.weights,observations:lastResult.samples,completedSteps:lastResult.completed,customExamples:lastResult.customSamples};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${lastResult.kind}-weights.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();if(!running){trainer?.dispose();trainer=null;}}});
addEventListener('resize',()=>{if(!running)trainer?.resize($('viewport').clientWidth,Math.max(300,Math.round($('viewport').clientWidth*.56)));});
window.trainableLab={train:()=>active=train(),stop,restore,get result(){return lastResult;},get status(){return {running,custom,allocated:!!trainer,trainer:trainer?.status??null};}};
draw();controls();

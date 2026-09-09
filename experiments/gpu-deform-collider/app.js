import {createEngine} from './engine.js';
let engine=null,frame=null,generation=0,pendingGeneration=null;
const status=document.querySelector('#status');
const runButton=document.querySelector('#run');
const view=document.querySelector('#view');
function syncButton(){runButton.disabled=pendingGeneration!==null||engine!==null||document.hidden;}
function release(candidate){
  if(engine===candidate)engine=null;
  candidate?.dispose();
}
function stop(message='Stopped.'){
  generation++;
  if(frame!==null)cancelAnimationFrame(frame);
  frame=null;
  const previous=engine;engine=null;
  try{release(previous);status.textContent=message+(pendingGeneration!==null?' Waiting for initialization cleanup.':' GPU device released.');}
  catch(error){status.textContent=String(error.stack??error);}
  finally{syncButton();}
}
document.querySelector('#stop').onclick=()=>stop();
runButton.onclick=async()=>{
  // A cancelled createEngine() cannot be aborted through its current API. Keep
  // admission closed until its promise settles and its device is disposed.
  if(pendingGeneration!==null||document.hidden)return;
  stop();view.replaceChildren();
  const token=++generation;
  pendingGeneration=token;syncButton();status.textContent='Initializing…';
  const family=document.querySelector('#family').value;
  const method=document.querySelector('#method').value;
  let candidate=null;
  const current=()=>token===generation&&!document.hidden;
  try{
    candidate=await createEngine({family,segments:32,queryCount:128,container:view});
    if(!current())return;
    // Publish this exact instance so Stop can dispose it during later awaits.
    engine=candidate;
    if(method==='interval'){
      await candidate.enableIntervals();
      if(!current())return;
    }
    const validation=await candidate.validate(.4,method);
    if(!current())return;
    if(!validation.ok)throw Error('Correctness gate failed');
    status.textContent=JSON.stringify({fixture:candidate.info,validation},null,2);
    const start=performance.now();
    function draw(now){
      if(!current()||engine!==candidate)return;
      frame=null;
      if(now-start>12000){stop('Preview finished.');return;}
      try{candidate.render((now-start)/2500,method);frame=requestAnimationFrame(draw);}
      catch(error){stop(String(error.stack??error));}
    }
    frame=requestAnimationFrame(draw);
  }catch(error){
    // A rejected await from a cancelled run must not stop a later run or write
    // stale status. The finally block still releases the cancelled instance.
    if(token===generation)stop(String(error.stack??error));
  }finally{
    if(!current()&&candidate){
      try{release(candidate);}catch(error){console.error('Cancelled collider cleanup failed:',error);}
    }
    if(pendingGeneration===token){
      pendingGeneration=null;
      if(!current())status.textContent=status.textContent.replace(/ Waiting for initialization cleanup\.$/,' GPU device released.');
    }
    syncButton();
  }
};
window.colliderLab={createEngine};
document.addEventListener('visibilitychange',()=>{
  if(document.hidden)stop('Preview stopped while the page was hidden.');
  else syncButton();
});
window.addEventListener('pagehide',()=>stop('Page closed.'));
syncButton();

import {solveHologram} from './wave.mjs';
import {OPTICS,maskAmplitude} from './targets.mjs';
self.onmessage=({data})=>{
  try{
    const targets=data.masks.map((mask,i)=>({distance:OPTICS.distances[i],amplitude:maskAmplitude(mask)}));
    const solution=solveHologram(targets,{...OPTICS,iterations:80,onProgress:p=>self.postMessage({type:'progress',id:data.id,iteration:p.iteration,loss:p.loss})});
    self.postMessage({type:'result',id:data.id,solution,targets},[solution.phase.buffer,solution.aperture.buffer,...targets.map(t=>t.amplitude.buffer)]);
  }catch(error){self.postMessage({type:'error',id:data.id,message:error.message});}
};

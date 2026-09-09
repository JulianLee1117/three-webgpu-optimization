import {createExperiment} from './model.js';
import {validateEvaluator} from './validate.js';
import {numericProgram} from './programs.js';
import {validate} from './fixtures.mjs';
import {solveConstrained} from './solver-constrained.mjs';
import {validateRenderedProgram} from './render-parity.js';

export async function runProbe(kind,{fit=false,variant='original'}={}) {
  const experiment=await createExperiment(kind,variant),{fixture,evaluator}=experiment;
  const result={kind,variant,stage:2,status:'failed',adapter:evaluator.adapter,compiler:evaluator.stats,forwardFactoryCalls:experiment.forwardFactoryCalls,validations:[],fits:[],shaders:evaluator.shaders,errors:evaluator.errors};
  try{
    const records=[];
    for(let i=0;i<67;i++)records.push((i+.31)/67,.02*Math.cos(i),.02*Math.sin(i),(i*.61803398875)%1);
    const samples=new Float32Array(records),numeric=(p,t,w)=>numericProgram(kind,p,t,w);
    result.validations.push(await validateEvaluator(evaluator,numeric,fixture.initial,samples));
    if(!result.validations[0].passed)throw Error('Initial native graph parity failed');
    result.initialRender=await validateRenderedProgram({fixture,evaluator});
    if(!result.initialRender.passed)throw Error('Initial rendered geometry parity failed');
    if(fit){
      const methods=kind==='ribbon'?['ad','fd']:['fd','ad'];if(variant==='late-target')methods.reverse();
      for(const method of methods){
        const deadline=performance.now()+12000;
        let firstUpload=null;
        const fitResult=await solveConstrained(fixture,{evaluate:async(p,s,options)=>{const output=await evaluator.evaluate(p,s,{...options,method});if(!firstUpload)firstUpload={requested:[...p],actual:await evaluator.readParameters()};return output;},shouldStop:()=>performance.now()>deadline});
        fitResult.firstUpload=firstUpload;
        if(firstUpload.requested.some((v,i)=>v!==fixture.initial[i])||firstUpload.actual.some((v,i)=>v!==Math.fround(fixture.initial[i])))throw Error('Solver lane started from wrong parameters');
        fitResult.method=method;fitResult.stage=2;fitResult.validation=validate(fixture,fitResult.params);fitResult.float32Validation=validate(fixture,fitResult.params.map(Math.fround));result.fits.push(fitResult);
        const snapshot=await validateEvaluator(evaluator,numeric,fitResult.params,samples);result.validations.push(snapshot);
        if(!snapshot.passed)throw Error('Final native graph parity failed');
        const exported=JSON.parse(JSON.stringify(fitResult.uniformPatch));
        const before=await evaluator.evaluate(fitResult.params,samples),after=await evaluator.evaluate(exported.parameters,samples);
        fitResult.reload={exported,before,after,exact:before.positions.every((v,i)=>v===after.positions[i])&&before.jacobians.every((v,i)=>v===after.jacobians[i])};
        if(!fitResult.reload.exact)throw Error('Parameter export/reload failed');
        fitResult.renderParity=await validateRenderedProgram({fixture,evaluator,parameters:fitResult.params});
        if(!fitResult.renderParity.passed)throw Error('Edited rendered geometry parity failed');
        fitResult.utilityPassed=fitResult.status!=='stopped'&&fitResult.validation.pass&&fitResult.float32Validation.pass&&fitResult.diagnostics.maximumViolation<=.001;
      }
      if(result.fits.some(f=>!f.utilityPassed))throw Error('Declared constrained utility gate failed');
    }
    result.status='passed';
  }catch(error){result.failure=String(error.stack??error);}
  finally{evaluator.dispose();result.finalState=evaluator.status;}
  return result;
}

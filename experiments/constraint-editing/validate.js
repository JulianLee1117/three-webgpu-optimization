/** Independent finite differences use the fixture's numeric forward oracle.
 * GPU AD and GPU finite differences both evaluate clones of one authored graph. */
export async function validateEvaluator(evaluator,numeric,parameters,samples) {
  const weights=Array.from(parameters,Math.fround),records=new Float32Array(samples);
  const ad=await evaluator.evaluate(weights,records,{method:'ad'});
  const fd=await evaluator.evaluate(weights,records,{method:'fd'});
  const n=weights.length,positions=[],jacobians=[],failures=[];
  const tolerances={primalAbsolute:2e-5,primalRelative:5e-5,adAbsolute:1e-4,adRelative:.002,fdAbsolute:.001,fdRelative:.005};
  let maxPrimal=0,maxAD=0,maxFD=0;
  for(let i=0;i<records.length/4;i++){
    const p=Array.from(records.slice(4*i,4*i+3)),t=records[4*i+3],expected=numeric(p,t,weights);
    positions.push(...expected);
    for(let axis=0;axis<3;axis++){
      const index=3*i+axis;
      for(const [method,primal]of [['ad',ad.positions],['fd',fd.positions]]){
        const error=Math.abs(expected[axis]-primal[index]);maxPrimal=Math.max(maxPrimal,error);
        if(!Number.isFinite(error)||error>tolerances.primalAbsolute+tolerances.primalRelative*Math.abs(expected[axis]))failures.push({kind:method+'-primal',sample:i,axis,expected:expected[axis],actual:primal[index]});
      }
      for(let j=0;j<n;j++){
        const plus=[...weights],minus=[...weights],h=1e-5;plus[j]+=h;minus[j]-=h;
        const expectedJ=(numeric(p,t,plus)[axis]-numeric(p,t,minus)[axis])/(2*h),at=index*n+j;
        jacobians.push(expectedJ);
        for(const [method,actual] of [['ad',ad.jacobians[at]],['fd',fd.jacobians[at]]]){
          const delta=Math.abs(actual-expectedJ);if(method==='ad')maxAD=Math.max(maxAD,delta);else maxFD=Math.max(maxFD,delta);
          if(!Number.isFinite(delta)||delta>tolerances[method+'Absolute']+tolerances[method+'Relative']*Math.abs(expectedJ))failures.push({kind:method,sample:i,axis,parameter:j,expected:expectedJ,actual});
        }
      }
    }
  }
  const startingParameters=await evaluator.readParameters();
  if(startingParameters.some((v,i)=>v!==weights[i]))failures.push({kind:'parameter-upload',expected:weights,actual:startingParameters});
  return {passed:!failures.length,parameters:weights,samples:Array.from(records),tolerances,maxPrimal,maxAD,maxFD,failures,
    positions:{gpu:Array.from(ad.positions),cpu:positions,fdGPU:Array.from(fd.positions)},
    jacobians:{ad:Array.from(ad.jacobians),fd:Array.from(fd.jacobians),cpu:jacobians},
    startingParameters};
}

import {numericProgram} from './programs.js';
import {LIMITS, segmentDistance} from './fixtures.mjs';

/**
 * Evaluator contract shared by CPU FD and a prospective native-GPU AD evaluator:
 * evaluate(params, samples, {jacobian=true}) -> {positions, jacobians?}.
 * samples: N vec4 records [s,y,z,time]. positions: 3*N finite scalars.
 * jacobians[((sample*3+axis)*P)+parameter]: d(position)/d(parameter).
 * Arrays may be Float32/64 or ordinary arrays. No GPU API is used by this module.
 */
export function makeNumericEvaluator(kind) {
  const positions=(params,samples)=>{
    const out=new Float64Array(samples.length/4*3);
    for(let i=0;i<samples.length/4;i++)out.set(numericProgram(kind,Array.from(samples.slice(4*i,4*i+3)),samples[4*i+3],params),3*i);
    return out;
  };
  return async (params,samples,{jacobian=true}={})=>{
    const output=positions(params,samples);
    if(!jacobian)return {positions:output};
    const jacobians=new Float64Array(output.length*params.length);
    for(let k=0;k<params.length;k++) {
      const plus=Array.from(params),minus=Array.from(params);plus[k]+=LIMITS.fdStep;minus[k]-=LIMITS.fdStep;
      const a=positions(plus,samples),b=positions(minus,samples);
      for(let i=0;i<output.length;i++)jacobians[i*params.length+k]=(a[i]-b[i])/(2*LIMITS.fdStep);
    }
    return {positions:output,jacobians};
  };
}

function checkEvaluation(result,points,parameters,withJacobian) {
  if(result?.positions?.length!==points*3||Array.from(result.positions).some(x=>!Number.isFinite(x)))throw Error('Invalid/nonfinite evaluator positions');
  if(withJacobian&&(result.jacobians?.length!==points*3*parameters||Array.from(result.jacobians).some(x=>!Number.isFinite(x))))throw Error('Invalid/nonfinite evaluator Jacobians');
}

function makeReference(fixture) {
  const positions=new Float64Array(fixture.samples.length/4*3);
  for(let i=0;i<positions.length/3;i++)positions.set(numericProgram(fixture.kind,Array.from(fixture.samples.slice(4*i,4*i+3)),fixture.samples[4*i+3],fixture.initial),3*i);
  let velocitySq=0;
  for(const [a,b] of fixture.velocityRows)for(let k=0;k<3;k++)velocitySq+=((positions[3*b+k]-positions[3*a+k])/(2*LIMITS.velocityStep))**2;
  return {positions,velocityRMS:Math.sqrt(velocitySq/fixture.velocityRows.length)};
}

// Public for independent residual/Jacobian verification and GPU integration.
export function residuals(fixture,params,evaluation,{avoidObstacle=true,reference=makeReference(fixture)}={}) {
  const P=params.length,positions=evaluation.positions,J=evaluation.jacobians,values=[],derivatives=[];
  const add=(value,terms=[],parameter=null)=>{
    values.push(value);
    if(J)for(let k=0;k<P;k++) {
      let d=parameter===k?.01:0;
      for(const [component,coefficient] of terms)d+=coefficient*J[component*P+k];
      derivatives.push(d);
    }
  };
  for(let k=0;k<3;k++)add(100*(positions[3*fixture.targetRow+k]-fixture.target.position[k]),[[3*fixture.targetRow+k,100]]);
  if(avoidObstacle)for(const [ia,ib] of fixture.collisionRows) {
    const a=Array.from(positions.slice(3*ia,3*ia+3)),b=Array.from(positions.slice(3*ib,3*ib+3));
    const hit=segmentDistance(a,b,fixture.obstacle.center),deficit=fixture.obstacle.radius+LIMITS.crossSectionRadius+LIMITS.trainingClearance-hit.distance;
    const terms=[];
    if(deficit>0&&hit.distance>1e-12)for(let k=0;k<3;k++) {
      const factor=-100*hit.delta[k]/hit.distance;
      terms.push([3*ia+k,factor*(1-hit.u)],[3*ib+k,factor*hit.u]);
    }
    add(100*Math.max(0,deficit),terms);
  }
  const shapeWeight=.5/Math.sqrt(fixture.shapeRows.length);
  for(const i of fixture.shapeRows)for(let k=0;k<3;k++)add(shapeWeight*(positions[3*i+k]-reference.positions[3*i+k]),[[3*i+k,shapeWeight]]);
  const velocityWeight=.5/Math.sqrt(fixture.velocityRows.length)/reference.velocityRMS/(2*LIMITS.velocityStep);
  for(const [a,b] of fixture.velocityRows)for(let k=0;k<3;k++) {
    add(velocityWeight*(positions[3*b+k]-positions[3*a+k]-reference.positions[3*b+k]+reference.positions[3*a+k]),[[3*b+k,velocityWeight],[3*a+k,-velocityWeight]]);
  }
  for(let k=0;k<P;k++)add(.01*(params[k]-fixture.initial[k]),[],k);
  const cost=values.reduce((sum,x)=>sum+x*x,0)/2;
  if(!Number.isFinite(cost)||derivatives.some(x=>!Number.isFinite(x)))throw Error('Nonfinite residual system');
  return {values:new Float64Array(values),jacobians:J?new Float64Array(derivatives):undefined,cost};
}

function linearSolve(matrix,rhs) {
  const n=rhs.length,a=matrix.map((row,i)=>[...row,rhs[i]]);
  for(let k=0;k<n;k++) {
    let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[pivot][k]))pivot=i;
    if(Math.abs(a[pivot][k])<1e-20)return null;
    [a[k],a[pivot]]=[a[pivot],a[k]];
    for(let i=k+1;i<n;i++){const f=a[i][k]/a[k][k];for(let j=k;j<=n;j++)a[i][j]-=f*a[k][j];}
  }
  const x=Array(n).fill(0);
  for(let i=n-1;i>=0;i--){let value=a[i][n];for(let j=i+1;j<n;j++)value-=a[i][j]*x[j];x[i]=value/a[i][i];}
  return x.every(Number.isFinite)?x:null;
}

/** Shared fixed solver. No witness parameters or held-out samples are inspected. */
export async function solve(fixture,{evaluate=makeNumericEvaluator(fixture.kind),onProgress,shouldStop=()=>false,avoidObstacle=true}={}) {
  const start=performance.now(),reference=makeReference(fixture),P=fixture.initial.length,trace=[];
  let params=Array.from(fixture.initial),damping=LIMITS.initialDamping,evaluations=0,accepted=0,status='iteration-limit';
  const call=async (p,jacobian)=>{evaluations++;const result=await evaluate(p,fixture.samples,{jacobian});checkEvaluation(result,fixture.samples.length/4,P,jacobian);return result;};
  for(let iteration=0;iteration<LIMITS.iterations;iteration++) {
    if(await shouldStop()){status='stopped';break;}
    if(performance.now()-start>LIMITS.wallMs){status='wall-limit';break;}
    const evaluation=await call(params,true),system=residuals(fixture,params,evaluation,{avoidObstacle,reference});
    const H=Array.from({length:P},()=>Array(P).fill(0)),g=Array(P).fill(0);
    for(let i=0;i<system.values.length;i++)for(let k=0;k<P;k++) {
      const j=system.jacobians[i*P+k];g[k]+=j*system.values[i];
      for(let l=0;l<=k;l++)H[k][l]+=j*system.jacobians[i*P+l];
    }
    for(let k=0;k<P;k++)for(let l=0;l<k;l++)H[l][k]=H[k][l];
    const gradientNorm=Math.hypot(...g),entry={iteration,cost:system.cost,gradientNorm,damping,params:[...params],trials:[]};
    let improved=false;
    if(gradientNorm<1e-8){status='stationary';trace.push(entry);break;}
    for(let trial=0;trial<LIMITS.trialSteps;trial++) {
      if(await shouldStop()){status='stopped';break;}
      if(performance.now()-start>LIMITS.wallMs){status='wall-limit';break;}
      const A=H.map((row,k)=>row.map((x,l)=>x+(k===l?damping*Math.max(H[k][k],1e-6):0)));
      const step=linearSolve(A,g.map(x=>-x));
      if(!step){entry.trials.push({trial,damping,singular:true});damping*=10;continue;}
      const norm=Math.hypot(...step),scale=Math.min(1,LIMITS.maxStepNorm/Math.max(norm,1e-20));
      const candidate=params.map((x,k)=>Math.max(fixture.lower[k],Math.min(fixture.upper[k],x+step[k]*scale)));
      const next=residuals(fixture,candidate,await call(candidate,false),{avoidObstacle,reference});
      const acceptedTrial=next.cost<system.cost;
      entry.trials.push({trial,damping,cost:next.cost,stepNorm:Math.hypot(...candidate.map((x,k)=>x-params[k])),accepted:acceptedTrial});
      if(acceptedTrial){params=candidate;damping=Math.max(1e-9,damping/3);accepted++;improved=true;break;}
      damping=Math.min(1e12,damping*10);
    }
    trace.push(entry);await onProgress?.({iteration,params:[...params],cost:system.cost,accepted,elapsedMs:performance.now()-start});
    if(status==='stopped'||status==='wall-limit')break;
    if(!improved){status='no-descent';break;}
  }
  const finalEvaluation=await call(params,false),finalResiduals=residuals(fixture,params,finalEvaluation,{avoidObstacle,reference});
  return {params,status,cost:finalResiduals.cost,accepted,iterations:trace.length,evaluations,elapsedMs:performance.now()-start,avoidObstacle,trace,
    finalPositions:Array.from(finalEvaluation.positions),referenceVelocityRMS:reference.velocityRMS,
    uniformPatch:{kind:fixture.kind,parameters:[...params],period:1}};
}

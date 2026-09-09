import {makeFixture, LIMITS, segmentDistance} from './fixtures.mjs';
import {numericProgram} from './programs.js';
import {makeNumericEvaluator} from './solver.mjs';

export const CONSTRAINED_LIMITS=Object.freeze({outer:8,inner:12,backtracks:8,armijo:1e-4,
  initialDamping:.001,initialPenalty:1,penaltyGrowth:5,maxPenalty:1e6,maxStepNorm:.25,
  trainingVelocityNRMSE:.18,trainingPositionRMSE:.40,feasibilityTolerance:.001,
  stationarityTolerance:.001,matrixOptimizerMs:18000});
export const CASES=Object.freeze([{kind:'ribbon',variant:'original'},{kind:'tentacle',variant:'original'},
  {kind:'ribbon',variant:'late-target'},{kind:'tentacle',variant:'late-target'}].map(Object.freeze));

export function fixtureVariant(kind,variant='original') {
  if(!['original','late-target'].includes(variant))throw Error('Unknown constraint variant');
  const fixture=makeFixture(kind);fixture.variant=variant;
  if(variant==='late-target') {
    fixture.witness=kind==='ribbon'?[.30,.22,.08,.95,.14,0,.09,0]:[1,.13,.15,.03,.28,.10,.8,.06];
    fixture.target={p:[1,0,0],time:.625,position:numericProgram(kind,[1,0,0],.625,fixture.witness)};
    fixture.samples[4*fixture.targetRow+3]=.625;
  }
  return fixture;
}

function referenceFor(fixture) {
  const positions=new Float64Array(fixture.samples.length/4*3);
  for(let i=0;i<positions.length/3;i++)positions.set(numericProgram(fixture.kind,Array.from(fixture.samples.slice(i*4,i*4+3)),fixture.samples[i*4+3],fixture.initial),3*i);
  let velocitySq=0,length=0;
  for(const [a,b] of fixture.velocityRows)for(let k=0;k<3;k++)velocitySq+=((positions[3*b+k]-positions[3*a+k])/(2*LIMITS.velocityStep))**2;
  for(const [a,b] of fixture.collisionRows)length+=Math.hypot(...[0,1,2].map(k=>positions[3*b+k]-positions[3*a+k]));
  return {positions,velocitySq,length};
}

/** Build the shared objective and exact first derivatives of its constraints.
 * Position Jacobian layout is identical to solver.mjs. No AD traversal or GPU.
 */
export function constrainedSystem(fixture,params,evaluation,reference=referenceFor(fixture)) {
  const P=params.length,pos=evaluation.positions,J=evaluation.jacobians;
  const objective=[],equalities=[],inequalities=[];
  const row=(value,terms=[],direct=null)=>{
    let derivative;
    if(J){derivative=Array(P).fill(0);for(let k=0;k<P;k++){for(const [i,c] of terms)derivative[k]+=c*J[i*P+k];if(direct)derivative[k]+=direct[k];}}
    return {value,derivative};
  };
  const directRow=(value,derivative)=>({value,derivative:J?derivative:undefined});
  for(let k=0;k<3;k++)equalities.push(row((pos[3*fixture.targetRow+k]-fixture.target.position[k])/.01,[[3*fixture.targetRow+k,100]]));
  let length=0;const lengthGradient=Array(P).fill(0);
  for(const [a,b] of fixture.collisionRows) {
    const pa=Array.from(pos.slice(3*a,3*a+3)),pb=Array.from(pos.slice(3*b,3*b+3));
    const hit=segmentDistance(pa,pb,fixture.obstacle.center),terms=[];
    if(hit.distance>1e-12)for(let k=0;k<3;k++) {
      const factor=-hit.delta[k]/hit.distance/.02;
      terms.push([3*a+k,factor*(1-hit.u)],[3*b+k,factor*hit.u]);
    }
    inequalities.push(row((fixture.obstacle.radius+LIMITS.crossSectionRadius+LIMITS.trainingClearance-hit.distance)/.02,terms));
    const delta=pb.map((x,k)=>x-pa[k]),distance=Math.hypot(...delta);length+=distance;
    if(J&&distance>1e-12)for(let k=0;k<P;k++)for(let axis=0;axis<3;axis++)lengthGradient[k]+=delta[axis]/distance*(J[(3*b+axis)*P+k]-J[(3*a+axis)*P+k]);
  }
  let shapeSq=0;const shapeGradient=Array(P).fill(0),shapeWeight=.5/Math.sqrt(fixture.shapeRows.length);
  for(const i of fixture.shapeRows)for(let axis=0;axis<3;axis++) {
    const component=3*i+axis,delta=pos[component]-reference.positions[component];shapeSq+=delta*delta;
    objective.push(row(shapeWeight*delta,[[component,shapeWeight]]));
    if(J)for(let k=0;k<P;k++)shapeGradient[k]+=2*delta*J[component*P+k];
  }
  let velocitySq=0;const velocityGradient=Array(P).fill(0),inverseDt=1/(2*LIMITS.velocityStep),velocityWeight=.5/Math.sqrt(reference.velocitySq);
  for(const [a,b] of fixture.velocityRows)for(let axis=0;axis<3;axis++) {
    const delta=(pos[3*b+axis]-pos[3*a+axis]-reference.positions[3*b+axis]+reference.positions[3*a+axis])*inverseDt;
    velocitySq+=delta*delta;
    objective.push(row(velocityWeight*delta,[[3*b+axis,velocityWeight*inverseDt],[3*a+axis,-velocityWeight*inverseDt]]));
    if(J)for(let k=0;k<P;k++)velocityGradient[k]+=2*delta*inverseDt*(J[(3*b+axis)*P+k]-J[(3*a+axis)*P+k]);
  }
  const velocityDenominator=reference.velocitySq*CONSTRAINED_LIMITS.trainingVelocityNRMSE**2;
  const shapeDenominator=fixture.shapeRows.length*CONSTRAINED_LIMITS.trainingPositionRMSE**2;
  inequalities.push(directRow(velocitySq/velocityDenominator-1,velocityGradient.map(x=>x/velocityDenominator)));
  inequalities.push(directRow(shapeSq/shapeDenominator-1,shapeGradient.map(x=>x/shapeDenominator)));
  inequalities.push(directRow((length/reference.length-LIMITS.maxLengthRatio)/.2,lengthGradient.map(x=>x/reference.length/.2)));
  inequalities.push(directRow((LIMITS.minLengthRatio-length/reference.length)/.2,lengthGradient.map(x=>-x/reference.length/.2)));
  for(let k=0;k<P;k++){const derivative=Array(P).fill(0);derivative[k]=.01;objective.push(directRow(.01*(params[k]-fixture.initial[k]),derivative));}
  for(const r of [...objective,...equalities,...inequalities])if(!Number.isFinite(r.value)||r.derivative?.some(x=>!Number.isFinite(x)))throw Error('Nonfinite constrained system');
  return {objective,equalities,inequalities,diagnostics:{positionRMSE:Math.sqrt(shapeSq/fixture.shapeRows.length),velocityNRMSE:Math.sqrt(velocitySq/reference.velocitySq),lengthRatio:length/reference.length,
    maximumViolation:Math.max(...equalities.map(r=>Math.abs(r.value)),...inequalities.map(r=>Math.max(0,r.value)))}};
}

function augmented(system,lambdaE,lambdaI,penalty,P,derivatives) {
  let cost=0;const gradient=Array(P).fill(0),H=Array.from({length:P},()=>Array(P).fill(0));
  const add=(value,jacobian)=>{cost+=.5*value*value;if(derivatives)for(let k=0;k<P;k++){gradient[k]+=value*jacobian[k];for(let l=0;l<=k;l++)H[k][l]+=jacobian[k]*jacobian[l];}};
  for(const row of system.objective)add(row.value,row.derivative);
  const scale=Math.sqrt(penalty);
  for(let i=0;i<system.equalities.length;i++){const row=system.equalities[i];add(scale*(row.value+lambdaE[i]/penalty),derivatives?row.derivative.map(x=>scale*x):null);}
  for(let i=0;i<system.inequalities.length;i++){const row=system.inequalities[i],shift=row.value+lambdaI[i]/penalty;if(shift>0)add(scale*shift,derivatives?row.derivative.map(x=>scale*x):null);}
  if(derivatives)for(let k=0;k<P;k++)for(let l=0;l<k;l++)H[l][k]=H[k][l];
  return {cost,gradient,H};
}

function linearSolve(A,b) {
  const n=b.length,M=A.map((r,i)=>[...r,b[i]]);
  for(let k=0;k<n;k++) {
    let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;
    if(Math.abs(M[p][k])<1e-20)return null;
    [M[k],M[p]]=[M[p],M[k]];
    for(let i=k+1;i<n;i++){const f=M[i][k]/M[k][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}
  }
  const x=Array(n).fill(0);
  for(let i=n-1;i>=0;i--){let v=M[i][n];for(let j=i+1;j<n;j++)v-=M[i][j]*x[j];x[i]=v/M[i][i];}
  return x.every(Number.isFinite)?x:null;
}

function projectedGradient(params,gradient,fixture) {
  return Math.max(...gradient.map((g,k)=>(params[k]<=fixture.lower[k]+1e-10&&g>0)||(params[k]>=fixture.upper[k]-1e-10&&g<0)?0:Math.abs(g)));
}

export async function solveConstrained(fixture,{evaluate=makeNumericEvaluator(fixture.kind),onProgress,shouldStop=()=>false}={}) {
  const start=performance.now(),P=fixture.initial.length,reference=referenceFor(fixture),trace=[];
  let params=Array.from(fixture.initial),penalty=CONSTRAINED_LIMITS.initialPenalty,damping=CONSTRAINED_LIMITS.initialDamping;
  let evaluations=0,accepted=0,status='iteration-limit',previousViolation=Infinity,lambdaE,lambdaI,lastSystem;
  const call=async (p,jacobian)=>{
    evaluations++;const r=await evaluate(p,fixture.samples,{jacobian});
    if(r?.positions?.length!==fixture.samples.length/4*3||Array.from(r.positions).some(x=>!Number.isFinite(x)))throw Error('Invalid constrained evaluator positions');
    if(jacobian&&(r.jacobians?.length!==r.positions.length*P||Array.from(r.jacobians).some(x=>!Number.isFinite(x))))throw Error('Invalid constrained evaluator Jacobians');
    return r;
  };
  for(let outer=0;outer<CONSTRAINED_LIMITS.outer;outer++) {
    const round={outer,penalty,iterations:[]};trace.push(round);
    for(let inner=0;inner<CONSTRAINED_LIMITS.inner;inner++) {
      if(await shouldStop()){status='stopped';break;}
      const evaluation=await call(params,true),system=constrainedSystem(fixture,params,evaluation,reference);lastSystem=system;
      lambdaE??=Array(system.equalities.length).fill(0);lambdaI??=Array(system.inequalities.length).fill(0);
      const model=augmented(system,lambdaE,lambdaI,penalty,P,true),entry={inner,cost:model.cost,diagnostics:system.diagnostics,damping,params:[...params],trials:[]};
      round.iterations.push(entry);
      const A=model.H.map((row,k)=>row.map((x,l)=>x+(k===l?damping*Math.max(model.H[k][k],1e-6):0)));
      const step=linearSolve(A,model.gradient.map(x=>-x));
      let improved=false;
      if(step) {
        const norm=Math.hypot(...step),cap=Math.min(1,CONSTRAINED_LIMITS.maxStepNorm/Math.max(norm,1e-20));
        for(let trial=0;trial<CONSTRAINED_LIMITS.backtracks;trial++) {
          if(await shouldStop()){status='stopped';break;}
          const scale=cap*2**(-trial),candidate=params.map((x,k)=>Math.max(fixture.lower[k],Math.min(fixture.upper[k],x+step[k]*scale)));
          const directional=model.gradient.reduce((sum,g,k)=>sum+g*(candidate[k]-params[k]),0);
          const nextSystem=constrainedSystem(fixture,candidate,await call(candidate,false),reference);
          const next=augmented(nextSystem,lambdaE,lambdaI,penalty,P,false);
          const accept=directional<0&&next.cost<=model.cost+CONSTRAINED_LIMITS.armijo*directional;
          entry.trials.push({trial,cost:next.cost,directional,stepNorm:Math.hypot(...candidate.map((x,k)=>x-params[k])),accepted:accept});
          if(accept){params=candidate;accepted++;improved=true;damping=Math.max(1e-9,damping/2);break;}
        }
      } else entry.singular=true;
      if(!improved)damping=Math.min(1e9,damping*10);
      await onProgress?.({outer,inner,params:[...params],accepted,elapsedMs:performance.now()-start,diagnostics:system.diagnostics});
      if(status==='stopped')break;
    }
    if(status==='stopped')break;
    lastSystem=constrainedSystem(fixture,params,await call(params,true),reference);
    lambdaE=lambdaE.map((x,i)=>x+penalty*lastSystem.equalities[i].value);
    lambdaI=lambdaI.map((x,i)=>Math.max(0,x+penalty*lastSystem.inequalities[i].value));
    const lagrangian=Array(P).fill(0);
    for(const r of lastSystem.objective)for(let k=0;k<P;k++)lagrangian[k]+=r.value*r.derivative[k];
    for(let i=0;i<lambdaE.length;i++)for(let k=0;k<P;k++)lagrangian[k]+=lambdaE[i]*lastSystem.equalities[i].derivative[k];
    for(let i=0;i<lambdaI.length;i++)for(let k=0;k<P;k++)lagrangian[k]+=lambdaI[i]*lastSystem.inequalities[i].derivative[k];
    const stationarity=projectedGradient(params,lagrangian,fixture),violation=lastSystem.diagnostics.maximumViolation;
    round.final={params:[...params],diagnostics:lastSystem.diagnostics,stationarity,lambdaE:[...lambdaE],lambdaI:[...lambdaI]};
    if(violation<=CONSTRAINED_LIMITS.feasibilityTolerance&&stationarity<=CONSTRAINED_LIMITS.stationarityTolerance){status='converged';break;}
    if(violation>previousViolation*.5)penalty=Math.min(CONSTRAINED_LIMITS.maxPenalty,penalty*CONSTRAINED_LIMITS.penaltyGrowth);
    previousViolation=violation;
  }
  const finalEvaluation=await call(params,false);lastSystem=constrainedSystem(fixture,params,finalEvaluation,reference);
  return {params,status,accepted,evaluations,elapsedMs:performance.now()-start,trace,diagnostics:lastSystem.diagnostics,
    objectiveCost:lastSystem.objective.reduce((sum,r)=>sum+.5*r.value*r.value,0),lambdaE,lambdaI,penalty,
    finalPositions:Array.from(finalEvaluation.positions),uniformPatch:{kind:fixture.kind,variant:fixture.variant,parameters:[...params],period:1}};
}

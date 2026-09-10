import test from 'node:test';
import assert from 'node:assert/strict';
import {createMechanics,stableNeoHookean,symmetricEigen} from './mechanics.mjs';

function fixture({handles=2,young=1000,density=2,yOffset=.55,duplicate=false}={}){
  const restPositions=[],weights=[],gradients=[];
  for(let y=0;y<3;y++)for(let z=0;z<3;z++)for(let x=0;x<3;x++){
    const p=[x*.4-.4,y*.4-.4+yOffset,z*.4-.4];restPositions.push(...p);
    for(let h=0;h<handles;h++){
      if(h===handles-1||duplicate){weights.push(1);gradients.push(0,0,0);}
      else{const a=.7*p[0]+.4*p[1]-.3*p[2];weights.push(Math.sin(a));gradients.push(.7*Math.cos(a),.4*Math.cos(a),-.3*Math.cos(a));}
    }
  }
  return {restPositions,weights,gradients,volumes:Array(27).fill(1/27),density,young,poisson:.3};
}
const close=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<=t,`${a} versus ${b}, error ${Math.abs(a-b)}`);

test('stable Neo-Hookean stress is the derivative, including singular and inverted F',()=>{
  for(const F of [[1,.1,0,0,1.2,.2,0,0,.8],[1,0,0,0,1,0,0,0,0],[1,.1,0,0,1,0,0,0,-.5]]){
    const actual=stableNeoHookean(F,2345,.32);
    for(let j=0;j<9;j++){const a=[...F],b=[...F];a[j]+=1e-6;b[j]-=1e-6;const d=(stableNeoHookean(a,2345,.32).energy-stableNeoHookean(b,2345,.32).energy)/2e-6;close(actual.gradient[j],d,2e-6);}
  }
});

test('full affine-blend energy, gradient and Hessian include weight derivatives',()=>{
  const model=createMechanics(fixture()),q=Float64Array.from({length:model.dofs},(_,i)=>.04*Math.sin(i*1.7)),predicted=Float64Array.from(q,x=>x*.4);
  const options={predicted,dt:.027,gravity:[.2,-9.81,.4],drag:{restPosition:[.17,.82,-.13],weights:[.21,1],target:[.22,.91,-.1],stiffness:37},hessian:true};
  const e=model.energyGradient(q,options),D=model.dofs;
  for(let i=0;i<D;i++){
    const a=Float64Array.from(q),b=Float64Array.from(q),h=1e-6;a[i]+=h;b[i]-=h;
    const ea=model.energyGradient(a,options),eb=model.energyGradient(b,options);
    close(e.gradient[i],(ea.energy-eb.energy)/(2*h),3e-6);
    for(let j=0;j<D;j++){close(e.hessian[j*D+i],(ea.gradient[j]-eb.gradient[j])/(2*h),1e-5);close(e.hessian[j*D+i],e.hessian[i*D+j],1e-8);}
  }
});

test('constant handle represents stress-free rigid translation and finite rotation',()=>{
  const model=createMechanics(fixture()),q=new Float64Array(model.dofs),h=model.handleCount-1,a=.73,c=Math.cos(a),s=Math.sin(a);
  q[12*h]=c-1;q[12*h+1]=-s;q[12*h+4]=s;q[12*h+5]=c-1;q[12*h+3]=.5;q[12*h+7]=-.2;q[12*h+11]=.13;
  const e=model.energyGradient(q);close(e.energy,0,2e-12);assert.ok(Math.max(...Array.from(e.gradient,Math.abs))<1e-10);
});

test('dependent affine modes are dropped, with mass and point projection audited',()=>{
  const model=createMechanics(fixture({duplicate:true}));assert.equal(model.diagnostics.scalarMassRank,4);assert.equal(model.reducedDofs,12);
  assert.ok(model.diagnostics.whiteningError<1e-12);
  const q=Float64Array.from({length:model.dofs},(_,i)=>.1*Math.sin(i)),a=model.evaluatePositions(q),b=model.evaluatePositions(model.fromReduced(model.toReduced(q)));
  for(let i=0;i<a.length;i++)close(a[i],b[i],1e-12);
});

test('zero elasticity implicit Euler reproduces uniform gravitational acceleration',()=>{
  const model=createMechanics(fixture({handles:1,young:0})),state=model.createState(),dt=.025,g=[.7,-9.81,.2];
  state.velocity[3]=.3;state.velocity[7]=-.1;
  const next=model.step(state,{dt,gravity:g,maxMilliseconds:1000}),p=model.evaluatePositions(next.state.q),source=fixture({handles:1,young:0}).restPositions;
  assert.equal(next.diagnostics.status,'converged');
  for(let i=0;i<p.length;i++){const axis=i%3;close(p[i],source[i]+dt*[.3,-.1,0][axis]+dt*dt*g[axis],2e-10);}
});

test('hard pins recover prescribed positions; arbitrary visible-point spring matches indexed spring',()=>{
  const input=fixture({handles:1}),model=createMechanics(input),state=model.createState();state.q[3]=.15;state.q[7]=.1;
  const center=13,target=input.restPositions.slice(3*center,3*center+3),next=model.step(state,{dt:.02,gravity:[0,0,0],pins:[{index:center,target}],maxMilliseconds:1000});
  assert.ok(next.diagnostics.pinError<1e-9);
  const p=model.evaluatePositions(next.state.q);for(let j=0;j<3;j++)close(p[3*center+j],target[j],1e-9);
  const point=5,at=input.restPositions.slice(3*point,3*point+3),to=at.map((x,i)=>x+[.1,-.1,.2][i]);
  const a=model.energyGradient(state.q,{drag:{index:point,target:to,stiffness:23}});
  const b=model.energyGradient(state.q,{drag:{restPosition:at,weights:[1],target:to,stiffness:23}});
  close(a.energy,b.energy,1e-12);for(let i=0;i<model.dofs;i++)close(a.gradient[i],b.gradient[i],1e-12);
});

test('sampled hard floor remains feasible and accepted Newton steps decrease the incremental potential',()=>{
  const model=createMechanics(fixture({handles:1,young:100,density:1}));let state=model.createState(),contact=false;
  for(let i=0;i<24;i++){
    const next=model.step(state,{dt:.035,floor:0,maxIterations:48,maxMilliseconds:1000});state=next.state;
    assert.ok(next.diagnostics.floorPenetration<=1e-8);assert.ok(next.diagnostics.pinError<=1e-8);
    assert.ok(next.diagnostics.finalEnergy<=next.diagnostics.initialEnergy+1e-10);
    for(const h of next.diagnostics.history)if(h.nextEnergy!==undefined)assert.ok(h.nextEnergy<=h.energy+1e-10);
    contact ||=next.diagnostics.activeFloorSamples>0;
  }
  assert.ok(contact);const p=model.evaluatePositions(state.q);assert.ok(Math.min(...Array.from({length:27},(_,i)=>p[3*i+1]))>=-1e-8);
});

test('released stretched body reduces elastic energy with zero external forcing',()=>{
  const model=createMechanics(fixture({handles:1,young:100,density:1})),state=model.createState();
  state.q[0]=.18;state.q[5]=-.08;state.q[10]=.07;
  const before=model.energyGradient(state.q),next=model.step(state,{dt:.05,gravity:[0,0,0],maxMilliseconds:1000}),after=model.energyGradient(next.state.q);
  assert.equal(next.diagnostics.status,'converged');
  assert.ok(before.elasticEnergy>0);assert.ok(after.elasticEnergy<before.elasticEnergy*.8);
  assert.ok(next.diagnostics.finalEnergy<=next.diagnostics.initialEnergy+1e-10);
});

test('near-stationary energy cancellation accepts a feasible gradient-reducing Newton step, without relaxing convergence',()=>{
  // This fixed case stalls in the previous strict-Armijo implementation at
  // projected gradient 1.5138e-6. Keep its arithmetic/order fixed: the intended
  // regression is subtraction roundoff near the stress-free rest energy.
  const restPositions=[];
  for(let y=0;y<3;y++)for(let z=0;z<3;z++)for(let x=0;x<3;x++)restPositions.push(x*.4-.4,y*.4+.15,z*.4-.4);
  const model=createMechanics({restPositions,weights:Array(27).fill(1),gradients:Array(81).fill(0),
    volumes:Array(27).fill(1/27),density:1,young:1000,poisson:.3}),state=model.createState();
  for(let j=0;j<12;j++)state.q[j]=2e-10*Math.sin(3.4+j*2.3);
  const next=model.step(state,{dt:.05,gravity:[0,0,0],pins:[{index:13,target:restPositions.slice(39,42)}],
    floor:0,maxMilliseconds:1000,gradientTolerance:1e-6}),d=next.diagnostics;
  assert.equal(d.status,'converged');assert.equal(d.roundoffAcceptedSteps,1);
  assert.ok(d.projectedGradient<1e-10);assert.ok(d.pinError<1e-12);assert.equal(d.floorPenetration,0);
  const h=d.history.find(x=>x.acceptanceReason==='energy-roundoff-gradient-reduction');
  assert.ok(h.projectedGradient>1e-6);assert.ok(h.trialProjectedGradient<=.5*h.projectedGradient);
  assert.ok(h.nextEnergy>h.energy); // An actual energy-roundoff rejection.
  close(h.roundoffAllowance,64*Number.EPSILON*(h.energyComponentMagnitude+h.trialEnergyComponentMagnitude),1e-25);
  assert.ok(h.nextEnergy-h.energy<=h.roundoffAllowance);assert.ok(h.predictedDecrease<=h.roundoffAllowance);
});

test('invalid/infeasible inputs reject rather than silently regularizing constraints',()=>{
  assert.throws(()=>createMechanics({...fixture(),poisson:.5}),/Poisson/);
  const model=createMechanics(fixture({handles:1}));
  assert.throws(()=>model.step(model.createState(),{pins:[{index:0,target:[0,0,0]},{index:0,target:[1,0,0]}]}),/inconsistent/);
  assert.throws(()=>model.step(model.createState(),{floor:2}),/feasible start/);
  const e=symmetricEigen([2,1,1,2],2);close(e.values[0],3,1e-12);close(e.values[1],1,1e-12);
});

import {numericProgram} from './programs.js';

export const KINDS=Object.freeze(['ribbon','tentacle']);
export const LIMITS=Object.freeze({parameters:8, length:3, crossSectionRadius:.025,
  trainSections:33, trainTimes:16, velocitySections:9, velocityTimes:8, velocityStep:1/1024,
  heldoutSections:129, heldoutTimes:128, heldoutVelocitySections:33, heldoutVelocityTimes:64,
  heldoutVelocityStep:1/4096, iterations:48, trialSteps:6, wallMs:10000, fdStep:1e-4,
  maxStepNorm:.25, initialDamping:.001, targetTolerance:.01, requiredClearance:.005,
  trainingClearance:.02, maxPositionRMSE:.42, maxVelocityNRMSE:.20,
  minLengthRatio:.8, maxLengthRatio:1.2, controlPenetration:.005});

export function makeFixture(kind) {
  if (!KINDS.includes(kind)) throw Error('Unknown fixture');
  const isRibbon=kind==='ribbon';
  const initial=isRibbon?[0,.22,0,0,.14,0,.09,0]:[.9,.12,.2,.04,0,.10,0,.06];
  const witness=isRibbon?[.4,.22,0,.9,.14,0,.09,0]:[1.02,.12,.2,.04,.22,.10,.7,.06];
  const lower=isRibbon?[-.7,.08,-.6,-1.2,.04,-.7,.03,-.6]:[.65,.04,-.6,-.12,-.7,.03,-1.2,-.2];
  const upper=isRibbon?[.7,.4,.6,1.2,.28,.7,.18,.6]:[1.25,.25,.6,.12,.7,.2,1.2,.2];
  const target={p:[1,0,0],time:.25,position:numericProgram(kind,[1,0,0],.25,witness)};
  const referenceMid=numericProgram(kind,[.5,0,0],.25,initial);
  const obstacle={center:isRibbon?[1.5,.10,0]:[referenceMid[0],referenceMid[1],.04],radius:isRibbon?.09:.065};
  const rows=[], shapeRows=[], collisionRows=[], velocityRows=[];
  const append=(s,time)=>{const index=rows.length/4;rows.push(s,0,0,time);return index;};
  for(let t=0;t<LIMITS.trainTimes;t++) {
    const ring=[];
    for(let s=0;s<LIMITS.trainSections;s++) {const i=append(s/(LIMITS.trainSections-1),t/LIMITS.trainTimes);ring.push(i);shapeRows.push(i);}
    for(let s=0;s+1<ring.length;s++) collisionRows.push([ring[s],ring[s+1]]);
  }
  for(let t=0;t<LIMITS.velocityTimes;t++) for(let s=0;s<LIMITS.velocitySections;s++) {
    const time=(t+.37)/LIMITS.velocityTimes;
    velocityRows.push([append(s/(LIMITS.velocitySections-1),time-LIMITS.velocityStep),append(s/(LIMITS.velocitySections-1),time+LIMITS.velocityStep)]);
  }
  const targetRow=append(1,target.time);
  return {kind,initial,lower,upper,witness,target,obstacle,samples:new Float64Array(rows),shapeRows,collisionRows,velocityRows,targetRow,
    train:{sections:LIMITS.trainSections,times:Array.from({length:LIMITS.trainTimes},(_,i)=>i/LIMITS.trainTimes)},
    heldout:{sections:LIMITS.heldoutSections,times:Array.from({length:LIMITS.heldoutTimes},(_,i)=>(i+.5)/LIMITS.heldoutTimes)}};
}

export function segmentDistance(a,b,center) {
  let vv=0,pv=0;const v=b.map((x,i)=>x-a[i]);
  for(let k=0;k<3;k++){vv+=v[k]*v[k];pv+=(center[k]-a[k])*v[k];}
  const u=vv>1e-20?Math.max(0,Math.min(1,pv/vv)):0;
  const delta=a.map((x,k)=>x+u*v[k]-center[k]);
  return {distance:Math.hypot(...delta),u,delta};
}

// Dense independent CPU validation of the rendered centerline capsule envelope.
// The sphere-to-segment distances are exact for this polyline, not continuous time.
export function validate(fixture,params,{translation=[0,0,0]}={}) {
  const point=(s,t,w=params,reference=false)=>numericProgram(fixture.kind,[s,0,0],t,w).map((x,k)=>x+(reference?0:translation[k]));
  let minClearance=Infinity,positionSq=0,positions=0,length=0,originalLength=0,worstSample=null;
  for(const time of fixture.heldout.times) {
    let previous=null,previousOriginal=null;
    for(let i=0;i<LIMITS.heldoutSections;i++) {
      const s=i/(LIMITS.heldoutSections-1),p=point(s,time),original=point(s,time,fixture.initial,true);
      for(let k=0;k<3;k++)positionSq+=(p[k]-original[k])**2;
      positions++;
      if(previous) {
        const clearance=segmentDistance(previous,p,fixture.obstacle.center).distance-fixture.obstacle.radius-LIMITS.crossSectionRadius;
        if(clearance<minClearance){minClearance=clearance;worstSample={time,segment:i-1};}
        length+=Math.hypot(...p.map((v,k)=>v-previous[k]));originalLength+=Math.hypot(...original.map((v,k)=>v-previousOriginal[k]));
      }
      previous=p;previousOriginal=original;
    }
  }
  let velocitySq=0,originalVelocitySq=0,currentVelocitySq=0;
  for(let t=0;t<LIMITS.heldoutVelocityTimes;t++)for(let i=0;i<LIMITS.heldoutVelocitySections;i++) {
    const s=i/(LIMITS.heldoutVelocitySections-1),time=(t+.31)/LIMITS.heldoutVelocityTimes,h=LIMITS.heldoutVelocityStep;
    const a=point(s,time-h),b=point(s,time+h),c=point(s,time-h,fixture.initial,true),d=point(s,time+h,fixture.initial,true);
    for(let k=0;k<3;k++){const velocity=(b[k]-a[k])/(2*h),original=(d[k]-c[k])/(2*h);velocitySq+=(velocity-original)**2;originalVelocitySq+=original**2;currentVelocitySq+=velocity**2;}
  }
  const tip=point(1,fixture.target.time),targetError=Math.hypot(...tip.map((x,k)=>x-fixture.target.position[k]));
  const anchorError=Math.max(...fixture.heldout.times.map(time=>Math.hypot(...point(0,time))));
  const loopError=Math.max(...Array.from({length:33},(_,i)=>Math.hypot(...point(i/32,0).map((v,k)=>v-point(i/32,1)[k]))));
  const positionRMSE=Math.sqrt(positionSq/positions),velocityNRMSE=Math.sqrt(velocitySq/originalVelocitySq),speedRatio=Math.sqrt(currentVelocitySq/originalVelocitySq),lengthRatio=length/originalLength;
  const finite=[minClearance,positionRMSE,velocityNRMSE,speedRatio,lengthRatio,targetError,anchorError,loopError].every(Number.isFinite);
  const gates={finite,target:targetError<=LIMITS.targetTolerance,clearance:minClearance>=LIMITS.requiredClearance,
    position:positionRMSE<=LIMITS.maxPositionRMSE,velocity:velocityNRMSE<=LIMITS.maxVelocityNRMSE,
    shape:lengthRatio>=LIMITS.minLengthRatio&&lengthRatio<=LIMITS.maxLengthRatio,anchor:anchorError<1e-8,loop:loopError<1e-8};
  return {pass:Object.values(gates).every(Boolean),gates,targetError,minClearance,worstSample,positionRMSE,velocityNRMSE,speedRatio,lengthRatio,anchorError,loopError,tip};
}

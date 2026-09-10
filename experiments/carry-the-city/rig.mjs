import * as THREE from 'three';
import {samplePoint} from './motion.mjs';

// One sampled field supplies rendered surfaces and the frames of their riders.
// Translation/steering belongs to the game; this adapter does not simulate thrust.
export function createMotionRig(town,clips){
 const up=new THREE.Vector3(),x=new THREE.Vector3(),z=new THREE.Vector3(),matrix=new THREE.Matrix4(),q=new THREE.Quaternion(),yaw=new THREE.Quaternion(),axis=new THREE.Vector3(0,1,0);
 const field=(p,time,blend)=>{
  const a=samplePoint(clips.swim,time,p),b=samplePoint(clips.fly,time,p);
  return a.map((v,i)=>v*(1-blend)+b[i]*blend);
 };
 function frame(p,time,blend){
  const e=.015,center=field(p,time,blend),px=field([p[0]+e,p[1],p[2]],time,blend),pz=field([p[0],p[1],p[2]+e],time,blend);
  x.set(px[0]-center[0],px[1]-center[1],px[2]-center[2]).normalize();z.set(pz[0]-center[0],pz[1]-center[1],pz[2]-center[2]).normalize();up.crossVectors(z,x).normalize();z.crossVectors(x,up).normalize();matrix.makeBasis(x,up,z);q.setFromRotationMatrix(matrix);
  return {position:center,quaternion:q.clone(),normal:up.clone()};
 }
 function update(time,blend){
  for(const surface of town.surfaces){
   const p=surface.mesh.geometry.attributes.position,r=surface.restPositions;
   for(let i=0;i<p.count;i++){const v=field([r[i*3],r[i*3+1],r[i*3+2]],time,blend);p.setXYZ(i,...v);}
   p.needsUpdate=true;surface.mesh.geometry.computeVertexNormals();surface.mesh.frustumCulled=false;
  }
  for(const a of town.attachments){if(a.detached)continue;const f=frame(a.anchor,time,blend);a.object.position.set(...f.position);yaw.setFromAxisAngle(axis,a.baseYaw??0);a.object.quaternion.copy(f.quaternion).multiply(yaw);}
 }
 function worldPoint(p,time,blend){town.root.updateMatrixWorld(true);return town.root.localToWorld(new THREE.Vector3(...field(p,time,blend)));}
 return {update,field,frame,worldPoint};
}

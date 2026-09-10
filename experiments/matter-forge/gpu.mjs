// Bounded MLS-MPM/APIC experiment. Standard transfer and constitutive models;
// this is not a new simulation algorithm. CPU parity must precede gameplay use.
import { normalizeConfig, makeState, determinant } from './cpu.mjs';
const common = /* wgsl */`
struct Particle { x: vec4f, v: vec4f, C: mat3x3f, F: mat3x3f, aux: vec4f }
struct Cell { x: atomic<i32>, y: atomic<i32>, z: atomic<i32>, mass: atomic<i32>, solid: atomic<i32> }
struct Params { n:u32, count:u32, boxes:u32, anchors:u32, dt:f32, bulk:f32, gamma:f32, mu:f32,
  lambda:f32, boundary:f32, scale:f32, viscosity:f32, gravity:vec4f }
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> grid:array<Cell>;
@group(0) @binding(2) var<storage,read_write> velocities:array<vec4f>;
@group(0) @binding(3) var<storage,read> boxes:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> faults:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> u:Params;
fn index(q:vec3i)->u32 { return u32((q.x*i32(u.n)+q.y)*i32(u.n)+q.z); }
fn enc(x:f32)->i32 { if(!(abs(x)<2000.)){atomicOr(&faults[0],64u);return 0;}return i32(round(x*u.scale)); }
fn add(destination:ptr<storage,atomic<i32>,read_write>,value:i32){
 let previous=atomicAdd(destination,value);
 if(value>0){if(previous>2147483647-value){atomicOr(&faults[0],128u);}}
 if(value<0){if(previous<(-2147483647-1)-value){atomicOr(&faults[0],128u);}}
}
fn dec(x:i32)->f32 { return f32(x)/u.scale; }
fn weights(f:vec3f)->array<vec3f,3> {return array<vec3f,3>(.5*(1.5-f)*(1.5-f),.75-(f-1)*(f-1),.5*(f-.5)*(f-.5));}
fn inside(p:vec3f,k:u32)->bool { return all(p>=boxes[2*k].xyz)&&all(p<=boxes[2*k+1].xyz); }
fn identity()->mat3x3f {return mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1));}
`;
const kernels = {
clear: /* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 let k=id.x;if(k>=u.n*u.n*u.n){return;}
 atomicStore(&grid[k].x,0);atomicStore(&grid[k].y,0);atomicStore(&grid[k].z,0);
 atomicStore(&grid[k].mass,0);atomicStore(&grid[k].solid,0);velocities[k]=vec4f(0);
}`,
mass: /* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=u.count){return;}let p=particles[id.x];let base=vec3i(floor(p.x.xyz-.5));
 let f=p.x.xyz-vec3f(base);let w=weights(f);
 for(var a=0;a<3;a++){for(var b=0;b<3;b++){for(var c=0;c<3;c++){
  let q=base+vec3i(a,b,c);let k=index(q);let d=vec3f(q)-p.x.xyz;
  let m=p.v.w*w[a].x*w[b].y*w[c].z;let momentum=m*(p.v.xyz+p.C*d);
  add(&grid[k].x,enc(momentum.x));add(&grid[k].y,enc(momentum.y));add(&grid[k].z,enc(momentum.z));
  add(&grid[k].mass,enc(m));if(p.x.w>.5){add(&grid[k].solid,enc(m));}
 }}}
}`,
stress: /* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=u.count){return;}let p=particles[id.x];let base=vec3i(floor(p.x.xyz-.5));
 let f=p.x.xyz-vec3f(base);let w=weights(f);var volume=p.aux.x;var stress=identity()*0.;
 if(p.x.w<.5){
  var density=0.;for(var a=0;a<3;a++){for(var b=0;b<3;b++){for(var c=0;c<3;c++){
   density+=w[a].x*w[b].y*w[c].z*dec(atomicLoad(&grid[index(base+vec3i(a,b,c))].mass));
  }}}
  if(!(density>1e-8)){atomicOr(&faults[0],1u);return;}
  volume=p.v.w/density;let rho0=p.v.w/p.aux.x;
  let pressure=u.bulk*max(pow(density/rho0,u.gamma)-1.,0.);
  stress=-pressure*identity()+u.viscosity*(p.C+transpose(p.C));
 }else{
  let j=determinant(p.F);if(!(j>.05&&j<20.)){atomicOr(&faults[0],2u);return;}
  stress=u.mu*(p.F*transpose(p.F)-identity())+u.lambda*log(j)*identity();
 }
 let affine=-4.*u.dt*volume*stress;
 for(var a=0;a<3;a++){for(var b=0;b<3;b++){for(var c=0;c<3;c++){
  let q=base+vec3i(a,b,c);let k=index(q);let d=vec3f(q)-p.x.xyz;
  let momentum=w[a].x*w[b].y*w[c].z*(affine*d);
  if(any(abs(momentum)>vec3f(100.))||any(momentum!=momentum)){atomicOr(&faults[0],4u);return;}
  add(&grid[k].x,enc(momentum.x));add(&grid[k].y,enc(momentum.y));add(&grid[k].z,enc(momentum.z));
 }}}
}`,
grid: /* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 let k=id.x;if(k>=u.n*u.n*u.n){return;}let m=dec(atomicLoad(&grid[k].mass));if(m<=1e-8){return;}
 var v=vec3f(dec(atomicLoad(&grid[k].x)),dec(atomicLoad(&grid[k].y)),dec(atomicLoad(&grid[k].z)))/m+u.dt*u.gravity.xyz;
 let q=vec3f(f32(k/(u.n*u.n)),f32((k/u.n)%u.n),f32(k%u.n));
 for(var a=0;a<3;a++){if(q[a]<u.boundary&&v[a]<0.){v[a]=0.;}if(q[a]>f32(u.n)-1.-u.boundary&&v[a]>0.){v[a]=0.;}}
 for(var b=0u;b<u.boxes;b++){if(inside(q,b)){v=vec3f(0);}}
 if(atomicLoad(&grid[k].solid)>0){for(var b=u.boxes;b<u.boxes+u.anchors;b++){if(inside(q,b)){v=vec3f(0);}}}
 if(any(abs(v)>vec3f(100.))||any(v!=v)){atomicOr(&faults[0],8u);return;}
 velocities[k]=vec4f(v,m);
}`,
gather: /* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=u.count){return;}var p=particles[id.x];let base=vec3i(floor(p.x.xyz-.5));
 let f=p.x.xyz-vec3f(base);let w=weights(f);var v=vec3f(0);var C=identity()*0.;
 for(var a=0;a<3;a++){for(var b=0;b<3;b++){for(var c=0;c<3;c++){
  let q=base+vec3i(a,b,c);let d=vec3f(q)-p.x.xyz;
  let weighted=w[a].x*w[b].y*w[c].z*velocities[index(q)].xyz;
  v+=weighted;C+=4.*mat3x3f(weighted*d.x,weighted*d.y,weighted*d.z);
 }}}
 let x=p.x.xyz+u.dt*v;
 if(any(x<vec3f(.5))||any(x>=vec3f(f32(u.n)-1.5))||any(x!=x)){atomicOr(&faults[0],16u);return;}
 if(p.x.w>.5){p.F=(identity()+u.dt*C)*p.F;let j=determinant(p.F);if(!(j>.05&&j<20.)){atomicOr(&faults[0],32u);return;}}
 p.x=vec4f(x,p.x.w);p.v=vec4f(v,p.v.w);p.C=C;particles[id.x]=p;
}`,
};

export const shaderSources=Object.fromEntries(Object.entries(kernels).map(([k,v])=>[k,common+v]));
export function packState(state){
 const count=state.ids.length,out=new Float32Array(count*36);
 for(let i=0;i<count;i++){
  const k=i*36;out.set(state.x.subarray(i*3,i*3+3),k);out[k+3]=state.mode[i];
  out.set(state.v.subarray(i*3,i*3+3),k+4);out[k+7]=state.mass[i];
  for(let c=0;c<3;c++)for(let r=0;r<3;r++){out[k+8+4*c+r]=state.C[i*9+3*r+c];out[k+20+4*c+r]=state.F[i*9+3*r+c];}
  out[k+32]=state.volume[i];out[k+34]=state.ids[i];
 }
 return out;
}
export function unpackState(values,prior,steps,dt){
 const n=prior.ids.length,state={...prior,ids:prior.ids.slice(),mode:prior.mode.slice(),mass:prior.mass.slice(),volume:prior.volume.slice(),x:new Float64Array(n*3),v:new Float64Array(n*3),C:new Float64Array(n*9),F:new Float64Array(n*9),time:prior.time+steps*dt,steps:prior.steps+steps};
 for(let i=0;i<n;i++){
  const k=i*36;
  if(values[k+34]!==prior.ids[i]||values[k+7]!==Math.fround(prior.mass[i])||values[k+32]!==Math.fround(prior.volume[i]))throw Error('GPU particle identity/mass/volume changed');
  state.ids[i]=values[k+34];state.mass[i]=values[k+7];state.volume[i]=values[k+32];
  state.x.set(values.subarray(k,k+3),i*3);state.v.set(values.subarray(k+4,k+7),i*3);state.mode[i]=values[k+3];
  for(let c=0;c<3;c++)for(let r=0;r<3;r++){state.C[i*9+3*r+c]=values[k+8+4*c+r];state.F[i*9+3*r+c]=values[k+20+4*c+r];}
 }
 return state;
}
export async function createSimulator(device,config,initial){
 if(config.viscosity)throw Error('Viscosity is not part of the verified CPU contract');
 config=normalizeConfig(config);initial=makeState(initial);
 const n=config.N,count=initial.ids.length;
 if(!Number.isInteger(n)||n<8||n>40||count<1||count>8192)throw Error('Bounded prototype: N8..40, 1..8192 particles');
 if(!Number.isFinite(config.dt)||config.dt<=0||config.dt>1/120)throw Error('Invalid timestep');
 function validateState(state){
  if(state.ids.length!==count)throw Error('Particle count must remain fixed');
  for(let i=0;i<count;i++){
   if(state.ids[i]>16777215)throw Error('Particle ID exceeds exact float representation');
   for(let a=0;a<3;a++)if(state.x[3*i+a]<.5||state.x[3*i+a]>=n-1.5)throw Error('Initial particle outside grid stencil');
   if(state.mode[i]&&!(determinant(state.F,i*9)>.05&&determinant(state.F,i*9)<20))throw Error('Initial solid deformation outside bounded range');
  }
 }
 validateState(initial);
 const owned=[],make=(bytes,usage,label)=>{const data=typeof bytes==='number'?null:bytes,b=device.createBuffer({size:Math.max(16,data?data.byteLength:bytes),usage,label});owned.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
 try{
 let disposed=false,busy=false,totalSteps=0;
 const particleBuffer=make(packState(initial),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,'Matter particles');
 const grid=make(n**3*20,GPUBufferUsage.STORAGE,'Matter fixed-point grid');
 const velocities=make(n**3*16,GPUBufferUsage.STORAGE,'Matter grid velocities');
 const boxList=[...(config.colliders??[]),...(config.anchors??[])],boxData=new Float32Array(Math.max(8,boxList.length*8));
 boxList.forEach((b,i)=>{boxData.set(b.min,i*8);boxData.set(b.max,i*8+4);});
 const boxes=make(boxData,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'Visible colliders and named anchors');
 const faults=make(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,'Matter fault flags');
 const faultReadback=make(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,'Matter bounded-step fault readback');
 const data=new ArrayBuffer(64),ui=new Uint32Array(data),f=new Float32Array(data);
 ui.set([n,count,config.colliders?.length??0,config.anchors?.length??0]);
 f.set([config.dt,config.bulk??120,config.gamma??5,config.mu??80,config.lambda??120,config.boundary??2,1e6,config.viscosity??0],4);f.set(config.gravity??[0,-9.81,0],12);
 const params=make(data,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,'Matter parameters');
 const layout=device.createBindGroupLayout({entries:[...Array.from({length:5},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===3?'read-only-storage':'storage'}})),{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}]});
 const group=device.createBindGroup({layout,entries:[particleBuffer,grid,velocities,boxes,faults,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]}),pipelines={};
 try{for(const [name,code] of Object.entries(shaderSources)){
  const module=device.createShaderModule({code,label:'Matter '+name}),info=await module.getCompilationInfo();
  const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(name+': '+errors.map(m=>m.message).join('; '));
  pipelines[name]=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'main'}});
 }}catch(e){owned.forEach(b=>b.destroy());throw e;}
 async function read(){
  if(disposed)throw Error('Simulator disposed');
  const size=count*144+16,b=make(size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,'Matter readback');
  const e=device.createCommandEncoder();e.copyBufferToBuffer(particleBuffer,0,b,0,count*144);e.copyBufferToBuffer(faults,0,b,count*144,16);device.queue.submit([e.finish()]);
  try{await b.mapAsync(GPUMapMode.READ);const copy=b.getMappedRange().slice(0),flag=new Uint32Array(copy,count*144,4)[0];
   if(flag)throw Error('Matter integration stopped: fault mask '+flag);
   return unpackState(new Float32Array(copy,0,count*36),initial,totalSteps,config.dt);
  }finally{b.unmap();b.destroy();owned.splice(owned.indexOf(b),1);}
 }
 return {particleBuffer,count,shaderSources,
  async advance(steps,{signal}={}){
   if(disposed||busy||!Number.isInteger(steps)||steps<1||steps>512)throw Error('Invalid/busy/disposed bounded step');
   busy=true;const start=performance.now();
   try{for(let offset=0;offset<steps;offset+=4){
    if(signal?.aborted||performance.now()-start>12000)throw Error('Matter run cancelled or time limit reached');
    const chunk=Math.min(4,steps-offset),e=device.createCommandEncoder();
    for(let s=0;s<chunk;s++){for(const name of ['clear','mass','stress','grid','gather']){
     const pass=e.beginComputePass();pass.setPipeline(pipelines[name]);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil((name==='clear'||name==='grid'?n**3:count)/64));pass.end();
    }}e.copyBufferToBuffer(faults,0,faultReadback,0,16);device.queue.submit([e.finish()]);
    await faultReadback.mapAsync(GPUMapMode.READ);const fault=new Uint32Array(faultReadback.getMappedRange())[0];faultReadback.unmap();
    totalSteps+=chunk;if(fault)throw Error('Matter integration stopped: fault mask '+fault);
    await new Promise(resolve=>setTimeout(resolve,0));
   }return await read();}finally{busy=false;}
  },
  read,
  replaceState(input){
   if(disposed||busy)throw Error('Cannot edit a running or disposed simulator');
   const next=makeState(input);validateState(next);
   for(let i=0;i<count;i++)if(next.ids[i]!==initial.ids[i]||Math.fround(next.mass[i])!==Math.fround(initial.mass[i])||Math.fround(next.volume[i])!==Math.fround(initial.volume[i]))throw Error('Material edits must retain identity, mass and volume');
   if(next.steps!==initial.steps+totalSteps)throw Error('Material edit must use the latest snapshot');
   device.queue.writeBuffer(particleBuffer,0,packState(next));
   initial=next;totalSteps=0;
  },
  dispose(){if(!disposed){disposed=true;owned.forEach(b=>b.destroy());owned.length=0;}}
 };
 }catch(error){owned.forEach(buffer=>buffer.destroy());throw error;}
}

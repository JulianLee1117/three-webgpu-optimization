// Two ordinary WGSL reductions. No floating-point atomics or full image readback.
export class GpuImageLoss {
  constructor(device, reference, candidate, side=128) {
    if(side!==128)throw Error('The screen only supports 128x128 loss images.');
    this.device=device;
    this.partials=device.createBuffer({size:256*4,usage:GPUBufferUsage.STORAGE});
    this.output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.staging=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const first=device.createShaderModule({code:`
@group(0) @binding(0) var reference: texture_2d<f32>;
@group(0) @binding(1) var candidate: texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> partials: array<f32>;
var<workgroup> sums: array<f32,64>;
@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u) {
  let p=vec2i(group.xy*8u+local.xy);let i=local.y*8u+local.x;
  let d=textureLoad(reference,p,0).rgb-textureLoad(candidate,p,0).rgb;
  sums[i]=dot(d,d);workgroupBarrier();
  for(var stride=32u;stride>0u;stride=stride/2u){
    if(i<stride){sums[i]=sums[i]+sums[i+stride];}workgroupBarrier();
  }
  if(i==0u){partials[group.y*16u+group.x]=sums[0];}
}`});
    const second=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> partials:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<f32>;
var<workgroup> sums:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i:u32){
  sums[i]=partials[i];workgroupBarrier();
  for(var stride=128u;stride>0u;stride=stride/2u){
    if(i<stride){sums[i]=sums[i]+sums[i+stride];}workgroupBarrier();
  }
  if(i==0u){result[0]=sums[0]/(128.0*128.0*3.0);result[1]=0.0;result[2]=0.0;result[3]=0.0;}
}`});
    this.first=device.createComputePipeline({layout:'auto',compute:{module:first,entryPoint:'main'}});
    this.second=device.createComputePipeline({layout:'auto',compute:{module:second,entryPoint:'main'}});
    this.firstBindings=device.createBindGroup({layout:this.first.getBindGroupLayout(0),entries:[
      {binding:0,resource:reference.createView()},{binding:1,resource:candidate.createView()},{binding:2,resource:{buffer:this.partials}}
    ]});
    this.secondBindings=device.createBindGroup({layout:this.second.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.partials}},{binding:1,resource:{buffer:this.output}}
    ]});
    this.readbacks=0;
  }
  async evaluate(){
    const encoder=this.device.createCommandEncoder();
    let pass=encoder.beginComputePass();pass.setPipeline(this.first);pass.setBindGroup(0,this.firstBindings);pass.dispatchWorkgroups(16,16);pass.end();
    pass=encoder.beginComputePass();pass.setPipeline(this.second);pass.setBindGroup(0,this.secondBindings);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(this.output,0,this.staging,0,16);this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const value=new Float32Array(this.staging.getMappedRange())[0];this.staging.unmap();this.readbacks++;
    if(!Number.isFinite(value)||value<0)throw Error('Invalid GPU image loss');
    return value;
  }
  dispose(){this.partials.destroy();this.output.destroy();this.staging.destroy();}
}

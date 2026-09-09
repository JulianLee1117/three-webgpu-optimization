import * as THREE from 'three/webgpu';
import {positionLocal, uniform} from 'three/tsl';
import {numericProgram} from './programs.js';
import {LIMITS} from './fixtures.mjs';

/** Declared before the first rendering run. This tests silhouette geometry,
 * not lighting, normals, temporal continuity, or hidden/occluded vertices. */
export const RENDER_PARITY_LIMITS = Object.freeze({width:256,height:192,sections:192,sides:10,
  times:Object.freeze([0,.25,.59375]),minimumPixelAllowance:2,relativePixelAllowance:.01,boundaryRadius:1});

function encodeMask(mask) {
  const packed=new Uint8Array(Math.ceil(mask.length/8));
  for(let i=0;i<mask.length;i++)if(mask[i])packed[i>>3]|=1<<(i&7);
  let binary='';for(const byte of packed)binary+=String.fromCharCode(byte);
  return btoa(binary);
}

/** A pixel is on a silhouette boundary if any of its eight neighbors has a
 * different mask value. Pixels outside the image are background. */
function boundary(mask,width,height) {
  const result=new Uint8Array(mask.length);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const at=y*width+x;
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++) {
      const xx=x+dx,yy=y+dy,value=xx<0||xx>=width||yy<0||yy>=height?0:mask[yy*width+xx];
      if(value!==mask[at])result[at]=1;
    }
  }
  return result;
}

/** CPU-only helper, also usable by the retained-evidence verifier. */
export function compareRenderMasks(native,cpu,width=RENDER_PARITY_LIMITS.width,height=RENDER_PARITY_LIMITS.height) {
  if(native.length!==width*height||cpu.length!==width*height)throw Error('Invalid render mask size');
  if([...native,...cpu].some(x=>x!==0&&x!==1))throw Error('Render masks must be binary');
  const a=boundary(native,width,height),b=boundary(cpu,width,height);
  let nativeForeground=0,cpuForeground=0,differingPixels=0,interiorMismatches=0;
  const close=(edges,x,y)=>{
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++) {
      const xx=x+dx,yy=y+dy;if(xx>=0&&xx<width&&yy>=0&&yy<height&&edges[yy*width+xx])return true;
    }
    return false;
  };
  for(let i=0;i<native.length;i++) {
    nativeForeground+=native[i];cpuForeground+=cpu[i];
    if(native[i]!==cpu[i]) {
      differingPixels++;const x=i%width,y=Math.floor(i/width);
      if(!close(a,x,y)||!close(b,x,y))interiorMismatches++;
    }
  }
  const pixelAllowance=Math.max(RENDER_PARITY_LIMITS.minimumPixelAllowance,
    Math.ceil(RENDER_PARITY_LIMITS.relativePixelAllowance*Math.max(nativeForeground,cpuForeground)));
  return {nativeForeground,cpuForeground,differingPixels,interiorMismatches,pixelAllowance,
    passed:nativeForeground>0&&cpuForeground>0&&differingPixels<=pixelAllowance&&interiorMismatches===0};
}

function classify(bytes) {
  const count=RENDER_PARITY_LIMITS.width*RENDER_PARITY_LIMITS.height;
  if(!(bytes instanceof Uint8Array)||bytes.length!==count*4)throw Error('Expected contiguous RGBA8 readback');
  const mask=new Uint8Array(count);let black=0,white=0,invalid=0;
  for(let i=0;i<count;i++) {
    const at=i*4,r=bytes[at],g=bytes[at+1],b=bytes[at+2],a=bytes[at+3];
    if(r===0&&g===0&&b===0&&a===255)black++;
    else if(r===255&&g===255&&b===255&&a===255){white++;mask[i]=1;}
    else invalid++;
  }
  return {mask,outputBytes:{length:bytes.length,black,white,invalid,valid:invalid===0}};
}

/** Render an existing authored graph through material.positionNode and compare
 * it with the independent numericProgram baked into the same Float32 rest tube.
 * Caller must serialize use of evaluator.renderer. This owns/disposes its five
 * resources and restores renderer state; it never destroys the caller device.
 * This loads the supplied parameters into the caller's live parameter buffer;
 * its final contents are read back and checked. Parameters are explicitly
 * rounded to Float32 in both lanes. Masks use packed
 * LSB-first bits in readback row order, exactly width*height pixels.
 */
export async function validateRenderedProgram({fixture,evaluator,parameters=fixture.initial}) {
  const limits=RENDER_PARITY_LIMITS,{renderer}=evaluator;
  if(evaluator.status.busy||evaluator.status.disposed)throw Error('Evaluator unavailable for render parity');
  if(parameters.length!==8||Array.from(parameters).some(v=>!Number.isFinite(v)))throw Error('Invalid render parity parameters');
  const weights=Array.from(parameters,Math.fround),rest=[],indices=[];
  for(let i=0;i<=limits.sections;i++)for(let j=0;j<=limits.sides;j++) {
    const angle=j/limits.sides*2*Math.PI;rest.push(i/limits.sections,Math.cos(angle)*LIMITS.crossSectionRadius,Math.sin(angle)*LIMITS.crossSectionRadius);
  }
  for(let i=0;i<limits.sections;i++)for(let j=0;j<limits.sides;j++) {
    const a=i*(limits.sides+1)+j,b=a+limits.sides+1;indices.push(a,b,a+1,a+1,b,b+1);
  }
  const restArray=new Float32Array(rest),geometry=new THREE.BufferGeometry(),baked=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(restArray,3));geometry.setIndex(indices);
  baked.setAttribute('position',new THREE.BufferAttribute(new Float32Array(restArray.length),3));baked.setIndex(indices);
  const nativeMaterial=new THREE.MeshBasicNodeMaterial({color:0xffffff,side:THREE.DoubleSide,toneMapped:false});
  const cpuMaterial=new THREE.MeshBasicNodeMaterial({color:0xffffff,side:THREE.DoubleSide,toneMapped:false});
  const clock=uniform(0);nativeMaterial.positionNode=evaluator.bind(positionLocal,clock);
  const nativeMesh=new THREE.Mesh(geometry,nativeMaterial),cpuMesh=new THREE.Mesh(baked,cpuMaterial);
  nativeMesh.frustumCulled=cpuMesh.frustumCulled=false;
  const scene=new THREE.Scene();scene.add(nativeMesh,cpuMesh);
  const camera=new THREE.OrthographicCamera(-2.6,2.6,1.95,-1.95,.1,40);
  const center=new THREE.Vector3(1.4,.7,0);camera.position.copy(center).add(new THREE.Vector3(4,3,7));camera.lookAt(center);
  const target=new THREE.RenderTarget(limits.width,limits.height,{samples:0,type:THREE.UnsignedByteType,format:THREE.RGBAFormat,
    minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,generateMipmaps:false,depthBuffer:true,stencilBuffer:false,colorSpace:THREE.LinearSRGBColorSpace});
  const previous={target:renderer.getRenderTarget(),clear:renderer.getClearColor(new THREE.Color()).clone(),alpha:renderer.getClearAlpha(),
    toneMapping:renderer.toneMapping,autoClear:renderer.autoClear};
  const result={kind:'constraint-editing-render-parity-v1',fixture:fixture.kind,parameters:weights,limits,
    maskEncoding:'base64-packed-lsb-first-row-major',restVertices:restArray.length/3,triangles:indices.length/3,
    camera:{position:camera.position.toArray(),lookAt:center.toArray(),orthographic:[-2.6,2.6,1.95,-1.95,.1,40]},cases:[],passed:false};
  try {
    evaluator.load(weights);
    renderer.setRenderTarget(target);renderer.setClearColor(0,1);renderer.toneMapping=THREE.NoToneMapping;renderer.autoClear=true;
    for(const time of limits.times) {
      clock.value=time;const positions=baked.attributes.position.array;
      for(let i=0;i<restArray.length;i+=3)positions.set(numericProgram(fixture.kind,Array.from(restArray.subarray(i,i+3)),Math.fround(time),weights),i);
      baked.attributes.position.needsUpdate=true;
      nativeMesh.visible=true;cpuMesh.visible=false;renderer.render(scene,camera);
      const native=classify(await renderer.readRenderTargetPixelsAsync(target,0,0,limits.width,limits.height));
      nativeMesh.visible=false;cpuMesh.visible=true;renderer.render(scene,camera);
      const cpu=classify(await renderer.readRenderTargetPixelsAsync(target,0,0,limits.width,limits.height));
      evaluator.check();const comparison=compareRenderMasks(native.mask,cpu.mask);
      result.cases.push({time,comparison,outputBytes:{native:native.outputBytes,cpu:cpu.outputBytes},
        masks:{native:encodeMask(native.mask),cpu:encodeMask(cpu.mask)},
        passed:comparison.passed&&native.outputBytes.valid&&cpu.outputBytes.valid});
    }
    result.uploadedParameters=await evaluator.readParameters();
    result.uploadExact=result.uploadedParameters.length===weights.length&&result.uploadedParameters.every((v,i)=>v===weights[i]);
    result.passed=result.cases.every(item=>item.passed)&&result.uploadExact;return result;
  } finally {
    renderer.setRenderTarget(previous.target);renderer.setClearColor(previous.clear,previous.alpha);
    renderer.toneMapping=previous.toneMapping;renderer.autoClear=previous.autoClear;
    target.dispose();nativeMaterial.dispose();cpuMaterial.dispose();geometry.dispose();baked.dispose();
  }
}

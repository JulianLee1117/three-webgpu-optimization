import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector3} from 'three';
import {StorageBufferAttribute} from 'three/webgpu';
import {uniform, vec3, float, sin, cos, mix, Fn, storage, instanceIndex} from 'three/tsl';
import {rebind, rebindWithStats, REBIND_LIMITS} from './rebind.js';
import {evaluateGradients} from '../material-mips/autograd.js';

const canonical = input => {
  let n = input;
  while ((n.self?.isNode && n.self !== n) || n.constructor.type === 'VarNode') n = n.self?.isNode && n.self !== n ? n.self : n.node;
  return n;
};
const near = (a,b,tolerance=1e-10) => assert.ok(Number.isFinite(a) && Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(b)), `${a} != ${b}`);
const scalar = (node, variables, point) => evaluateGradients(node,variables,new Map(variables.map((n,i)=>[n,point[i]]))).value;

test('nonlinear animation rebinds onto new point/time/parameters without rewriting its formula', () => {
  const p=uniform(new Vector3()),time=uniform(0),amplitude=uniform(0),phase=uniform(0);
  const angle=p.y.mul(.7).add(time.mul(.4)).add(phase);
  const output=vec3(p.x.mul(cos(angle)).sub(p.z.mul(sin(angle))),
    p.y.add(sin(p.x.mul(2.1).add(time)).mul(amplitude)),p.x.mul(sin(angle)).add(p.z.mul(cos(angle))));
  const newP=uniform(new Vector3()),newTime=uniform(0),newAmp=uniform(0),newPhase=uniform(0);
  const {root,stats}=rebindWithStats(output,new Map([[p,newP],[time,newTime],[amplitude,newAmp],[phase,newPhase]]));
  assert.notEqual(canonical(root),canonical(output));assert.equal(stats.usedBindings,4);
  const variables=[newP,newTime,newAmp,newPhase];
  for(let i=0;i<32;i++){
    const point=[[-.7+i*.03,.3-i*.009,.2+i*.011],i*.17,.2+i*.01,-.2+i*.013];
    const [[x,y,z],t,a,phase]=point,angle=.7*y+.4*t+phase;
    const expected=[x*Math.cos(angle)-z*Math.sin(angle),y+Math.sin(2.1*x+t)*a,x*Math.sin(angle)+z*Math.cos(angle)];
    for(const [j,axis] of [...'xyz'].entries()) near(scalar(root[axis],variables,point),expected[j]);
  }
});

test('shared subexpressions remain shared and the original source is immutable', () => {
  const x=uniform(0),y=uniform(0),shared=sin(x.mul(y)),source=shared.mul(shared).add(shared),z=uniform(0),w=uniform(0);
  const original=canonical(source),originalA=original.aNode,originalB=original.bNode;
  const root=canonical(rebind(source,new Map([[x,z],[y,w]])));
  assert.equal(canonical(root.aNode).aNode,root.bNode);
  assert.equal(canonical(root.aNode).bNode,root.bNode);
  assert.notEqual(root.bNode,canonical(shared));
  assert.equal(original.aNode,originalA);assert.equal(original.bNode,originalB);
  near(scalar(source,[x,y],[.4,.7]),Math.sin(.28)**2+Math.sin(.28));
  near(scalar(root,[z,w],[.9,.2]),Math.sin(.18)**2+Math.sin(.18));
});

test('min/max preserve exact constructor arity and vector constants are copied', () => {
  const x=uniform(0),z=uniform(0),constant=vec3(.2,.4,.6);
  const source=mix(constant,vec3(x),float(.3)).x.add(x.max(-.2).min(.6));
  const bound=rebind(source,new Map([[x,z]]));
  for(const point of [-.8,.1,.9])near(scalar(bound,[z],[point]),.2*.7+point*.3+Math.min(Math.max(point,-.2),.6));
  const original=canonical(constant),copy=canonical(rebind(constant,new Map()));
  assert.equal(original.constructor.type,'ConstNode');
  assert.notEqual(copy.value,original.value);assert.deepEqual(copy.value.toArray(),original.value.toArray());
  copy.value.x=9;near(original.value.x,.2);
});

test('exact storage swizzles are replacement boundaries, unbound storage rejects', () => {
  const p=uniform(new Vector3()),time=uniform(0),samples=storage(new StorageBufferAttribute(new Float32Array(8),4),'vec4',2);
  const sampleP=samples.element(instanceIndex).xyz,sampleTime=samples.element(instanceIndex).w;
  const source=p.mul(sin(time));
  const root=rebind(source,new Map([[p,sampleP],[time,sampleTime]]));
  const values=new Map([[sampleP,[2,3,4]],[sampleTime,.7]]);
  for(const [j,axis] of [...'xyz'].entries())near(evaluateGradients(root[axis],[sampleP,sampleTime],values).value,[2,3,4][j]*Math.sin(.7));
  assert.throws(()=>rebind(sampleP,new Map()),/Unsupported|unbound/);
});

test('unbound, effectful, function, unsupported and malformed source graphs fail closed', () => {
  const x=uniform(0),y=uniform(0),z=uniform(0);
  assert.throws(()=>rebind(x.add(y),new Map([[x,z]])),/unbound/);
  assert.throws(()=>rebind(x.toVar(),new Map([[x,z]])),/VarNode/);
  assert.throws(()=>rebind(x.toConst(),new Map([[x,z]])),/VarNode/);
  assert.throws(()=>rebind(x.add(1).before(y),new Map([[x,z],[y,z]])),/side effects/);
  assert.throws(()=>rebind(Fn(()=>x.add(1))(),new Map([[x,z]])),/Unsupported/);
  assert.throws(()=>rebind(x.log(),new Map([[x,z]])),/Unsupported/);
  assert.throws(()=>rebind(x.greaterThan(y).select(x,y),new Map([[x,z],[y,z]])),/Unsupported/);
  assert.throws(()=>rebind(x,new Map([[x,z.toVar()]])),/VarNode/);
  assert.throws(()=>rebind(x,new Map([[x,vec3(1)]])),/width/);
  assert.throws(()=>rebind(x,new Map([[x,Fn(()=>z)()]])),/replacement/);
  assert.throws(()=>rebind(x,new Map([[x,z],[x.toVarIntent(),z]])),/Duplicate/);
  assert.throws(()=>rebind(float(Infinity),new Map()),/finite/);
  assert.throws(()=>rebind(x.pow(y),new Map([[x,z],[y,z]])),/literal/);
  assert.throws(()=>rebind(x,new Map([[x,uniform(1,'int')]])),/floating/);
  const integers=storage(new StorageBufferAttribute(new Uint32Array(4),4),'uvec4',1).element(instanceIndex).x;
  assert.throws(()=>rebind(x,new Map([[x,integers]])),/floating/);
});

test('cycles, depth and node-count limits are enforced; unused mappings are reported', () => {
  const x=uniform(0),z=uniform(0),cyclic=x.add(1),node=canonical(cyclic);node.aNode=node;
  assert.throws(()=>rebind(cyclic,new Map([[x,z]])),/Cyclic/);
  let deep=x;for(let i=0;i<129;i++)deep=sin(deep);
  assert.throws(()=>rebind(deep,new Map([[x,z]])),/depth/);
  let layer=Array.from({length:2200},(_,i)=>sin(x.add(i*.01)));
  while(layer.length>1){const next=[];for(let i=0;i<layer.length;i+=2)next.push(layer[i+1]?layer[i].add(layer[i+1]):layer[i]);layer=next;}
  assert.throws(()=>rebind(layer[0],new Map([[x,z]])),/4096/);
  const result=rebindWithStats(float(2),new Map([[x,z]]));assert.equal(result.stats.unusedBindings,1);
  assert.equal(REBIND_LIMITS.maxNodes,4096);
});

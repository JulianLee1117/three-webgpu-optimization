import {float, vec2, vec3, vec4, sin, cos, dFdx, dFdy} from 'three/tsl';
import {gradients, evaluateGradients} from '../material-mips/autograd.js';

/** Native Three r185 finite Fourier expansion, followed by a joint pixel/shutter
 * box integral. Products are expanded BEFORE filtering: correlated factors are
 * not independent random variables. The mathematics is established Fourier
 * filtering, not a new antialiasing algorithm.
 *
 * Color grammar: finite numeric constants, + - *, division by a nonzero numeric
 * constant, mix, vec2/3/4 construction, swizzles, integer powers 0..4, and sin/cos
 * of pure scalar phases. Each phase may use explicitly designated native inputs
 * and the pure AD subset. Uniform parameters are held constant during exposure.
 * Explicit variables, arbitrary Fn, texture reads and unsupported operations
 * reject. Source graphs are never mutated. This is not a general shader compiler.
 *
 * Exact real-arithmetic integration requires constant amplitudes and phases
 * affine in screen x, screen y and time over the box. dFdx/dFdy and local d/dt
 * only approximate nonlinear phases and perspective footprints. Material color
 * is integrated before any subsequent lighting/tone mapping. No geometry,
 * visibility, texture or camera-motion shutter integration is implied.
 */
export const LIMITS = Object.freeze({nodes:2048, depth:128, terms:128, phaseFactors:16});
const type = n => n?.constructor?.type ?? n?.constructor?.name;
const pack = a => a.length===1?a[0]:[null,null,vec2,vec3,vec4][a.length](...a);
const width = t => ({float:1,vec2:2,vec3:3,vec4:4})[t];
function canonical(input) {
  let n=input;const seen=new Set();
  for(;;){
    if(!n?.isNode||seen.has(n))throw Error('Expected an acyclic native TSL expression.');
    seen.add(n);if(n._beforeNodes?.length)throw Error('before() side effects are unsupported.');
    if(n.self?.isNode&&n.self!==n){n=n.self;continue;}
    if(type(n)!=='VarNode')return n;
    if(n.intent!==true||n.name!==null||n.readOnly===true)throw Error('Explicit or mutable variables are unsupported.');
    n=n.node;
  }
}
const keyOf = factors => [...factors].filter(([,v])=>v!==0).sort((a,b)=>a[0]-b[0]).map(([i,v])=>`${i}:${v}`).join(',');
const factorsOf = key => new Map(key?key.split(',').map(s=>s.split(':').map(Number)):[]);
const scalar = value => new Map(value===0?[]:[['',[value,0]]]);
function put(out,key,re,im){
  const old=out.get(key)??[0,0],next=[old[0]+re,old[1]+im];
  if(next.some(x=>!Number.isFinite(x)||Math.abs(x)>1e12))throw Error('Nonfinite or excessive Fourier coefficient.');
  if(next[0]===0&&next[1]===0)out.delete(key);else out.set(key,next);
  if(out.size>LIMITS.terms)throw Error(`Fourier expansion exceeds ${LIMITS.terms} terms per channel.`);
}
function add(a,b,sign=1){const out=new Map(a);for(const[k,[r,i]]of b)put(out,k,r*sign,i*sign);return out;}
function multiply(a,b){
  const out=new Map();
  for(const[ka,[ar,ai]]of a)for(const[kb,[br,bi]]of b){
    const factors=factorsOf(ka);for(const[i,v]of factorsOf(kb))factors.set(i,(factors.get(i)??0)+v);
    if([...factors.values()].reduce((a,v)=>a+Math.abs(v),0)>LIMITS.phaseFactors)throw Error('Harmonic degree exceeds bounded compiler limit.');
    put(out,keyOf(factors),ar*br-ai*bi,ar*bi+ai*br);
  }return out;
}
function numericConstant(a){if(a.size===0)return 0;if(a.size!==1||!a.has('')||a.get('')[1]!==0)throw Error('Expected a numeric constant expression.');return a.get('')[0];}
function broadcast(a,b,fn){
  const n=Math.max(a.length,b.length);if(a.length!==b.length&&a.length!==1&&b.length!==1)throw Error('Incompatible vector widths.');
  return Array.from({length:n},(_,i)=>fn(a[a.length===1?0:i],b[b.length===1?0:i]));
}

function expand(root){
  const cache=new Map(),active=new Set(),phaseIds=new Map(),phases=[];let visited=0,maxDepth=0;
  function visit(input,depth=0){
    const n=canonical(input);maxDepth=Math.max(maxDepth,depth);
    if(depth>LIMITS.depth)throw Error('Expression depth exceeds compiler limit.');
    if(active.has(n))throw Error('Cyclic expression.');if(cache.has(n))return cache.get(n);
    if(++visited>LIMITS.nodes)throw Error('Native node budget exceeded.');active.add(n);
    const child=x=>visit(x,depth+1);let result;
    switch(type(n)){
      case 'ConstNode':{
        const count=width(n.nodeType)??(n.nodeType==null?(typeof n.value==='number'?1:[2,3,4].find(k=>n.value?.[`isVector${k}`])):null);if(!count)throw Error('Only float/vector constants are supported.');
        const values=count===1?[n.value]:[...'xyzw'.slice(0,count)].map(a=>n.value?.[a]);
        if(values.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error('Constants must be finite.');
        result=values.map(scalar);break;
      }
      case 'OperatorNode':{
        const a=child(n.aNode),b=child(n.bNode);
        if(n.op==='+')result=broadcast(a,b,add);
        else if(n.op==='-')result=broadcast(a,b,(x,y)=>add(x,y,-1));
        else if(n.op==='*')result=broadcast(a,b,multiply);
        else if(n.op==='/')result=broadcast(a,b,(x,y)=>{const d=numericConstant(y);if(d===0)throw Error('Division by zero.');return multiply(x,scalar(1/d));});
        else throw Error(`Unsupported color operator ${n.op}.`);break;
      }
      case 'MathNode':
        if(['sin','cos'].includes(n.method)&&!n.bNode&&!n.cNode){
          const phase=canonical(n.aNode);let id=phaseIds.get(phase);
          if(id===undefined){id=phases.length;phaseIds.set(phase,id);phases.push(n.aNode);}
          result=[new Map(n.method==='cos'?[[`${id}:1`,[.5,0]],[`${id}:-1`,[.5,0]]]:[[`${id}:1`,[0,-.5]],[`${id}:-1`,[0,.5]]])];
        }else if(n.method==='negate')result=child(n.aNode).map(x=>multiply(x,scalar(-1)));
        else if(n.method==='oneMinus')result=child(n.aNode).map(x=>add(scalar(1),x,-1));
        else if(n.method==='mix'){
          const a=child(n.aNode),b=child(n.bNode),t=child(n.cNode);
          result=broadcast(broadcast(a,b,(x,y)=>add(y,x,-1)),t,multiply);
          result=broadcast(a,result,add);
        }else if(n.method==='pow'){
          const exponent=child(n.bNode);if(exponent.length!==1)throw Error('Scalar exponent required.');
          const k=numericConstant(exponent[0]);if(!Number.isInteger(k)||k<0||k>4)throw Error('Color power must be an integer from zero to four.');
          result=child(n.aNode).map(x=>{let out=scalar(1);for(let i=0;i<k;i++)out=multiply(out,x);return out;});
        }else throw Error(`Unsupported color operation ${n.method}; expected finite sin/cos algebra.`);
        break;
      case 'JoinNode':result=n.nodes.flatMap(child);if(result.length<2||result.length>4||(n.nodeType!=null&&width(n.nodeType)!==result.length))throw Error('Invalid vector join.');break;
      case 'SplitNode':{
        const a=child(n.node);if(!/^[xyzw]{1,4}$/.test(n.components))throw Error('Unsupported swizzle.');
        result=[...n.components].map(k=>{const i=a.length===1?0:'xyzw'.indexOf(k);if(i>=a.length)throw Error('Swizzle accesses an absent component.');return a[i];});break;
      }
      case 'ConvertNode':{
        const a=child(n.node),count=width(n.convertTo);if(!count)throw Error('Only float/vector conversion is supported.');
        result=Array.from({length:count},(_,i)=>a.length===1?a[0]:i<a.length?a[i]:scalar(i===3?1:0));break;
      }
      default:throw Error(`Unsupported color node ${type(n)}. Spatial inputs belong inside sin/cos phases; amplitudes must be constant.`);
    }
    active.delete(n);cache.set(n,result);return result;
  }
  const channels=visit(root);
  // Pair conjugate harmonics to evaluate each real oscillation only once.
  const keys=new Set(channels.flatMap(c=>[...c.keys()])),terms=[];
  for(const key of keys){
    const factors=factorsOf(key),first=[...factors.values()][0];if(first<0)continue;
    const inverse=keyOf(new Map([...factors].map(([i,v])=>[i,-v])));
    const coefficients=channels.map(c=>{
      const a=c.get(key)??[0,0];if(key==='')return a;
      const b=c.get(inverse)??[0,0];return[a[0]+b[0],a[1]-b[1]];
    });
    let phase=float(0);for(const[i,v]of factors)phase=phase.add(phases[i].mul(v));
    terms.push({phase,coefficients,constant:key==='',key});
  }
  return {terms,phases,channels:channels.length,stats:{nativeColorNodes:visited,maxDepth,phaseAtoms:phases.length,
    expandedTerms:channels.map(c=>c.size),realHarmonics:terms.filter(t=>!t.constant).length}};
}

// Safe at the origin, including GPUs which evaluate both select operands.
function sincNode(x){const a=x.abs(),safe=a.max(1e-4);return a.lessThan(1e-3).select(float(1).sub(x.mul(x).div(6)),sin(safe).div(safe));}
export function sinc(x){return Math.abs(x)<1e-5?1-x*x/6:Math.sin(x)/x;}

function validateShutter(shutter){
  if(typeof shutter==='number'){
    if(!Number.isFinite(shutter)||shutter<0)throw Error('Shutter duration must be finite and nonnegative.');
    return;
  }
  const node=canonical(shutter),kind=type(node);
  if(!['ConstNode','UniformNode'].includes(kind)||![null,'float','f32'].includes(node.nodeType??null))
    throw Error('Shutter must be a scalar float constant or uniform, or a number.');
  if(typeof node.value!=='number'||!Number.isFinite(node.value)||node.value<0)
    throw Error('Shutter duration must be finite and nonnegative.');
}

export function compileFootprint(root,{time=null,shutter=float(0),inputs=[]}={}){
  if(!Array.isArray(inputs))throw Error('inputs must be an array of exact native nodes.');
  validateShutter(shutter);
  const declaration=gradients(float(0),time===null?[]:[time],{constants:inputs});
  if(time!==null&&declaration.stats.scalarParameters!==1)throw Error('The time input must be scalar.');
  const expansion=expand(root),constants=inputs;
  // Validate EVERY phase, even phases removed by exact algebraic cancellation.
  for(const phase of expansion.phases)gradients(phase,time?[time]:[],{constants});
  const output=Array.from({length:expansion.channels},()=>float(0));
  for(const term of expansion.terms){
    if(term.constant){term.coefficients.forEach(([r,i],j)=>{if(i!==0)throw Error('Non-real constant term.');output[j]=output[j].add(r);});continue;}
    const phase=term.phase;
    const speed=time?gradients(phase,[time],{constants}).gradients[0]:float(0);
    const weight=sincNode(dFdx(phase).mul(.5)).mul(sincNode(dFdy(phase).mul(.5))).mul(sincNode(speed.mul(shutter).mul(.5)));
    term.coefficients.forEach(([r,i],j)=>{
      let wave=float(0);if(r!==0)wave=wave.add(cos(phase).mul(r));if(i!==0)wave=wave.sub(sin(phase).mul(i));
      if(r!==0||i!==0)output[j]=output[j].add(wave.mul(weight));
    });
  }
  return {node:pack(output),stats:Object.freeze({...expansion.stats,method:'joint finite Fourier box integration',temporal:!!time})};
}

/** CPU evaluation of the compiled finite expansion, for independent oracle
 * comparisons. dx/dy contain input changes across one pixel, in inputs order.
 * Empty dx/dy mean zero change for all inputs; otherwise each entry must match
 * its designated scalar/vector width exactly. Shutter is finite, nonnegative.
 * Numeric phase derivatives come from native AD; no renderer is involved. */
export function evaluateFootprint(root,{values,inputs=[],time=null,shutter=0,dx=[],dy=[]}){
  if(!Array.isArray(inputs))throw Error('inputs must be an array of exact native nodes.');
  if(!(values instanceof Map))throw Error('values must be a Map of exact native inputs to numeric values.');
  if(!Number.isFinite(shutter)||shutter<0)throw Error('Shutter duration must be finite and nonnegative.');
  const variables=[...inputs,...(time===null?[]:[time])];
  // Register all inputs before expansion, even if the color is constant or its
  // phase terms later cancel. This also validates every supplied numeric value.
  const declaration=evaluateGradients(float(0),variables,values);
  if(time!==null&&typeof declaration.gradients.at(-1)!=='number')throw Error('The time input must be scalar.');
  function validateFootprint(changes,label){
    if(!Array.isArray(changes))throw Error(`${label} must be an array in inputs order.`);
    if(changes.length===0)return;
    if(changes.length!==inputs.length)throw Error(`${label} must have one entry per designated input.`);
    for(let j=0;j<inputs.length;j++){
      const shape=declaration.gradients[j],change=changes[j];
      if(typeof shape==='number'){
        if(typeof change!=='number'||!Number.isFinite(change))throw Error(`${label}[${j}] must be a finite scalar.`);
      }else{
        if(!(Array.isArray(change)||ArrayBuffer.isView(change))||change.length!==shape.length||!Array.from(change).every(Number.isFinite))
          throw Error(`${label}[${j}] must be a finite vector of width ${shape.length}.`);
      }
    }
  }
  validateFootprint(dx,'dx');validateFootprint(dy,'dy');
  const e=expand(root),result=new Array(e.channels).fill(0);
  const dotInput=(g,d)=>typeof g==='number'?g*(d??0):g.reduce((s,v,i)=>s+v*(d?.[i]??0),0);
  for(const phase of e.phases)evaluateGradients(phase,variables,values);
  for(const term of e.terms){
    const evaluation=evaluateGradients(term.phase,variables,values),p=evaluation.value;
    const x=inputs.reduce((s,_,j)=>s+dotInput(evaluation.gradients[j],dx[j]),0),y=inputs.reduce((s,_,j)=>s+dotInput(evaluation.gradients[j],dy[j]),0);
    const dt=time?evaluation.gradients.at(-1)*shutter:0,weight=sinc(x*.5)*sinc(y*.5)*sinc(dt*.5);
    if(![p,x,y,dt,weight].every(Number.isFinite))throw Error('Nonfinite phase or footprint evaluation.');
    term.coefficients.forEach(([r,i],j)=>result[j]+=(r*Math.cos(p)-i*Math.sin(p))*weight);
  }
  if(!result.every(Number.isFinite))throw Error('Nonfinite filtered color.');
  return {value:result.length===1?result[0]:result,stats:e.stats};
}

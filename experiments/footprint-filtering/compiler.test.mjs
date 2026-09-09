import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector3,Texture} from 'three';
import {uniform,float,vec2,vec3,vec4,sin,cos,mix,Fn,texture} from 'three/tsl';
import {compileFootprint,evaluateFootprint,LIMITS} from './compiler.js';

const canonical=input=>{let n=input;while((n.self?.isNode&&n.self!==n)||n.constructor.type==='VarNode')n=n.self?.isNode&&n.self!==n?n.self:n.node;return n;};
const array=x=>Array.isArray(x)?x:[x];
function near(actual,expected,tolerance=2e-10) {
  const a=array(actual),b=array(expected);assert.equal(a.length,b.length);
  a.forEach((x,i)=>assert.ok(Number.isFinite(x)&&Math.abs(x-b[i])<=tolerance*Math.max(1,Math.abs(b[i])),`${x} != ${b[i]}`));
}
function broadcast(values,fn) {
  const arrays=values.map(array),width=Math.max(...arrays.map(a=>a.length));
  const result=Array.from({length:width},(_,i)=>fn(...arrays.map(a=>a[a.length===1?0:i])));
  return width===1?result[0]:result;
}

// Independent interpreter of the ACTUAL generated native node, not its Fourier
// representation. Spatial derivative nodes use endpoint differences. They are
// exact for the affine phases exercised below, without using the compiler's AD.
function nativeValue(root,values,dx=new Map(),dy=new Map()) {
  const normalized=new Map([...values].map(([k,v])=>[canonical(k),v]));
  function evaluate(node,bindings,cache=new Map()) {
    const n=canonical(node);if(bindings.has(n))return bindings.get(n);if(cache.has(n))return cache.get(n);
    const child=x=>evaluate(x,bindings,cache),type=n.constructor.type;let out;
    if(type==='ConstNode')out=typeof n.value==='number'?n.value:n.value.toArray();
    else if(type==='OperatorNode') {
      const operators={'+':(a,b)=>a+b,'-':(a,b)=>a-b,'*':(a,b)=>a*b,'/':(a,b)=>a/b,
        '<':(a,b)=>a<b,'<=':(a,b)=>a<=b,'>':(a,b)=>a>b,'>=':(a,b)=>a>=b,'==':(a,b)=>a===b};
      assert.ok(operators[n.op],`Test interpreter operator ${n.op}`);out=broadcast([child(n.aNode),child(n.bNode)],operators[n.op]);
    } else if(type==='MathNode') {
      if(n.method==='dFdx'||n.method==='dFdy') {
        const changes=n.method==='dFdx'?dx:dy;
        const shifted=scale=>new Map([...bindings].map(([k,v])=>{
          const delta=[...changes].find(([key])=>canonical(key)===k)?.[1]??0;
          return[k,broadcast([v,delta],(a,b)=>a+scale*b)];
        }));
        out=broadcast([evaluate(n.aNode,shifted(.5)),evaluate(n.aNode,shifted(-.5))],(a,b)=>a-b);
      } else {
        const operations={sin:Math.sin,cos:Math.cos,abs:Math.abs,max:Math.max,min:Math.min,pow:Math.pow,
          negate:x=>-x,oneMinus:x=>1-x};
        assert.ok(operations[n.method],`Test interpreter math ${n.method}`);
        out=broadcast([n.aNode,n.bNode,n.cNode].filter(Boolean).map(child),operations[n.method]);
      }
    } else if(type==='ConditionalNode')out=broadcast([child(n.condNode),child(n.ifNode),child(n.elseNode)],(c,a,b)=>c?a:b);
    else if(type==='JoinNode')out=n.nodes.flatMap(x=>array(child(x)));
    else if(type==='SplitNode') {
      const a=array(child(n.node)),result=[...n.components].map(c=>a[a.length===1?0:'xyzw'.indexOf(c)]);out=result.length===1?result[0]:result;
    } else if(type==='ConvertNode') {
      const a=array(child(n.node)),width={float:1,vec2:2,vec3:3,vec4:4}[n.convertTo];
      const result=Array.from({length:width},(_,i)=>a.length===1?a[0]:i<a.length?a[i]:i===3?1:0);out=width===1?result[0]:result;
    } else throw Error('Unsupported test interpreter node '+type);
    cache.set(n,out);return out;
  }
  return evaluate(root,normalized);
}

// Numerical Gauss-Legendre integration of the original plain-JS formula.
// No trigonometric identities, Fourier terms, sinc, or compiler output are used.
function gauss(count=24) {
  const pairs=[];
  for(let i=0;i<count;i++) {
    let z=Math.cos(Math.PI*(i+.75)/(count+.5)),derivative;
    for(let k=0;k<30;k++) {
      let p0=1,p1=z;for(let n=2;n<=count;n++){const next=((2*n-1)*z*p1-(n-1)*p0)/n;p0=p1;p1=next;}
      derivative=count*(z*p1-p0)/(z*z-1);const next=z-p1/derivative;
      if(Math.abs(next-z)<1e-15){z=next;break;}z=next;
    }
    pairs.push([z/2,1/((1-z*z)*derivative*derivative)]);
  }
  return pairs;
}
const QUADRATURE=gauss();
function integrate(formula,{center,dx,dy,time,shutter}) {
  let sum;
  const ax=dx.some(Boolean)?QUADRATURE:[[0,1]],ay=dy.some(Boolean)?QUADRATURE:[[0,1]],at=shutter?QUADRATURE:[[0,1]];
  for(const[x,wx]of ax)for(const[y,wy]of ay)for(const[t,wt]of at) {
    const value=array(formula(center.map((v,i)=>v+dx[i]*x+dy[i]*y),time+shutter*t));
    sum??=Array(value.length).fill(0);value.forEach((v,i)=>sum[i]+=v*wx*wy*wt);
  }
  return sum.length===1?sum[0]:sum;
}
function check(root,inputs,time,values,dx,dy,shutter,expected,tolerance=2e-10) {
  near(evaluateFootprint(root,{inputs,time,values,dx,dy,shutter}).value,expected,tolerance);
  const compiled=compileFootprint(root,{inputs,time,shutter:float(shutter)});
  assert.ok(compiled.node.isNode);assert.ok(Object.isFrozen(compiled.stats));
  near(nativeValue(compiled.node,values,new Map(inputs.map((n,i)=>[n,dx[i]])),new Map(inputs.map((n,i)=>[n,dy[i]]))),expected,tolerance);
  return compiled;
}

test('correlated sine squared retains its half mean over full periods',()=>{
  const x=uniform(0),s=sin(x),root=s.mul(s);
  for(const center of [-2.3,0,.37,5])check(root,[x],null,new Map([[x,center]]),[4*Math.PI],[0],0,.5);
});

test('nearby frequencies retain the difference-frequency beat',()=>{
  const x=uniform(0),t=uniform(0),root=sin(x.mul(19).add(t.mul(.7))).mul(sin(x.mul(18).sub(t.mul(.2))));
  const fixture={center:[.37],dx:[.8],dy:[.17],time:.23,shutter:.9};
  const expected=integrate(([x],t)=>Math.sin(19*x+.7*t)*Math.sin(18*x-.2*t),fixture);
  assert.ok(Math.abs(expected)>.2,'Fixture must retain a visible beat, not just average to zero');
  check(root,[x],t,new Map([[x,.37],[t,.23]]),[.8],[.17],.9,expected);
});

test('triple products use the joint space and shutter box',()=>{
  const p=uniform(new Vector3()),t=uniform(0),a=p.x.mul(4).add(p.y.mul(3)).add(t.mul(2)),b=p.y.mul(5).sub(t),c=p.x.sub(p.y).mul(2).add(t.mul(.7));
  const root=sin(a).mul(cos(b)).mul(sin(c)).add(.2),fixture={center:[.21,-.37,.5],dx:[.6,.1,0],dy:[-.15,.7,0],time:.43,shutter:.8};
  const expected=integrate(([x,y],t)=>Math.sin(4*x+3*y+2*t)*Math.cos(5*y-t)*Math.sin(2*(x-y)+.7*t)+.2,fixture);
  check(root,[p],t,new Map([[p,fixture.center],[t,fixture.time]]),[fixture.dx],[fixture.dy],fixture.shutter,expected);
});

test('scalar broadcasts, repeated swizzles, vector padding, mix and integer powers agree',()=>{
  const x=uniform(0),t=uniform(0),s=sin(x.add(t)),c=cos(x.mul(2).sub(t));
  const root=vec4(vec2(s,c)).wyxx.mul(.2).add(mix(vec4(.1,.2,.3,.4),vec4(s.pow(2)),c.mul(.2).add(.5)));
  const fixture={center:[.2],dx:[1.3],dy:[-.3],time:.4,shutter:.7};
  const expected=integrate(([x],t)=>{const s=Math.sin(x+t),c=Math.cos(2*x-t),u=.5+.2*c;return[1,c,s,s].map((v,i)=>.2*v+[.1,.2,.3,.4][i]*(1-u)+s*s*u);},fixture);
  check(root,[x],t,new Map([[x,.2],[t,.4]]),[1.3],[-.3],.7,expected);
});

test('zero footprint equals the original expression, including cancellation and negative constants',()=>{
  const x=uniform(0),t=uniform(0),s=sin(x.mul(2).add(t)),c=cos(x.sub(t.mul(3)));
  const root=vec3(s.mul(c).sub(s.pow(3)).add(.7),s.sub(s),c.div(-2).add(-.3));
  for(let i=0;i<12;i++){const a=-.8+i*.13,b=i*.23,sv=Math.sin(2*a+b),cv=Math.cos(a-3*b);check(root,[x],t,new Map([[x,a],[t,b]]),[0],[0],0,[sv*cv-sv**3+.7,0,-cv/2-.3]);}
});

test('near-zero and pure temporal footprints remain finite and match quadrature',()=>{
  const x=uniform(0),t=uniform(0),root=cos(x.add(t.mul(9)));
  for(const shutter of [0,1e-9,1e-4,.8]) {
    const f={center:[.4],dx:[1e-10],dy:[0],time:.7,shutter};
    check(root,[x],t,new Map([[x,.4],[t,.7]]),f.dx,f.dy,shutter,integrate(([x],t)=>Math.cos(x+9*t),f));
  }
});

test('compiling leaves source topology, inputs and shared nodes unchanged',()=>{
  const x=uniform(.3),t=uniform(.2),phase=x.mul(3).add(t),s=sin(phase),root=s.mul(s).add(vec3(.1,.2,.3));
  const records=new Map(),walk=input=>{const n=canonical(input);if(records.has(n))return;records.set(n,{keys:Reflect.ownKeys(n),values:Reflect.ownKeys(n).map(k=>n[k])});
    for(const key of ['aNode','bNode','cNode','node'])if(n[key]?.isNode)walk(n[key]);for(const child of n.nodes??[])walk(child);};
  walk(root);compileFootprint(root,{inputs:[x],time:t,shutter:float(.3)});
  for(const[n,snapshot]of records){assert.deepEqual(Reflect.ownKeys(n),snapshot.keys);snapshot.keys.forEach((k,i)=>assert.equal(n[k],snapshot.values[i],`${n.constructor.type}.${String(k)} mutated`));}
});

test('arbitrary functions, textures, mutable variables and unbound phase inputs reject',()=>{
  const x=uniform(0),y=uniform(0),options={inputs:[x]};
  for(const root of [Fn(()=>sin(x))(),sin(Fn(()=>x.add(1))()),texture(new Texture()).x,sin(x.toVar()),sin(x.add(y)),sin(x.add(1).before(y))])
    assert.throws(()=>compileFootprint(root,options));
  // Validation must not be bypassed just because the final color cancels.
  const invalid=sin(Fn(()=>x)());assert.throws(()=>compileFootprint(invalid.sub(invalid),options));
});

test('variable amplitudes, variable denominators and unsupported color operations reject',()=>{
  const x=uniform(0),a=uniform(1),options={inputs:[x,a]};
  for(const root of [sin(x).mul(a),sin(x).div(a),sin(x).abs(),sin(x).pow(.5),sin(x).pow(5),sin(vec3(x)),sin(x).div(0)])
    assert.throws(()=>compileFootprint(root,options));
});

test('cycles and depth budgets reject before traversal can run indefinitely',()=>{
  const x=uniform(0),root=sin(x).add(1);canonical(root).aNode=canonical(root);
  assert.throws(()=>compileFootprint(root,{inputs:[x]}),/Cyclic/);
  let deep=sin(x);for(let i=0;i<=LIMITS.depth;i++)deep=deep.add(.1);
  assert.throws(()=>compileFootprint(deep,{inputs:[x]}),/depth/);
});

test('Fourier term and harmonic-degree budgets reject exponential growth',()=>{
  const inputs=Array.from({length:8},()=>uniform(0));let root=float(1);
  for(const x of inputs)root=root.mul(sin(x));
  assert.throws(()=>compileFootprint(root,{inputs}),/terms/);
  const x=uniform(0),s=sin(x);let high=s;for(let i=0;i<LIMITS.phaseFactors;i++)high=high.mul(s);
  assert.throws(()=>compileFootprint(high,{inputs:[x]}),/degree/);
});

test('native color node budget also limits graphs whose algebra cancels',()=>{
  const x=uniform(0),phase=sin(x),branches=Array.from({length:1100},()=>phase.mul(float(0)));
  while(branches.length>1){const next=[];for(let i=0;i<branches.length;i+=2)next.push(branches[i+1]?branches[i].add(branches[i+1]):branches[i]);branches.splice(0,branches.length,...next);}
  assert.throws(()=>compileFootprint(branches[0],{inputs:[x]}),/node budget/);
});

test('CPU footprint durations and per-input widths fail closed',()=>{
  const p=uniform(new Vector3()),x=uniform(0),root=sin(p.x.add(x)),options={inputs:[p,x],values:new Map([[p,[1,2,3]],[x,.2]])};
  for(const shutter of [NaN,Infinity,-1,'0'])assert.throws(()=>evaluateFootprint(root,{...options,shutter}),/Shutter/);
  for(const dx of [[[1,2],.1],[[1,2,3],NaN],[[1,Infinity,3],.1],[[1,2,3],[.1]],[[1,2,3]],[[1,2,3],.1,0],[undefined,.1],null])
    assert.throws(()=>evaluateFootprint(root,{...options,dx}),/dx/);
  assert.throws(()=>evaluateFootprint(root,{...options,dy:[[0,0,0],Infinity]}),/dy/);
  near(evaluateFootprint(root,{...options}).value,Math.sin(1.2));
  assert.ok(Number.isFinite(evaluateFootprint(root,{...options,dx:[new Float32Array([.1,.2,.3]),.2]}).value));
});

test('scalar time and all designated inputs are validated even for constant color',()=>{
  const t=uniform(new Vector3()),x=uniform(0);
  assert.throws(()=>compileFootprint(float(2),{time:t}),/scalar/);
  assert.throws(()=>evaluateFootprint(float(2),{time:t,values:new Map([[t,[0,0,0]]])}),/scalar/);
  assert.throws(()=>evaluateFootprint(float(2),{inputs:[x],values:new Map([[x,NaN]])}),/finite/i);
  assert.throws(()=>compileFootprint(float(2),{inputs:[x],time:x}),/Duplicate/);
  assert.throws(()=>compileFootprint(float(2),{shutter:Infinity}),/Shutter/);
});

test('shader shutter accepts only initially valid scalar numbers, constants or uniforms',()=>{
  const x=uniform(0),t=uniform(0),root=sin(x.add(t)),options={inputs:[x],time:t};
  for(const shutter of [0,.1,float(0),float(.1),uniform(0),uniform(.1)])assert.ok(compileFootprint(root,{...options,shutter}).node.isNode);
  for(const shutter of [vec2(.1),uniform(new Vector3(.1,.1,.1)),uniform(1,'int'),float(NaN),float(Infinity),float(-.1),
    uniform(NaN),uniform(-.1),float(.1).mul(2),Fn(()=>float(.1))(),texture(new Texture()).x])
    assert.throws(()=>compileFootprint(root,{...options,shutter}),/Shutter/);
});

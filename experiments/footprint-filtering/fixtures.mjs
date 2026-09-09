import {float,vec2,vec3,sin,cos,abs,select} from 'three/tsl';

const TAU=2*Math.PI;
export const KINDS=Object.freeze(['weave','energy','interference']);
export const FIXTURES=Object.freeze({
  weave:Object.freeze({kind:'weave',name:'Correlated weave',description:'Close-frequency pair products retain their coarse difference pattern.',range:[.1,.9]}),
  energy:Object.freeze({kind:'energy',name:'Squared-wave energy',description:'Squared sine/cosine signals retain their constant mean under minification.',range:[.1,.9]}),
  interference:Object.freeze({kind:'interference',name:'Three-wave interference',description:'A+B-C has zero spatial frequency and a slow temporal beat.',range:[.1,.9]}),
});
// Entries [u cycles,v cycles,time cycles,phase radians]. Frequency scales only UV.
const PHASES=Object.freeze({
  weave:[[5,1.25,8,.2],[5.35,1.10,8.25,-.3],[-1.5,4,-5.5,.1]],
  energy:[[3,-2,7,.3],[1,3,-4,-.2]],
  interference:[[2,.5,6,.2],[-.4,1.75,-4,-.3],[1.6,2.25,1.5,.1]],
});
const requireKind=kind=>{if(!KINDS.includes(kind))throw Error('Unknown footprint fixture');};
function checkedFrequency(value=1){if(!Number.isFinite(value)||value<0||value>128)throw Error('Frequency must be finite in [0,128]');return value;}

/** An authored ordinary native TSL forward program; no filtering or compiler. */
export function makeFixture(kind,{uv,time,frequency}) {
  requireKind(kind);
  const phases=PHASES[kind].map(([u,v,t,b])=>uv.x.mul(u).add(uv.y.mul(v)).mul(frequency).add(time.mul(t)).mul(TAU).add(b));
  const [a,b,c]=phases;
  if(kind==='weave')return vec3(sin(a).mul(sin(b)).mul(.4).add(.5),sin(a).mul(sin(c)).mul(.4).add(.5),cos(b).mul(cos(c)).mul(.4).add(.5));
  if(kind==='energy'){const sa=sin(a),cb=cos(b);return vec3(sa.mul(sa).mul(.8).add(.1),cb.mul(cb).mul(.8).add(.1),sa.mul(cb).mul(.4).add(.5));}
  return vec3(sin(a).mul(sin(b)).mul(cos(c)).mul(.4).add(.5),cos(a).mul(cos(b)).mul(cos(c)).mul(.4).add(.5),sin(a).mul(cos(b)).mul(sin(c)).mul(.4).add(.5));
}

/** Independent numeric original products, not an evaluation of a Fourier list. */
export function numericFixture(kind,u,v,t,{frequency=1}={}) {
  requireKind(kind);checkedFrequency(frequency);
  if(![u,v,t].every(Number.isFinite))throw Error('Nonfinite point');
  if(kind==='weave') {
    const a=TAU*(frequency*(5*u+1.25*v)+8*t)+.2,b=TAU*(frequency*(5.35*u+1.10*v)+8.25*t)-.3,c=TAU*(frequency*(-1.5*u+4*v)-5.5*t)+.1;
    return [.5+.4*Math.sin(a)*Math.sin(b),.5+.4*Math.sin(a)*Math.sin(c),.5+.4*Math.cos(b)*Math.cos(c)];
  }
  if(kind==='energy') {
    const a=TAU*(frequency*(3*u-2*v)+7*t)+.3,b=TAU*(frequency*(u+3*v)-4*t)-.2;
    return [.1+.8*Math.sin(a)**2,.1+.8*Math.cos(b)**2,.5+.4*Math.sin(a)*Math.cos(b)];
  }
  const a=TAU*(frequency*(2*u+.5*v)+6*t)+.2,b=TAU*(frequency*(-.4*u+1.75*v)-4*t)-.3,c=TAU*(frequency*(1.6*u+2.25*v)+1.5*t)+.1;
  return [.5+.4*Math.sin(a)*Math.sin(b)*Math.cos(c),.5+.4*Math.cos(a)*Math.cos(b)*Math.cos(c),.5+.4*Math.sin(a)*Math.cos(b)*Math.sin(c)];
}

// Independent hand-applied product-to-sum identities. Each tuple is
// [sin/cos, coefficient, integer weights of A/B/C]. No generic product expansion.
const MANUAL={
  weave:[[['cos',.2,[1,-1,0]],['cos',-.2,[1,1,0]]],[['cos',.2,[1,0,-1]],['cos',-.2,[1,0,1]]],[['cos',.2,[0,1,-1]],['cos',.2,[0,1,1]]]],
  energy:[[['cos',-.4,[2,0]]],[['cos',.4,[0,2]]],[['sin',.2,[1,1]],['sin',.2,[1,-1]]]],
  interference:[
    [['cos',.1,[1,-1,1]],['cos',.1,[1,-1,-1]],['cos',-.1,[1,1,1]],['cos',-.1,[1,1,-1]]],
    [['cos',.1,[1,1,1]],['cos',.1,[1,1,-1]],['cos',.1,[1,-1,1]],['cos',.1,[1,-1,-1]]],
    [['cos',.1,[1,1,-1]],['cos',.1,[1,-1,-1]],['cos',-.1,[1,1,1]],['cos',-.1,[1,-1,1]]],
  ],
};
function combinedPhase(kind,weights) {
  return [0,1,2,3].map(k=>weights.reduce((sum,w,i)=>sum+w*PHASES[kind][i][k],0));
}

/** Per-channel explicit coefficients in radians: phase=ku*u+kv*v+kt*t+offset. */
export function analyticTerms(kind,frequency=1) {
  requireKind(kind);checkedFrequency(frequency);
  return MANUAL[kind].map(terms=>({constant:.5,terms:terms.map(([trig,amplitude,weights])=>{
    const [u,v,t,offset]=combinedPhase(kind,weights);
    return {trig,amplitude,phase:[TAU*frequency*u,TAU*frequency*v,TAU*t,offset]};
  })}));
}
export function sinc(x) {
  if(!Number.isFinite(x))throw Error('Nonfinite sinc argument');
  if(Math.abs(x)<1e-4){const xx=x*x;return 1-xx/6+xx*xx/120;}
  return Math.sin(x)/x;
}
function checkedBox(box) {
  const {u,v,t,du=[0,0],dv=[0,0],dt=0,frequency=1}=box;
  if(![u,v,t,dt].every(Number.isFinite)||dt<0||du.length!==2||dv.length!==2||![...du,...dv].every(Number.isFinite))throw Error('Invalid affine UV/time box');
  checkedFrequency(frequency);return {u,v,t,du,dv,dt,frequency};
}

/** Exact real-arithmetic affine box integral. du/dv are full pixel edges in UV;
 * dt is full shutter width. This is box averaging, not an ideal low-pass filter.
 */
export function integrateBox(kind,box) {
  const {u,v,t,du,dv,dt,frequency}=checkedBox(box);
  return analyticTerms(kind,frequency).map(channel=>channel.constant+channel.terms.reduce((sum,term)=>{
    const [ku,kv,kt,b]=term.phase,phase=ku*u+kv*v+kt*t+b;
    const attenuation=sinc((ku*du[0]+kv*du[1])/2)*sinc((ku*dv[0]+kv*dv[1])/2)*sinc(kt*dt/2);
    return sum+term.amplitude*Math[term.trig](phase)*attenuation;
  },0));
}

function nativeSinc(x) {
  const xx=x.mul(x),small=abs(x).lessThan(1e-3),denominator=select(small,float(1),x);
  return select(small,float(1).sub(xx.div(6)).add(xx.mul(xx).div(120)),sin(x).div(denominator));
}
/** A manual analytical GPU baseline; never calls the automatic compiler. */
export function manualFilteredFixture(kind,{uv,time,frequency,du=[0,0],dv=[0,0],dt=0}) {
  requireKind(kind);du=Array.isArray(du)?vec2(...du):du;dv=Array.isArray(dv)?vec2(...dv):dv;dt=float(dt);
  return vec3(...MANUAL[kind].map(terms=>terms.reduce((sum,[trig,amplitude,weights])=>{
    const [u,v,t,b]=combinedPhase(kind,weights);
    const phase=uv.x.mul(u).add(uv.y.mul(v)).mul(frequency).add(time.mul(t)).mul(TAU).add(b);
    const ax=du.x.mul(u).add(du.y.mul(v)).mul(frequency).mul(Math.PI);
    const ay=dv.x.mul(u).add(dv.y.mul(v)).mul(frequency).mul(Math.PI),at=dt.mul(Math.PI*t);
    const filtered=(trig==='sin'?sin(phase):cos(phase)).mul(nativeSinc(ax)).mul(nativeSinc(ay)).mul(nativeSinc(at));
    return sum.add(filtered.mul(amplitude));
  },float(.5))));
}

function gaussLegendre(n) {
  const x=Array(n),w=Array(n);
  for(let i=0;i<Math.ceil(n/2);i++) {
    let z=Math.cos(Math.PI*(i+.75)/(n+.5)),previous,derivative=0;
    do {let p=1,q=0;for(let j=1;j<=n;j++){const old=q;q=p;p=((2*j-1)*z*q-(j-1)*old)/j;}derivative=n*(z*p-q)/(z*z-1);previous=z;z=previous-p/derivative;}while(Math.abs(z-previous)>2e-15);
    x[i]=-z/2;x[n-1-i]=z/2;w[i]=w[n-1-i]=1/((1-z*z)*derivative*derivative);
  }
  return x.map((position,i)=>[position,w[i]]);
}

/** Independent direct-product quadrature, deliberately no analyticTerms use.
 * Tensor Gauss-Legendre over the unit box; zero-width axes collapse to one sample.
 */
export function quadratureBox(kind,box,{orders=[32,32,16]}={}) {
  requireKind(kind);const b=checkedBox(box),active=[Math.hypot(...b.du)>0,Math.hypot(...b.dv)>0,b.dt>0];
  if(orders.length!==3||orders.some(n=>!Number.isInteger(n)||n<2||n>64))throw Error('Quadrature orders must be three integers in [2,64]');
  const counts=orders.map((n,i)=>active[i]?n:1),samples=counts.reduce((a,c)=>a*c,1);
  if(samples>65536)throw Error('Bounded quadrature budget exceeded');
  const axes=counts.map(n=>n===1?[[0,1]]:gaussLegendre(n)),rgb=[0,0,0];
  for(const [x,wx] of axes[0])for(const [y,wy] of axes[1])for(const [s,ws] of axes[2]) {
    const value=numericFixture(kind,b.u+x*b.du[0]+y*b.dv[0],b.v+x*b.du[1]+y*b.dv[1],b.t+s*b.dt,{frequency:b.frequency});
    for(let c=0;c<3;c++)rgb[c]+=value[c]*wx*wy*ws;
  }
  return {rgb,samples,orders:counts};
}

export const CELLS=Object.freeze([
  {id:'point',u:.13,v:-.27,t:.17,frequency:3,du:[0,0],dv:[0,0],dt:0},
  {id:'magnified',u:.13,v:-.27,t:.17,frequency:1,du:[.0006,.0002],dv:[-.0001,.0008],dt:0},
  {id:'transition',u:-.21,v:.19,t:.31,frequency:4,du:[.015,.003],dv:[-.004,.014],dt:0},
  {id:'minified',u:.137,v:.093,t:.211,frequency:16,du:[.035,.012],dv:[-.008,.025],dt:0},
  {id:'anisotropic',u:.27,v:-.18,t:.27,frequency:24,du:[.032,.006],dv:[-.0001,.0007],dt:0},
  {id:'temporal',u:-.17,v:.23,t:.41,frequency:4,du:[0,0],dv:[0,0],dt:.25},
  {id:'combined',u:.16,v:-.23,t:.137,frequency:12,du:[.025,.008],dv:[-.01,.02],dt:.18},
  {id:'correlated-carrier-zero',u:.137,v:.093,t:.211,frequency:20,du:[.01,0],dv:[0,0],dt:0},
].map(cell=>Object.freeze({...cell,du:Object.freeze(cell.du),dv:Object.freeze(cell.dv)})));

export const CPU_PROTOCOL=Object.freeze({kind:'affine-trig-box-oracle-v1',fixtures:KINDS,cells:CELLS.map(c=>c.id),
  assumptions:['Globally affine phase in u,v,time; frequency is spatially and temporally constant within each box.',
    'Real-arithmetic box integral reference, not arbitrary material/rasterization or ideal spectral bandlimiting.',
    'No use of automatic compiler internals; numeric original products and hand product-to-sum expansion are independent.'],
  quadratureOrders:[32,32,16],convergenceOrders:[48,48,24],absoluteTolerance:2e-8,pointTolerance:2e-12,
  maxQuadratureSamplesPerCell:65536,scope:'24 deterministic boxes plus exact correlation/constant-mean negative controls; CPU only.'});

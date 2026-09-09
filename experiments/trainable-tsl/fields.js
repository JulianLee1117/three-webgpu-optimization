import {float, sin, exp} from 'three/tsl';

// Only the forward program is authored. Training derivatives come from autograd.js.
// p is an ordinary native vec3 TSL node; weights are native scalar input nodes.
export const FIELDS = Object.freeze({
  islands: {name: 'Procedural islands', count: 28, rate: .025},
  neural: {name: 'Neural terrain', count: 49, rate: .025},
});
export function field(kind, p, w) {
  if (!FIELDS[kind] || w.length !== FIELDS[kind].count) throw Error('Unknown field or weight count');
  if (kind === 'islands') {
    let height = w[27];
    for (let i = 0; i < 9; i++) {
      const dx = p.x.sub(w[3*i+1]), dz = p.z.sub(w[3*i+2]);
      height = height.add(w[3*i].mul(exp(dx.mul(dx).add(dz.mul(dz)).mul(-5))));
    }
    return height;
  }
  let height = w[48];
  for (let i = 0; i < 12; i++) {
    const hidden = sin(p.x.mul(w[4*i]).add(p.z.mul(w[4*i+1])).add(w[4*i+2]));
    height = height.add(hidden.mul(w[4*i+3]));
  }
  return height;
}

export function seeded(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}
export function initialWeights(kind, seed = 11) {
  const random = seeded(seed), w = new Float32Array(FIELDS[kind].count);
  if (kind === 'islands') {
    for (let i = 0; i < 9; i++) {
      w[3*i] = .035 * (random() - .5);
      w[3*i+1] = (i % 3 - 1) * .65;
      w[3*i+2] = (Math.floor(i / 3) - 1) * .65;
    }
  } else for (let i = 0; i < 12; i++) {
    w[4*i] = (random()-.5)*5;
    w[4*i+1] = (random()-.5)*5;
    w[4*i+2] = (random()-.5)*2;
    w[4*i+3] = (random()-.5)*.08;
  }
  return w;
}

// Independent numeric forward oracle. It does not traverse or evaluate the AD tape.
export function numericField(kind, x, z, w) {
  if (kind === 'islands') {
    let y = w[27];
    for (let i=0;i<9;i++) y += w[3*i] * Math.exp(-5*((x-w[3*i+1])**2+(z-w[3*i+2])**2));
    return y;
  }
  let y = w[48];
  for (let i=0;i<12;i++) y += Math.sin(x*w[4*i]+z*w[4*i+1]+w[4*i+2])*w[4*i+3];
  return y;
}
export function targetHeight(kind, x, z) {
  if (kind === 'islands') return .9*Math.exp(-6*((x+.42)**2+(z+.22)**2))
    + .72*Math.exp(-8*((x-.48)**2+(z-.28)**2)) - .4*Math.exp(-7*((x-.2)**2+(z+.6)**2));
  return .48*Math.sin(2.6*x+.5)*Math.cos(2.1*z-.35)+.2*Math.cos(3.5*z+x);
}
export function makeSamples(kind, seed=11, count=128) {
  if (!Number.isInteger(count) || count < 1 || count > 256) throw Error('Sample budget exceeded');
  const random=seeded(seed ^ 0x971), result=new Float32Array(count*4);
  for(let i=0;i<count;i++) {
    const x=Math.fround(2*random()-1), z=Math.fround(2*random()-1);
    result.set([x,z,targetHeight(kind,x,z),1],4*i);
  }
  return result;
}
export function loss(kind, weights, samples) {
  let sum=0;
  for(let i=0;i<samples.length;i+=4) sum+=(numericField(kind,samples[i],samples[i+1],weights)-samples[i+2])**2;
  return sum/(samples.length/4);
}
export function heldout(kind, weights) {
  let sum=0,maxError=0,count=0;
  // Offset grid, independent from the seeded random training observations.
  for(let i=0;i<31;i++) for(let j=0;j<31;j++) {
    const x=-1+2*(i+.37)/31,z=-1+2*(j+.61)/31;
    const error=Math.abs(numericField(kind,x,z,weights)-targetHeight(kind,x,z));
    sum+=error*error; maxError=Math.max(maxError,error);count++;
  }
  return {count,rmse:Math.sqrt(sum/count),maxError};
}

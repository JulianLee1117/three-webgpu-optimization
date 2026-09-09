import {vec3, sin, cos} from 'three/tsl';

// Forward-only native TSL programs. p.x is the material coordinate s in [0,1];
// p.y/z are fixed cross-section offsets (radius <= .025). Time has period 1.
// Eight scalar native inputs are the only editable parameters in either case.
export function makeProgram(kind, p, t, w) {
  if (w.length !== 8) throw Error('Expected eight native scalar parameters');
  const s=p.x, s2=s.mul(s), local=s.mul(s.negate().add(1)), phase=t.mul(2*Math.PI);
  if (kind==='ribbon') {
    const y=s2.mul(w[0].add(w[1].mul(sin(phase.add(w[2])))))
      .add(local.mul(w[3].add(w[4].mul(sin(s.mul(Math.PI).sub(phase))))));
    const z=s2.mul(w[5].add(w[6].mul(cos(phase))))
      .add(local.mul(w[7].mul(sin(phase.mul(2)))));
    return vec3(s.mul(3),p.y.add(y),p.z.add(z));
  }
  if (kind==='tentacle') {
    const theta=w[0].add(w[1].mul(sin(phase.add(w[2])))).add(w[3].mul(sin(phase.mul(2))));
    const angle=s.mul(theta), x=sin(angle).mul(3).div(theta), y=cos(angle).negate().add(1).mul(3).div(theta);
    const z=s2.mul(w[4].add(w[5].mul(sin(phase))))
      .add(local.mul(w[6].add(w[7].mul(cos(phase.mul(2))))));
    return vec3(x,p.y.add(y),p.z.add(z));
  }
  throw Error('Unknown constraint-editing program');
}

// Independent arithmetic oracle; does not evaluate the native graph or AD tape.
export function numericProgram(kind, p, t, w) {
  const s=p[0], phase=2*Math.PI*t, local=s*(1-s);
  if (kind==='ribbon') return [3*s,
    p[1]+s*s*(w[0]+w[1]*Math.sin(phase+w[2]))+local*(w[3]+w[4]*Math.sin(Math.PI*s-phase)),
    p[2]+s*s*(w[5]+w[6]*Math.cos(phase))+local*w[7]*Math.sin(2*phase)];
  if (kind==='tentacle') {
    const theta=w[0]+w[1]*Math.sin(phase+w[2])+w[3]*Math.sin(2*phase);
    if (!(theta>.2)) throw Error('Tentacle curvature outside its nonsingular contract');
    return [3*Math.sin(s*theta)/theta,p[1]+3*(1-Math.cos(s*theta))/theta,
      p[2]+s*s*(w[4]+w[5]*Math.sin(phase))+local*(w[6]+w[7]*Math.cos(2*phase))];
  }
  throw Error('Unknown constraint-editing program');
}

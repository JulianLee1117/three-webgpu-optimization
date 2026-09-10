// Independent scalar quadratic oracle; no Three imports or source rewriting.
export function ellipsoidIntersection(covariance,origin,direction,{cutoff=2,flatness=1e-4}={}){
  const [a,b,c,d,e,f]=covariance,delta=Math.max(a,d,f)*flatness;
  const A=a+delta,D=d+delta,F=f+delta;
  const cof=[D*F-e*e,c*e-b*F,b*e-c*D,A*F-c*c,b*c-A*e,A*D-b*b];
  const determinant=A*cof[0]+b*cof[1]+c*cof[2];
  if(!(determinant>0))throw Error('Oracle requires positive-definite regularized covariance');
  const m=cof.map(x=>x/determinant),mul=v=>[
    m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[1]*v[0]+m[3]*v[1]+m[4]*v[2],m[2]*v[0]+m[4]*v[1]+m[5]*v[2]],
    dot=(u,v)=>u.reduce((s,x,i)=>s+x*v[i],0),md=mul(direction),mo=mul(origin),
    quadratic=[dot(direction,md),2*dot(origin,md),dot(origin,mo)-cutoff*cutoff],
    discriminant=quadratic[1]**2-4*quadratic[0]*quadratic[2];
  if(discriminant<0)return {hit:false,quadratic,discriminant,regularization:delta};
  const roots=[(-quadratic[1]-Math.sqrt(discriminant))/(2*quadratic[0]),(-quadratic[1]+Math.sqrt(discriminant))/(2*quadratic[0])],
    distance=roots.find(t=>t>=0);
  if(distance===undefined)return {hit:false,roots,quadratic,discriminant,regularization:delta};
  const point=origin.map((v,i)=>v+distance*direction[i]);
  return {hit:true,distance,point,residual:Math.abs(dot(point,mul(point))-cutoff*cutoff),roots,quadratic,discriminant,regularization:delta};
}

export function fixture(name,axis,{offset=1.8,opacity=255}={}){
  const length=Math.hypot(...axis),v=axis.map(x=>x/length),
    cov=Float32Array.from([.01+.99*v[0]*v[0],.99*v[0]*v[1],.99*v[0]*v[2],
      .01+.99*v[1]*v[1],.99*v[1]*v[2],.01+.99*v[2]*v[2]]),
    direction=v[2]===0?[0,0,1]:[Math.SQRT1_2,-Math.SQRT1_2,0],
    origin=v.map((x,i)=>offset*x-direction[i]);
  return {name,covariance:Array.from(cov),origin,direction,opacity,offset,
    longAxis:v,unroundedEigenvalues:[1,.01,.01]};
}

export const FIXTURES=[
  fixture('rotated-45deg-true-hit',[1,1,0]),
  fixture('aligned-true-hit',[1,0,0]),
  fixture('rotated-three-axes-true-hit',[1,1,1]),
  fixture('rotated-center-true-hit',[1,1,0],{offset:0}),
  fixture('rotated-outside-miss',[1,1,0],{offset:2.2}),
  fixture('faint-splat-excluded',[1,1,0],{opacity:25})
];

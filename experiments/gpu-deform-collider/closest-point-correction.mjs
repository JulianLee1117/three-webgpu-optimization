// Experiment-local correction; the installed dependency remains untouched.
// Barycentric projection alone does not select the nearest edge when two
// coordinates are negative at an obtuse corner. Use standard Voronoi regions;
// boundary-segment fallback handles zero-area triangles for this primitive.
export const correctedTriangleWGSL = `
  fn closestPointToTriangle(p:vec3f, v0:vec3f, v1:vec3f, v2:vec3f) -> vec3f {
    let ab=v1-v0; let ac=v2-v0; let ap=p-v0;
    let n=cross(ab,ac); let n2=dot(n,n);
    if(n2>0.0){
      let d1=dot(ab,ap);let d2=dot(ac,ap);
      if(d1<=0.0 && d2<=0.0){return vec3f(1.0,0.0,0.0);}
      let bp=p-v1;let d3=dot(ab,bp);let d4=dot(ac,bp);
      if(d3>=0.0 && d4<=d3){return vec3f(0.0,1.0,0.0);}
      let vc=d1*d4-d3*d2;
      if(vc<=0.0 && d1>=0.0 && d3<=0.0){let v=d1/(d1-d3);return vec3f(1.0-v,v,0.0);}
      let cp=p-v2;let d5=dot(ab,cp);let d6=dot(ac,cp);
      if(d6>=0.0 && d5<=d6){return vec3f(0.0,0.0,1.0);}
      let vb=d5*d2-d1*d6;
      if(vb<=0.0 && d2>=0.0 && d6<=0.0){let w=d2/(d2-d6);return vec3f(1.0-w,0.0,w);}
      let va=d3*d6-d5*d4;
      if(va<=0.0 && d4-d3>=0.0 && d5-d6>=0.0){let w=(d4-d3)/((d4-d3)+(d5-d6));return vec3f(0.0,1.0-w,w);}
      let inv=1.0/(va+vb+vc);let v=vb*inv;let w=vc*inv;return vec3f(1.0-v-w,v,w);
    }
    let bc=v2-v1; let bp=p-v1;
    var tab=0.0; var tac=0.0; var tbc=0.0;
    if(dot(ab,ab)>0.0){tab=clamp(dot(ap,ab)/dot(ab,ab),0.0,1.0);}
    if(dot(ac,ac)>0.0){tac=clamp(dot(ap,ac)/dot(ac,ac),0.0,1.0);}
    if(dot(bc,bc)>0.0){tbc=clamp(dot(bp,bc)/dot(bc,bc),0.0,1.0);}
    let dab=p-(v0+tab*ab);let dac=p-(v0+tac*ac);let dbc=p-(v1+tbc*bc);
    let ab2=dot(dab,dab);let ac2=dot(dac,dac);let bc2=dot(dbc,dbc);
    if(ab2<=ac2 && ab2<=bc2){return vec3f(1.0-tab,tab,0.0);}
    if(ac2<=bc2){return vec3f(1.0-tac,0.0,tac);}
    return vec3f(0.0,1.0-tbc,tbc);
  }
`;

export function correctTriangleModule(source) {
  const marker='export const closestPointToTriangle = wgslTagFn/* wgsl */`';
  const start=source.indexOf(marker),end=source.indexOf('`;',start+marker.length);
  if(start<0||end<0||source.indexOf(marker,start+1)>=0)throw Error('Unexpected upstream triangle module');
  const old=source.slice(start+marker.length,end);
  if(!old.includes('else if ( v < 0.0 )')||!old.includes('return vec3f( w, u, v );'))throw Error('Upstream function changed; review correction');
  return source.slice(0,start+marker.length)+correctedTriangleWGSL+source.slice(end);
}

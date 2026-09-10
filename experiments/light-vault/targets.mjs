import {silhouette} from '../caustic-sketch/solver.mjs';
export const OPTICS=Object.freeze({n:256,pitch:10e-6,wavelength:532e-9,apertureRadius:54,distances:[.012,.030],range:[.004,.038]});
export function presetAmplitude(kind){
  const {n}=OPTICS,a=new Float64Array(n*n);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const ax=(x+.5-n/2)/42,ay=(n/2-y-.5)/42;
    a[y*n+x]=silhouette(kind,ax,ay)?1:Math.hypot(ax,ay)<1.6?Math.sqrt(.002):0;
  }
  return a;
}
export function maskAmplitude(mask,width=128,height=128){
  if(!(mask instanceof Uint8Array)||mask.length!==width*height||width<8||height<8||width>256||height>256)throw Error('Invalid drawing');
  const {n}=OPTICS,a=new Float64Array(n*n);let sum=0;
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const ax=(x+.5-n/2)/42,ay=(y+.5-n/2)/42;
    const mx=Math.floor((ax+1)*width/2),my=Math.floor((ay+1)*height/2);
    const ink=mx>=0&&my>=0&&mx<width&&my<height?mask[my*width+mx]/255:0;
    a[y*n+x]=Math.sqrt(ink+(Math.hypot(ax,ay)<1.6?.002:0));sum+=ink;
  }
  if(sum<20)throw Error('Draw a bolder mark on both cards.');return a;
}
export function aperture(){const {n,apertureRadius}=OPTICS;return Float64Array.from({length:n*n},(_,i)=>Math.hypot(i%n+.5-n/2,Math.floor(i/n)+.5-n/2)<apertureRadius?1:0);}
export function binIntensity(pixels,n=256,factor=4){const b=new Float64Array((n/factor)**2);for(let y=0;y<n;y++)for(let x=0;x<n;x++)b[Math.floor(y/factor)*(n/factor)+Math.floor(x/factor)]+=pixels[y*n+x]/factor**2;return b;}
export function matchImage(pixels,targets){
  const p=binIntensity(pixels),scores=targets.map(target=>{const q=binIntensity(Float64Array.from(target.amplitude??target,v=>v*v));let dot=0,pp=0,qq=0;for(let i=0;i<p.length;i++){dot+=p[i]*q[i];pp+=p[i]**2;qq+=q[i]**2;}return dot/Math.sqrt(pp*qq);});
  const order=[0,1].sort((a,b)=>scores[b]-scores[a]);return{scores,best:order[0],margin:scores[order[0]]-scores[order[1]],clear:scores[order[0]]>=.85&&scores[order[0]]-scores[order[1]]>=.15};
}

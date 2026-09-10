/** Original bounded scalar-wave experiment. SI units; no renderer or artwork lookup. */
const MAX_COMPONENT = 1e100;
function validateGrid(n) {
  if(!Number.isInteger(n)||n<8||n>512||(n&(n-1)))throw Error('Grid must be a power of two from 8 through 512');
}
function validateSamples(values,length,label,{nonnegative=false,float64=false}={}) {
  const numericArray=Array.isArray(values)||values instanceof Float32Array||values instanceof Float64Array;
  if(!numericArray||(float64&&!(values instanceof Float64Array))||values.length!==length)throw Error(`Invalid ${label} array length or type`);
  for(let i=0;i<length;i++)if(!Number.isFinite(values[i])||Math.abs(values[i])>MAX_COMPONENT||(nonnegative&&values[i]<0))throw Error(`Invalid ${label} sample`);
}
function validateComplex(field,n,label='complex field') {
  validateGrid(n);validateSamples(field,2*n*n,label,{float64:true});
}
function validateOptics(pitch,wavelength,distance) {
  // Broad optical SI bounds keep reciprocal frequencies and squared wave numbers finite.
  if(!Number.isFinite(pitch)||pitch<1e-9||pitch>1||!Number.isFinite(wavelength)||wavelength<1e-12||wavelength>1||!Number.isFinite(distance)||Math.abs(distance)>.5)throw Error('Invalid optical units: pitch 1 nm–1 m, wavelength 1 pm–1 m, |distance| <= 0.5 m');
}
export function fft2(field,n,inverse=false) {
  validateComplex(field,n);if(typeof inverse!=='boolean')throw Error('FFT direction must be boolean');
  return fft2Unchecked(field,n,inverse);
}
function fft2Unchecked(field,n,inverse=false) {
  for(let axis=0;axis<2;axis++)for(let line=0;line<n;line++){
    const stride=axis===0?1:n,offset=axis===0?line*n:line;
    for(let i=1,j=0;i<n;i++){
      let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
      if(i<j){const a=2*(offset+i*stride),b=2*(offset+j*stride);let t=field[a];field[a]=field[b];field[b]=t;t=field[a+1];field[a+1]=field[b+1];field[b+1]=t;}
    }
    for(let len=2;len<=n;len*=2){
      const angle=(inverse?2:-2)*Math.PI/len,wr=Math.cos(angle),wi=Math.sin(angle);
      for(let i=0;i<n;i+=len){let cr=1,ci=0;
        for(let j=0;j<len/2;j++){
          const a=2*(offset+(i+j)*stride),b=2*(offset+(i+j+len/2)*stride);
          const br=field[b]*cr-field[b+1]*ci,bi=field[b]*ci+field[b+1]*cr,ar=field[a],ai=field[a+1];
          field[a]=ar+br;field[a+1]=ai+bi;field[b]=ar-br;field[b+1]=ai-bi;
          const tr=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=tr;
        }
      }
    }
  }
  if(inverse)for(let i=0;i<field.length;i++)field[i]/=n*n;
  return field;
}
export function transfer(n,pitch,wavelength,distance) {
  validateGrid(n);validateOptics(pitch,wavelength,distance);
  const h=new Float64Array(2*n*n),k=2*Math.PI/wavelength;
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const fx=(x<n/2?x:x-n)/(n*pitch),fy=(y<n/2?y:y-n)/(n*pitch),s=wavelength*wavelength*(fx*fx+fy*fy);
    if(s>=1)continue;
    // Rationalized angular spectrum phase removes the irrelevant, very large carrier kz.
    const phase=-k*distance*s/(1+Math.sqrt(1-s)),i=2*(y*n+x);
    h[i]=Math.cos(phase);h[i+1]=Math.sin(phase);
  }
  return h;
}
export function propagate(field,n,h,backward=false) {
  validateComplex(field,n);validateComplex(h,n,'transfer function');
  if(typeof backward!=='boolean')throw Error('Propagation direction must be boolean');
  for(let i=0;i<h.length;i+=2)if(h[i]*h[i]+h[i+1]*h[i+1]>1+1e-12)throw Error('Transfer function must be passive');
  return propagateUnchecked(field,n,h,backward);
}
function propagateUnchecked(field,n,h,backward=false) {
  const out=fft2Unchecked(field.slice(),n);
  for(let i=0;i<out.length;i+=2){const hr=h[i],hi=h[i+1]*(backward?-1:1),a=out[i],b=out[i+1];out[i]=a*hr-b*hi;out[i+1]=a*hi+b*hr;}
  return fft2Unchecked(out,n,true);
}
export function intensity(field){
  const n=Math.sqrt((field?.length??0)/2);validateComplex(field,n);
  const out=new Float64Array(field.length/2);for(let i=0;i<out.length;i++)out[i]=field[2*i]**2+field[2*i+1]**2;return out;
}
export function sourceFromPhase(phase,aperture){
  const n=Math.sqrt(phase?.length??0);validateGrid(n);
  validateSamples(phase,n*n,'phase');validateSamples(aperture,n*n,'aperture',{nonnegative:true});
  return sourceFromPhaseUnchecked(phase,aperture);
}
function sourceFromPhaseUnchecked(phase,aperture){const field=new Float64Array(phase.length*2);for(let i=0;i<phase.length;i++){field[2*i]=aperture[i]*Math.cos(phase[i]);field[2*i+1]=aperture[i]*Math.sin(phase[i]);}return field;}
export function solveHologram(targets,{n=256,pitch=10e-6,wavelength=532e-9,iterations=70,onProgress,seed=731,apertureRadius=54}={}){
  validateGrid(n);validateOptics(pitch,wavelength,0);
  if(!Array.isArray(targets)||targets.length<1||targets.length>3)throw Error('Bounded 1–3 plane target required');
  if(!Number.isInteger(iterations)||iterations<1||iterations>120)throw Error('Iterations must be an integer from 1 through 120');
  if(!Number.isFinite(apertureRadius)||apertureRadius<1||apertureRadius>n/2)throw Error('Aperture radius must be from 1 through half the grid size');
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw Error('Seed must be an unsigned 32-bit integer');
  if(onProgress!==undefined&&typeof onProgress!=='function')throw Error('onProgress must be a function');
  for(const target of targets){
    if(target===null||typeof target!=='object')throw Error('Invalid target plane');
    validateSamples(target.amplitude,n*n,'target amplitude',{nonnegative:true});validateOptics(pitch,wavelength,target.distance);
    let targetEnergy=0;for(let i=0;i<target.amplitude.length;i++)targetEnergy+=target.amplitude[i]**2;
    if(!(targetEnergy>0)||!Number.isFinite(targetEnergy))throw Error('Empty or non-finite target energy');
  }
  const began=performance.now(),count=n*n,aperture=new Float64Array(count),phase=new Float64Array(count);let energy=0,random=seed>>>0;
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const i=y*n+x,r=Math.hypot(x+.5-n/2,y+.5-n/2);
    aperture[i]=r<apertureRadius?1:0;energy+=aperture[i]**2;
    random^=random<<13;random^=random>>>17;random^=random<<5;phase[i]=(random>>>0)/4294967296*2*Math.PI;
  }
  const planes=targets.map(t=>{const sum=t.amplitude.reduce((s,v)=>s+v*v,0);if(!sum)throw Error('Empty target');return{amplitude:Float64Array.from(t.amplitude,v=>v*Math.sqrt(energy/sum)),distance:t.distance,h:transfer(n,pitch,wavelength,t.distance)};});
  let field=sourceFromPhaseUnchecked(phase,aperture);const history=[];
  for(let iteration=0;iteration<iterations;iteration++){
    if(performance.now()-began>15000)throw Error('Wave fit exceeded 15-second CPU bound');
    const sum=new Float64Array(field.length);let loss=0;
    for(const plane of planes){
      const u=propagateUnchecked(field,n,plane.h);
      for(let i=0;i<count;i++){
        const a=Math.hypot(u[2*i],u[2*i+1]);loss+=(a-plane.amplitude[i])**2/energy/planes.length;
        const scale=plane.amplitude[i]/Math.max(a,1e-12);u[2*i]*=scale;u[2*i+1]*=scale;
      }
      const back=propagateUnchecked(u,n,plane.h,true);for(let i=0;i<sum.length;i++)sum[i]+=back[i];
    }
    for(let i=0;i<count;i++)phase[i]=Math.atan2(sum[2*i+1],sum[2*i]);field=sourceFromPhaseUnchecked(phase,aperture);
    if(iteration%10===0||iteration===iterations-1){history.push({iteration,loss});onProgress?.({iteration,phase:phase.slice(),loss});}
  }
  return {n,pitch,wavelength,phase,aperture,energy,history,milliseconds:performance.now()-began,distances:planes.map(p=>p.distance)};
}

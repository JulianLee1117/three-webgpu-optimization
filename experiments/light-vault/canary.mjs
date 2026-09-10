import {mkdir,writeFile} from 'node:fs/promises';
import {deflateSync} from 'node:zlib';
import {solveHologram,propagate,transfer,sourceFromPhase,intensity,fft2} from './wave.mjs';
import {silhouette} from '../caustic-sketch/solver.mjs';
const n=256,pitch=10e-6,wavelength=532e-9;
function target(kind){const mask=new Float64Array(n*n);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const ax=(x+.5-n/2)/42,ay=(n/2-y-.5)/42;mask[y*n+x]=silhouette(kind,ax,ay)?1:Math.hypot(ax,ay)<1.6?Math.sqrt(.002):0;}return mask;}
const targets=[{kind:'heart',distance:.012,amplitude:target('heart')},{kind:'letterA',distance:.030,amplitude:target('letterA')}];
const solved=solveHologram(targets,{n,pitch,wavelength,iterations:80});
const field=sourceFromPhase(solved.phase,solved.aperture);
function cosine(a,b){let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]**2;bb+=b[i]**2;}return dot/Math.sqrt(aa*bb);}
function bin(a,factor=4){const b=new Float64Array((n/factor)**2);for(let y=0;y<n;y++)for(let x=0;x<n;x++)b[Math.floor(y/factor)*(n/factor)+Math.floor(x/factor)]+=a[y*n+x]/factor**2;return b;}
const frames=[];
for(const distance of [.004,.012,.020,.030,.038]){const pixels=intensity(propagate(field,n,transfer(n,pitch,wavelength,distance)));frames.push({distance,pixels,energy:pixels.reduce((a,b)=>a+b,0),similarity:targets.map(t=>cosine(bin(pixels),bin(Float64Array.from(t.amplitude,v=>v*v))))});}
const out=`results/development/light-vault/${new Date().toISOString().replaceAll(':','-')}`;await mkdir(out,{recursive:true});
const report={parameters:{n,pitch,wavelength,apertureRadius:54},milliseconds:solved.milliseconds,history:solved.history,energy:solved.energy,frames:frames.map(({pixels,...frame})=>frame)};
await writeFile(out+'/report.json',JSON.stringify(report,null,2));await writeFile(out+'/solution.json',JSON.stringify({...solved,phase:Array.from(solved.phase),aperture:Array.from(solved.aperture)}));
// Original tiny PNG encoder for numerical fields (not an edited art asset).
function crc(bytes){let c=0xffffffff;for(const byte of bytes){c^=byte;for(let i=0;i<8;i++)c=(c>>>1)^(c&1?0xedb88320:0);}return(c^0xffffffff)>>>0;}
function chunk(name,raw){const type=Buffer.from(name),head=Buffer.alloc(4),tail=Buffer.alloc(4);head.writeUInt32BE(raw.length);tail.writeUInt32BE(crc(Buffer.concat([type,raw])));return Buffer.concat([head,type,raw,tail]);}
const w=n*5,h=n*2,rgba=new Uint8Array(w*h*4),peak=30;
for(let panel=0;panel<5;panel++)for(let y=0;y<n;y++)for(let x=0;x<n;x++){
  const p=frames[panel].pixels[y*n+x],v=255*(1-Math.exp(-p/peak)),j=(y*w+panel*n+x)*4;rgba[j]=v;rgba[j+1]=v*.88;rgba[j+2]=v*.56;rgba[j+3]=255;
  const tj=((y+n)*w+panel*n+x)*4,kind=panel===1?0:panel===3?1:-1,mask=kind>=0?targets[kind].amplitude[y*n+x]**2:0;rgba[tj]=rgba[tj+1]=rgba[tj+2]=mask*255;rgba[tj+3]=255;
}
const scan=Buffer.alloc(h*(w*4+1));for(let y=0;y<h;y++)scan.set(rgba.subarray(y*w*4,(y+1)*w*4),y*(w*4+1)+1);
const head=Buffer.alloc(13);head.writeUInt32BE(w);head.writeUInt32BE(h,4);head[8]=8;head[9]=6;
await writeFile(out+'/contact-sheet.png',Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',head),chunk('IDAT',deflateSync(scan)),chunk('IEND',Buffer.alloc(0))]));
console.log(JSON.stringify({out,...report},null,2));

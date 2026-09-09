/** Deterministic material-mip fixtures and independent numeric filtering controls.
 * Fine texels are vec4(normal.xyz, perceptual roughness). Coarse parameters are
 * vec4(slopeX,slopeY,perceptual roughness,0), decoded as normalize(sx,sy,1).
 * Target ordering is texel-major, direction-minor, vec4(linear RGB,1).
 * No GPU, AD compiler or shader implementation is imported here.
 */
export const MIP_LIMITS = Object.freeze({fineSide:64,coarseSide:16,footprint:4,trainPairs:32,heldoutPairs:64,
  gridCandidates:48,roughnessMin:.08,roughnessMax:1,slopeLimit:1.5,cpuTexels:8,cpuSteps:128,cpuWallMs:10000,
  learningRate:.02,cpuFDStep:1e-4,minimumHeldoutMSEImprovement:.30});
export const METAL_F0 = Object.freeze([.65,.3,.12]);
export const MATERIALS = Object.freeze({
  hammered:{name:'Hammered copper',description:'Crossed high-frequency height waves with a diagonal detail and correlated roughness.'},
  scored:{name:'Scored copper',description:'Modulated directional machining ridges, cross ridges and roughness correlated with ridge orientation.'},
});
export const VARIANCE_FILTER = Object.freeze({name:'Filament-style vMF roughness prefilter',threshold:.2,
  source:'https://github.com/google/filament/blob/main/tools/roughness-prefilter/src/main.cpp#L184-L217',checked:'2026-09-09',
  formula:'r=length(mean(unit normals)); kappa=(3r-r^3)/(1-r^2); variance=.25/kappa; r_out=sqrt(r_in^2+min(2*variance,.2^2))',
  caveat:'Reproduces the published practical channel-filter formula. This is not exact convolution of arbitrary anisotropic normal distributions with isotropic GGX.'});
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
const unit=value=>{const length=Math.hypot(...value);if(!(length>0))throw Error('Zero normal/direction');return value.map(x=>x/length);};
const randomGenerator=seed=>{let state=seed>>>0;return()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};};
const allTexels=()=>Array.from({length:MIP_LIMITS.coarseSide**2},(_,i)=>i);
function selectedTexels(texels){
  const selected=texels??allTexels();
  if(!Array.isArray(selected)||selected.length<1||selected.length>256||new Set(selected).size!==selected.length
    ||selected.some(i=>!Number.isInteger(i)||i<0||i>=256))throw Error('Expected unique bounded coarse texel indices');
  return selected;
}
function radicalInverse(index,base){let result=0,scale=1/base;while(index>0){result+=(index%base)*scale;index=Math.floor(index/base);scale/=base;}return result;}
function direction(z,turn){const radius=Math.sqrt(1-z*z),angle=2*Math.PI*turn;return unit([radius*Math.cos(angle),radius*Math.sin(angle),z]).map(Math.fround);}
export function makeDirections(split){
  if(!['train','heldout'].includes(split))throw Error('Unknown direction split');
  const count=split==='train'?32:64,start=split==='train'?1:65;
  return Array.from({length:count},(_,j)=>{const i=start+j;return {
    light:direction(.45+.55*radicalInverse(i,2),radicalInverse(i,3)),
    view:direction(.45+.55*radicalInverse(i,5),radicalInverse(i,7)),index:i,
  };});
}
export function cpuSubset(){
  const indices=allTexels(),random=randomGenerator(0x6d697073);
  for(let i=indices.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[indices[i],indices[j]]=[indices[j],indices[i]];}
  return indices.slice(0,MIP_LIMITS.cpuTexels).sort((a,b)=>a-b);
}
export function parametersToNormal(parameters,offset=0){return unit([parameters[offset],parameters[offset+1],1]);}
export function parametersAt(parameters,texels){const selected=selectedTexels(texels),output=new Float32Array(selected.length*4);selected.forEach((id,i)=>output.set(parameters.slice(4*id,4*id+4),4*i));return output;}

export function createFixture(kind){
  if(!Object.hasOwn(MATERIALS,kind))throw Error('Unknown material fixture');
  const fine=new Float32Array(64*64*4),tau=2*Math.PI;
  for(let y=0;y<64;y++)for(let x=0;x<64;x++){
    const u=(x+.5)/64,v=(y+.5)/64;
    let du,dv,roughness;
    if(kind==='hammered'){
      const a=tau*15*u,b=tau*13*v,c=tau*(23*u+19*v);
      du=.0035*tau*15*Math.cos(a)*Math.sin(b)-.0018*tau*23*Math.sin(c);
      dv=.0035*tau*13*Math.sin(a)*Math.cos(b)-.0018*tau*19*Math.sin(c);
      roughness=.14+.29*(.5+.5*Math.cos(a)*Math.sin(b))+.04*(.5+.5*Math.sin(c));
    }else{
      const phase=tau*21*u+.4*Math.sin(tau*3*v),cross=tau*17*v;
      du=.0028*tau*21*Math.cos(phase);
      dv=.0028*Math.cos(phase)*.4*tau*3*Math.cos(tau*3*v)+.0015*tau*17*Math.cos(cross);
      roughness=.12+.34*(.5+.5*Math.cos(phase))+.035*(.5+.5*Math.sin(cross));
    }
    fine.set([...unit([-du,-dv,1]),roughness],4*(y*64+x));
  }
  const average=new Float32Array(256*4),variance=new Float32Array(256*4),normalLengths=new Float64Array(256);
  const fixture={kind,name:MATERIALS[kind].name,fineSide:64,coarseSide:16,fine,trainPairs:makeDirections('train'),heldoutPairs:makeDirections('heldout'),
    baselines:{average,variance},meanNormalLengths:normalLengths};
  for(let id=0;id<256;id++){
    const texels=fineFootprint(fixture,id),sum=[0,0,0];let roughness=0;
    for(const value of texels){const normal=unit(value.slice(0,3));for(let k=0;k<3;k++)sum[k]+=normal[k]/16;roughness+=value[3]/16;}
    const length=clamp(Math.hypot(...sum),0,1),sx=sum[0]/sum[2],sy=sum[1]/sum[2];normalLengths[id]=length;
    // Fixed practical vMF channel filter from Filament's roughness-prefilter.
    // Re-expressed as inverse concentration to avoid infinity at r==1.
    const inverseConcentration=length<1?(1-length*length)/(3*length-length**3):0;
    const filtered=clamp(Math.sqrt(roughness*roughness+Math.min(.5*inverseConcentration,.2**2)),.08,1);
    average.set([sx,sy,roughness,0],4*id);variance.set([sx,sy,filtered,0],4*id);
  }
  const trainingKeys=new Set(fixture.trainPairs.map(pair=>JSON.stringify([pair.light,pair.view])));
  if(fixture.heldoutPairs.some(pair=>trainingKeys.has(JSON.stringify([pair.light,pair.view]))))throw Error('Train/held-out direction overlap');
  return fixture;
}
export function fineFootprint(fixture,coarseTexel){
  selectedTexels([coarseTexel]);const cx=coarseTexel%16,cy=Math.floor(coarseTexel/16),result=[];
  for(let y=0;y<4;y++)for(let x=0;x<4;x++){const offset=4*((4*cy+y)*64+4*cx+x);result.push(Array.from(fixture.fine.slice(offset,offset+4)));}
  return result;
}
export function buildTargets(fixture,pairs,numericBRDF,texels=null){
  const selected=selectedTexels(texels),output=new Float32Array(selected.length*pairs.length*4);
  selected.forEach((id,index)=>{const fine=fineFootprint(fixture,id);pairs.forEach((pair,j)=>{
    const sum=[0,0,0];for(const texel of fine){const rgb=numericBRDF(texel.slice(0,3),texel[3],pair.light,pair.view,METAL_F0,0);for(let k=0;k<3;k++)sum[k]+=rgb[k]/16;}
    if(sum.some(value=>!Number.isFinite(value)))throw Error('Nonfinite target response');
    output.set([...sum,1],4*(index*pairs.length+j));
  });});return output;
}
export function evaluateParameters(parameters,targets,pairs,numericBRDF){
  if(parameters.length%4!==0||targets.length!==parameters.length*pairs.length)throw Error('Parameter/target layout mismatch');
  let total=0,maxAbsolute=0;const perTexelMSE=[];
  for(let texel=0;texel<parameters.length/4;texel++){
    const normal=parametersToNormal(parameters,4*texel);let local=0;
    for(let j=0;j<pairs.length;j++){
      const rgb=numericBRDF(normal,parameters[4*texel+2],pairs[j].light,pairs[j].view,METAL_F0,0),offset=4*(texel*pairs.length+j);
      for(let k=0;k<3;k++){const error=rgb[k]-targets[offset+k];local+=error*error;maxAbsolute=Math.max(maxAbsolute,Math.abs(error));}
    }
    perTexelMSE.push(local/(pairs.length*3));total+=local;
  }
  const mse=total/(parameters.length/4*pairs.length*3);if(!Number.isFinite(mse))throw Error('Nonfinite material error');
  return {mse,rmse:Math.sqrt(mse),maxAbsolute,perTexelMSE};
}
export function roughnessGridBaseline(fixture,trainTargets,numericBRDF,texels=null){
  const selected=selectedTexels(texels),output=parametersAt(fixture.baselines.average,selected),chosenCandidate=[];
  if(trainTargets.length!==selected.length*fixture.trainPairs.length*4)throw Error('Training target layout mismatch');
  for(let i=0;i<selected.length;i++){
    const parameters=Array.from(output.slice(4*i,4*i+4));
    const target=trainTargets.slice(i*fixture.trainPairs.length*4,(i+1)*fixture.trainPairs.length*4);
    let best=Infinity,choice=0;
    for(let candidate=0;candidate<48;candidate++){
      parameters[2]=Math.fround(.08+.92*candidate/47);
      const error=evaluateParameters(parameters,target,fixture.trainPairs,numericBRDF).mse;
      if(error<best){best=error;choice=candidate;output[4*i+2]=parameters[2];}
    }
    chosenCandidate.push(choice);
  }
  return {parameters:output,chosenCandidate,candidates:48,selection:'Training RGB MSE only; first candidate wins ties; average-normal slopes remain fixed'};
}

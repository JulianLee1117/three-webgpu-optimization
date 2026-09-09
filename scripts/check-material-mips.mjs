/** Fixed CPU feasibility screen. No GPU, AD, rendering, tuning flags or retries. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {MIP_LIMITS,MATERIALS,VARIANCE_FILTER,createFixture,cpuSubset,parametersAt,buildTargets,evaluateParameters,roughnessGridBaseline} from '../experiments/material-mips/fixtures.mjs';
import {numericBRDF,SHADING_SOURCE_FILES,SHADING_SCOPE} from '../experiments/material-mips/shading.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const hash=value=>createHash('sha256').update(value).digest('hex');
const paths=['scripts/check-material-mips.mjs','experiments/material-mips/fixtures.mjs','experiments/material-mips/shading.js',...SHADING_SOURCE_FILES];
const sourceHashes=Object.fromEntries(await Promise.all(paths.map(async name=>[name,hash(await readFile(path.join(root,name)))])));
const texels=cpuSubset(),createdAt=new Date().toISOString();
const report={kind:'material-mips-cpu-feasibility-v1',createdAt,configuration:MIP_LIMITS,texels,sourceHashes,
  varianceFilter:VARIANCE_FILTER,shading:SHADING_SCOPE,method:'Per-texel double-precision central finite differences, 128 fixed full-batch Adam steps; initialization is the training-only 48-candidate roughness grid',
  caveats:['Eight predetermined coarse texels per fixture are a cheap feasibility screen, not full-map or GPU evidence.',
    'Fine normals, coarse initialization, directions and averaged targets are Float32; optimizer arithmetic is Float64.',
    'Three slopes/roughness parameters describe one isotropic lobe and cannot reproduce arbitrary anisotropy, multi-lobe mixtures, visibility or all lighting.',
    'The 30% held-out MSE gate is compared with the minimum held-out error among all three fixed baselines. No held-out observation enters training or initialization.',
    'Peak error is reported separately and is not a predeclared pass/fail gate.'],fixtures:[]};
assert.equal(process.argv.length,2,'No tuning options: run node scripts/check-material-mips.mjs');
for(const kind of Object.keys(MATERIALS)){
  const started=performance.now(),result={kind,status:'running',texels,optimizations:[]};
  try{
    const fixture=createFixture(kind),train=buildTargets(fixture,fixture.trainPairs,numericBRDF,texels),held=buildTargets(fixture,fixture.heldoutPairs,numericBRDF,texels);
    const average=parametersAt(fixture.baselines.average,texels),variance=parametersAt(fixture.baselines.variance,texels);
    const grid=roughnessGridBaseline(fixture,train,numericBRDF,texels),learned=Float64Array.from(grid.parameters);
    result.inputs={fine:Array.from(fixture.fine),trainPairs:fixture.trainPairs,heldoutPairs:fixture.heldoutPairs,trainTargets:Array.from(train),heldoutTargets:Array.from(held)};
    result.grid=grid;result.grid.parameters=Array.from(grid.parameters);
    result.baselines=Object.fromEntries(Object.entries({average,variance,roughnessGrid:grid.parameters}).map(([name,parameters])=>[name,
      {parameters:Array.from(parameters),train:evaluateParameters(parameters,train,fixture.trainPairs,numericBRDF),heldout:evaluateParameters(parameters,held,fixture.heldoutPairs,numericBRDF)}]));
    for(let index=0;index<texels.length;index++){
      const target=train.slice(index*32*4,(index+1)*32*4),w=Array.from(learned.slice(4*index,4*index+4));
      const m=[0,0,0],v=[0,0,0],g=[0,0,0],trace=[];
      const optimization={texel:texels[index],initial:[...w],completed:0,trace};result.optimizations.push(optimization);
      const objective=()=>evaluateParameters(w,target,fixture.trainPairs,numericBRDF).mse;
      for(let step=1;step<=MIP_LIMITS.cpuSteps;step++){
        if(performance.now()-started>MIP_LIMITS.cpuWallMs)throw Error('Predeclared CPU fixture wall budget exceeded');
        for(let parameter=0;parameter<3;parameter++){
          const original=w[parameter],h=MIP_LIMITS.cpuFDStep;
          w[parameter]=original+h;const plus=objective();w[parameter]=original-h;const minus=objective();w[parameter]=original;
          g[parameter]=(plus-minus)/(2*h);assert.ok(Number.isFinite(g[parameter]),'Nonfinite finite difference');
        }
        for(let parameter=0;parameter<3;parameter++){
          m[parameter]=.9*m[parameter]+.1*g[parameter];v[parameter]=.999*v[parameter]+.001*g[parameter]**2;
          const delta=MIP_LIMITS.learningRate*(m[parameter]/(1-.9**step))/(Math.sqrt(v[parameter]/(1-.999**step))+1e-8);
          const low=parameter===2?MIP_LIMITS.roughnessMin:-MIP_LIMITS.slopeLimit,high=parameter===2?MIP_LIMITS.roughnessMax:MIP_LIMITS.slopeLimit;
          w[parameter]=Math.max(low,Math.min(high,w[parameter]-delta));
        }
        optimization.completed=step;optimization.final=[...w];trace.push({step,trainMSE:objective(),parameters:[...w]});
      }
      learned.set(w,4*index);
    }
    result.learned={parameters:Array.from(learned),train:evaluateParameters(learned,train,fixture.trainPairs,numericBRDF),heldout:evaluateParameters(learned,held,fixture.heldoutPairs,numericBRDF)};
    const ranking=Object.entries(result.baselines).sort((a,b)=>a[1].heldout.mse-b[1].heldout.mse);
    result.strongestBaseline=ranking[0][0];result.strongestBaselineHeldoutMSE=ranking[0][1].heldout.mse;
    result.heldoutMSEImprovement=1-result.learned.heldout.mse/result.strongestBaselineHeldoutMSE;
    result.peakErrorChange=result.learned.heldout.maxAbsolute-ranking[0][1].heldout.maxAbsolute;
    result.passed=result.heldoutMSEImprovement>=MIP_LIMITS.minimumHeldoutMSEImprovement;
    result.status=result.passed?'passed':'failed-quality-gate';
  }catch(error){result.status='failed-error';result.passed=false;result.error={name:error.name,message:error.message};}
  result.cpuWallMs=performance.now()-started;report.fixtures.push(result);
  console.log(JSON.stringify({kind,status:result.status,baseline:result.strongestBaseline,baselineHeldoutMSE:result.strongestBaselineHeldoutMSE,
    learnedHeldoutMSE:result.learned?.heldout.mse,heldoutMSEImprovement:result.heldoutMSEImprovement,peakErrorChange:result.peakErrorChange,cpuWallMs:result.cpuWallMs,error:result.error}));
}
report.sourceUnchanged=(await Promise.all(paths.map(async name=>hash(await readFile(path.join(root,name)))===sourceHashes[name]))).every(Boolean);
report.status=report.sourceUnchanged&&report.fixtures.every(result=>result.passed)?'passed':'failed';
const output=path.join(root,'results/development/material-mips-cpu',createdAt.replaceAll(':','-'));await mkdir(output,{recursive:true});
const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(output,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(output,'report.sha256'),hash(bytes)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:report.status,report:path.relative(root,path.join(output,'report.json')).replaceAll('\\','/'),sha256:hash(bytes)}));
if(report.status!=='passed')process.exitCode=1;

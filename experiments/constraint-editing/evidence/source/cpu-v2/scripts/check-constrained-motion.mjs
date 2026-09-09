import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {numericProgram} from '../experiments/constraint-editing/programs.js';
import {validate, LIMITS} from '../experiments/constraint-editing/fixtures.mjs';
import {solve} from '../experiments/constraint-editing/solver.mjs';
import {solveConstrained,fixtureVariant,CASES,CONSTRAINED_LIMITS} from '../experiments/constraint-editing/solver-constrained.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const sourcePaths=['experiments/constraint-editing/programs.js','experiments/constraint-editing/fixtures.mjs','experiments/constraint-editing/solver.mjs','experiments/constraint-editing/PROTOCOL.md',
  'experiments/constraint-editing/solver-constrained.mjs','experiments/constraint-editing/CONSTRAINED-PROTOCOL.md','scripts/check-constrained-motion.mjs'];
const sourceHashes=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,sha(await fs.readFile(path.join(root,p)))])));
const started=new Date().toISOString(),directory=path.join(root,'results/development/constrained-motion-cpu',started.replaceAll(':','-'));
await fs.mkdir(directory,{recursive:true});
const report={kind:'constrained-motion-cpu-v2',started,status:'running',limits:LIMITS,constrainedLimits:CONSTRAINED_LIMITS,sourceHashes:await sourceHashes(),cells:[]};
const start=performance.now(),shouldStop=()=>performance.now()-start>CONSTRAINED_LIMITS.matrixOptimizerMs;
try {
  for(const selection of CASES) {
    const fixture=fixtureVariant(selection.kind,selection.variant),cell={...selection,fixture:{...fixture,samples:Array.from(fixture.samples)},witness:validate(fixture,fixture.witness),initial:validate(fixture,fixture.initial)};
    report.cells.push(cell);
    if(!cell.witness.pass){cell.status='invalid-witness';continue;}
    const oldTip=numericProgram(fixture.kind,fixture.target.p,fixture.target.time,fixture.initial),translation=fixture.target.position.map((x,k)=>x-oldTip[k]);
    cell.translation={translation,validation:validate(fixture,fixture.initial,{translation})};
    cell.handleOnly=await solve(fixture,{avoidObstacle:false,shouldStop});cell.handleOnly.validation=validate(fixture,cell.handleOnly.params);
    cell.constrained=await solveConstrained(fixture,{shouldStop});cell.constrained.validation=validate(fixture,cell.constrained.params);
    cell.constrained.float32Validation=validate(fixture,Array.from(new Float32Array(cell.constrained.params)));
    cell.gates={trainingFeasibility:cell.constrained.diagnostics.maximumViolation<=CONSTRAINED_LIMITS.feasibilityTolerance,
      quality:cell.constrained.validation.pass,float32Export:cell.constrained.float32Validation.pass,
      controlNecessity:cell.handleOnly.validation.minClearance<=-LIMITS.controlPenetration,
      bounded:!['stopped','wall-limit'].includes(cell.handleOnly.status)&&cell.constrained.status!=='stopped'};
    cell.status=Object.values(cell.gates).every(Boolean)?'passed':'failed';
    console.log(JSON.stringify({...selection,status:cell.status,gates:cell.gates,training:cell.constrained.diagnostics,
      heldout:cell.constrained.validation,controlClearance:cell.handleOnly.validation.minClearance,elapsedMs:cell.constrained.elapsedMs}));
  }
  report.sourceUnchanged=JSON.stringify(report.sourceHashes)===JSON.stringify(await sourceHashes());
  report.elapsedMs=performance.now()-start;
  report.status=report.sourceUnchanged&&report.elapsedMs<=20000&&report.cells.length===CASES.length&&report.cells.every(c=>c.status==='passed')?'passed':'failed';
} catch(error){report.status='error';report.failure=String(error?.message??error);}
report.finished=new Date().toISOString();
const bytes=Buffer.from(JSON.stringify(report,null,2)+'\n'),hash=sha(bytes);
await fs.writeFile(path.join(directory,'report.json'),bytes,{flag:'wx'});
await fs.writeFile(path.join(directory,'report.sha256'),`${hash}  report.json\n`,{flag:'wx'});
console.log(JSON.stringify({status:report.status,report:path.relative(root,path.join(directory,'report.json')).replaceAll('\\','/'),sha256:hash,elapsedMs:report.elapsedMs}));
if(report.status!=='passed')process.exitCode=1;

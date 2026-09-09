import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {makeFixture, KINDS, LIMITS, validate} from '../experiments/constraint-editing/fixtures.mjs';
import {numericProgram} from '../experiments/constraint-editing/programs.js';
import {solve} from '../experiments/constraint-editing/solver.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sourcePaths=['experiments/constraint-editing/programs.js','experiments/constraint-editing/fixtures.mjs','experiments/constraint-editing/solver.mjs','experiments/constraint-editing/PROTOCOL.md','scripts/check-constraint-editing.mjs'];
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const sources=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,hash(await fs.readFile(path.join(root,p)))])));
const started=new Date().toISOString(),directory=path.join(root,'results/development/constraint-editing-cpu',started.replaceAll(':','-'));
await fs.mkdir(directory,{recursive:true});
const report={kind:'constraint-editing-cpu-v1',started,limits:LIMITS,sourceHashes:await sources(),status:'running',cells:[],limitations:['Fixed synthetic feasibility witnesses; not a user study.','Sampled temporal collision checks; not continuous collision detection.','No GPU execution or AD performance comparison.']};
try {
  for(const kind of KINDS) {
    const fixture=makeFixture(kind),cell={kind,fixture:{...fixture,samples:Array.from(fixture.samples)},witness:validate(fixture,fixture.witness),initial:validate(fixture,fixture.initial)};
    report.cells.push(cell);
    if(!cell.witness.pass){cell.status='invalid-witness';console.log(JSON.stringify({kind,status:cell.status,witness:cell.witness}));continue;}
    const tip=numericProgram(kind,fixture.target.p,fixture.target.time,fixture.initial),translation=fixture.target.position.map((x,k)=>x-tip[k]);
    cell.translation={translation,validation:validate(fixture,fixture.initial,{translation})};
    cell.handleOnly=await solve(fixture,{avoidObstacle:false});cell.handleOnly.validation=validate(fixture,cell.handleOnly.params);
    cell.constrained=await solve(fixture);cell.constrained.validation=validate(fixture,cell.constrained.params);
    cell.gates={quality:cell.constrained.validation.pass,controlNecessity:cell.handleOnly.validation.minClearance<=-LIMITS.controlPenetration,
      bounded:!['wall-limit','stopped'].includes(cell.constrained.status)&&!['wall-limit','stopped'].includes(cell.handleOnly.status)};
    cell.status=Object.values(cell.gates).every(Boolean)?'passed':'failed';
    console.log(JSON.stringify({kind,status:cell.status,gates:cell.gates,handleOnly:cell.handleOnly.validation,constrained:cell.constrained.validation,elapsedMs:cell.handleOnly.elapsedMs+cell.constrained.elapsedMs}));
  }
  report.sourceUnchanged=JSON.stringify(report.sourceHashes)===JSON.stringify(await sources());
  report.status=report.sourceUnchanged&&report.cells.length===KINDS.length&&report.cells.every(c=>c.status==='passed')?'passed':'failed';
} catch(error) {report.status='error';report.failure=String(error?.message??error);}
report.finished=new Date().toISOString();
const bytes=Buffer.from(JSON.stringify(report,null,2)+'\n'),sha=hash(bytes);
await fs.writeFile(path.join(directory,'report.json'),bytes,{flag:'wx'});
await fs.writeFile(path.join(directory,'report.sha256'),`${sha}  report.json\n`,{flag:'wx'});
console.log(JSON.stringify({status:report.status,report:path.relative(root,path.join(directory,'report.json')).replaceAll('\\','/'),sha256:sha}));
if(report.status!=='passed')process.exitCode=1;

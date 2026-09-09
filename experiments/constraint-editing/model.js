import * as THREE from 'three/webgpu';
import {uniform} from 'three/tsl';
import {makeProgram} from './programs.js';
import {fixtureVariant} from './solver-constrained.mjs';
import {createProgramEvaluator} from './gpu.js';

export async function createExperiment(kind,variant='original') {
  const fixture=fixtureVariant(kind,variant),position=uniform(new THREE.Vector3()),time=uniform(0);
  const parameters=fixture.initial.map(v=>uniform(v));
  // The forward factory is called exactly once. All subsequent GPU evaluation,
  // finite differences and rendered deformations rebind this returned graph.
  const root=makeProgram(kind,position,time,parameters);
  const evaluator=await createProgramEvaluator({root,position,time,parameters});
  evaluator.load(fixture.initial);
  return {fixture,evaluator,root,inputs:{position,time,parameters},forwardFactoryCalls:1};
}

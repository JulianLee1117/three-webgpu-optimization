import { createIndexedGeometryFixtures } from '../scenes/geometry-fixtures.js';
import {
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  createImmediateAifPhase0Scenarios,
  validateImmediateAifPhase0Scenario,
} from './immediate-aif-plan.js';

const SCENARIO_IDS = Object.freeze(['v99', 'v20']);
const SCENARIO_SCALARS = Object.freeze([
  'scenarioId',
  'seed',
  'geometryTier',
  'objectCount',
  'bucketCount',
  'visibilityFraction',
  'expectedVisibleCount',
  'layout',
]);
const SCENARIO_ARRAYS = Object.freeze([
  'bucketCounts',
  'bucketBases',
  'visibleCounts',
  'objectBuckets',
  'cullOrder',
  'matrices',
  'bounds',
  'expectedVisibleIds',
]);

function exactBytes(left, right) {
  if (!ArrayBuffer.isView(left)
    || !ArrayBuffer.isView(right)
    || left.constructor !== right.constructor
    || left.byteLength !== right.byteLength) return false;
  const leftBytes = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const rightBytes = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  return leftBytes.every((value, index) => value === rightBytes[index]);
}

function exactJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function vectorExact(left, right) {
  return left !== undefined
    && right !== undefined
    && Object.is(left.x, right.x)
    && Object.is(left.y, right.y)
    && Object.is(left.z, right.z);
}

function attributeExact(actual, expected) {
  return actual?.itemSize === expected?.itemSize
    && actual?.count === expected?.count
    && actual?.normalized === expected?.normalized
    && actual?.usage === expected?.usage
    && actual?.gpuType === expected?.gpuType
    && exactBytes(actual?.array, expected?.array);
}

function morphAttributesExact(actual, expected) {
  const actualNames = Object.keys(actual ?? {}).sort();
  const expectedNames = Object.keys(expected ?? {}).sort();
  return exactJson(actualNames, expectedNames)
    && actualNames.every((name) => (
      Array.isArray(actual[name])
        && Array.isArray(expected[name])
        && actual[name].length === expected[name].length
        && actual[name].every((attribute, index) => (
          attributeExact(attribute, expected[name][index])
        ))
    ));
}

function geometryExact(actual, expected) {
  const actualNames = Object.keys(actual?.attributes ?? {}).sort();
  const expectedNames = Object.keys(expected?.attributes ?? {}).sort();
  return actual?.name === expected?.name
    && attributeExact(actual?.index, expected?.index)
    && exactJson(actualNames, expectedNames)
    && actualNames.every((name) => (
      attributeExact(actual.getAttribute(name), expected.getAttribute(name))
    ))
    && morphAttributesExact(actual?.morphAttributes, expected?.morphAttributes)
    && actual?.morphTargetsRelative === expected?.morphTargetsRelative
    && exactJson(actual?.groups, expected?.groups)
    && actual?.drawRange?.start === expected?.drawRange?.start
    && Object.is(actual?.drawRange?.count, expected?.drawRange?.count)
    && vectorExact(actual?.boundingBox?.min, expected?.boundingBox?.min)
    && vectorExact(actual?.boundingBox?.max, expected?.boundingBox?.max)
    && vectorExact(actual?.boundingSphere?.center, expected?.boundingSphere?.center)
    && Object.is(actual?.boundingSphere?.radius, expected?.boundingSphere?.radius);
}

function cloneScenario(scenario) {
  const clone = Object.fromEntries(SCENARIO_SCALARS.map((name) => [name, scenario[name]]));
  for (const name of SCENARIO_ARRAYS) clone[name] = scenario[name].slice();
  return Object.freeze(clone);
}

export function cloneImmediateAifPhase0ScenarioSet(scenarios) {
  return Object.freeze(Object.fromEntries(
    SCENARIO_IDS.map((scenarioId) => [scenarioId, cloneScenario(scenarios[scenarioId])]),
  ));
}

export function validateExactImmediateAifPhase0ScenarioSet(scenarios, canonicalScenarios) {
  if (!scenarios || !canonicalScenarios) {
    throw new TypeError('Exact Phase 0 validation requires observed and canonical scenarios.');
  }
  for (const scenarioId of SCENARIO_IDS) {
    const actual = scenarios[scenarioId];
    const expected = canonicalScenarios[scenarioId];
    validateImmediateAifPhase0Scenario(actual, expected);
    for (const name of SCENARIO_SCALARS) {
      if (!Object.is(actual[name], expected[name])) {
        throw new Error(`${scenarioId}.${name} differs from the canonical Phase 0 scenario.`);
      }
    }
    for (const name of SCENARIO_ARRAYS) {
      if (!exactBytes(actual[name], expected[name])) {
        throw new Error(`${scenarioId}.${name} bytes differ from the canonical Phase 0 scenario.`);
      }
    }
  }
  return true;
}

function validateExactGeometryFixtures(sourceGeometries, canonicalGeometries) {
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount) {
    throw new RangeError('Phase 0 requires exactly 32 ordered source geometries.');
  }
  let firstIndex = 0;
  const records = [];
  for (let bucket = 0; bucket < canonicalGeometries.length; bucket += 1) {
    const actual = sourceGeometries[bucket];
    const expected = canonicalGeometries[bucket];
    if (!geometryExact(actual, expected)) {
      throw new Error(
        `sourceGeometries[${bucket}] differs from the canonical medium Phase 0 fixture.`,
      );
    }
    records.push(Object.freeze({
      bucket,
      family: bucket % 4,
      name: actual.name,
      attributeNames: Object.keys(actual.attributes).sort(),
      vertexCount: actual.getAttribute('position').count,
      indexCount: actual.index.count,
      firstIndex,
      baseVertex: 0,
    }));
    firstIndex += actual.index.count;
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-exact-medium-geometry-validation',
    pass: true,
    tier: IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
    bucketCount: records.length,
    totalIndexCount: firstIndex,
    records: Object.freeze(records),
  });
}

/**
 * Regenerates the canonical workload and byte-compares caller inputs before
 * returning strategy-owned typed-array snapshots. Hashes are evidence only;
 * exact bytes are the acceptance gate.
 */
export function validateExactImmediateAifPhase0Workload({
  scenarios,
  sourceGeometries,
} = {}) {
  const canonicalGeometries = createIndexedGeometryFixtures(
    IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount,
    IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
  );
  let retained = false;
  try {
    const geometry = validateExactGeometryFixtures(sourceGeometries, canonicalGeometries);
    const canonicalScenarios = createImmediateAifPhase0Scenarios({
      geometrySpheres: canonicalGeometries.map((geometryFixture) => (
        geometryFixture.boundingSphere.clone()
      )),
    });
    const observedScenarios = scenarios ?? canonicalScenarios;
    validateExactImmediateAifPhase0ScenarioSet(observedScenarios, canonicalScenarios);
    const result = Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-exact-workload-validation',
      pass: true,
      geometry,
      ownedGeometries: canonicalGeometries,
      canonicalScenarios,
      ownedScenarios: cloneImmediateAifPhase0ScenarioSet(canonicalScenarios),
    });
    retained = true;
    return result;
  } finally {
    if (!retained) {
      for (const geometry of canonicalGeometries) geometry.dispose();
    }
  }
}

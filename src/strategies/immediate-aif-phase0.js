import { Group } from 'three/webgpu';
import { updateFrustumPlaneState } from '../culling/frustum-planes.js';
import { STORAGE_TRANSFORM_ADDRESS_MODES } from '../materials/storage-transform.js';
import {
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
  createImmediateAifAddressOracle,
  createImmediateAifCommandOracle,
  validateImmediateAifPhase0Lane,
  validateImmediateAifPhase0Scenario,
} from '../phase0/immediate-aif-plan.js';
import {
  cloneImmediateAifPhase0ScenarioSet,
  validateExactImmediateAifPhase0ScenarioSet,
  validateExactImmediateAifPhase0Workload,
} from '../phase0/immediate-aif-workload-validation.js';
import {
  createFixedSliceAddressLane,
  createFixedSliceSharedResources,
  readFixedSliceLaneSnapshot,
  validateFixedSliceLaneSnapshot,
} from './fixed-slice.js';

const [ATTRIBUTE_LANE, IMMEDIATE_LANE, FIRST_INSTANCE_LANE] =
  IMMEDIATE_AIF_PHASE0_LANES;

const ADDRESS_MODE_BY_LANE = Object.freeze({
  [ATTRIBUTE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  [IMMEDIATE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.IMMEDIATE_BASE,
  [FIRST_INSTANCE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
});

export const IMMEDIATE_AIF_PHASE0_STRATEGY_ID = 'immediate-aif-phase0';

export const IMMEDIATE_AIF_PHASE0_CORRECTNESS_STATES = Object.freeze({
  CONSTRUCTED: 'constructed',
  SCENARIO_LOADED: 'scenario-loaded',
  LIVE_VALIDATION: 'live-validation',
  LIVE_VALIDATED: 'live-validated',
  VISIBLE_IDS_FROZEN: 'visible-ids-frozen',
  PRIMING: 'priming',
  READY: 'ready',
  IMMEDIATE_RERECORD_REQUIRED: 'immediate-rerecord-required',
  IMMEDIATE_RERECORDING: 'immediate-rerecording',
  LANE_ORDER: 'lane-order',
  FAILED: 'failed',
  DISPOSED: 'disposed',
});

const STATE = IMMEDIATE_AIF_PHASE0_CORRECTNESS_STATES;
const CANONICAL_SCHEDULE = 'canonical';
const EXPECTED_SCENARIO_SEQUENCE = Object.freeze(['v99', 'v20']);
const EXPECTED_LANE_ORDER_COVERAGE_KEYS = Object.freeze(
  EXPECTED_SCENARIO_SEQUENCE.flatMap((scenarioId) => (
    IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.flatMap((scheduleId) => (
      IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map(
        (order) => `${scenarioId}/${scheduleId}/${order.join('')}`,
      )
    ))
  )),
);

function exactSequence(left, right) {
  return left?.length === right?.length
    && Array.from(right).every((value, index) => left[index] === value);
}

function freezeCopy(values) {
  return Object.freeze([...values]);
}

function hasExplicitCallbackPass(evidence) {
  return evidence !== null
    && typeof evidence === 'object'
    && !Array.isArray(evidence)
    && evidence.pass === true;
}

function validateLaneOrder(order, label = 'lane order') {
  if (!Array.isArray(order)
    || order.length !== IMMEDIATE_AIF_PHASE0_LANES.length
    || new Set(order).size !== IMMEDIATE_AIF_PHASE0_LANES.length
    || IMMEDIATE_AIF_PHASE0_LANES.some((lane) => !order.includes(lane))) {
    throw new RangeError(`${label} must be an exact A/I/F permutation.`);
  }
  if (!IMMEDIATE_AIF_PHASE0_LANE_ORDERS.some((candidate) => exactSequence(order, candidate))) {
    throw new RangeError(`${label} is not one of the six frozen Phase 0 orders.`);
  }
  return [...order];
}

function validateScenarioId(scenarioId) {
  if (scenarioId !== 'v99' && scenarioId !== 'v20') {
    throw new RangeError('scenarioId must be v99 or v20.');
  }
  return scenarioId;
}

function validateScheduleId(scheduleId) {
  if (!IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.includes(scheduleId)) {
    throw new RangeError('scheduleId must be canonical, S1, or S2.');
  }
  return scheduleId;
}

/**
 * Pure correctness-order guard for the renderer-dependent Phase 0 runtime.
 * It deliberately knows nothing about clocks, timestamps, or frame durations.
 */
export function createImmediateAifPhase0CorrectnessStateMachine() {
  let phase = STATE.CONSTRUCTED;
  let scenarioId = null;
  let scenarioSerial = 0;
  let activeScheduleId = CANONICAL_SCHEDULE;
  let immediateRecordedScheduleId = null;
  let validationOrder = null;
  let validationIndex = 0;
  let inFlightLane = null;
  let primeOrder = null;
  let primeIndex = 0;
  let activeLaneOrder = null;
  let laneOrderIndex = 0;
  let failure = null;
  const validatedLanes = [];
  const scenarioLoadSequence = [];
  const bundlePrimed = { A: false, I: false, F: false };
  const laneOrderCoverage = new Map();

  function requireHealthy(action) {
    if (phase === STATE.FAILED) {
      throw new Error(`${action} is unavailable after Phase 0 failed: ${failure}`);
    }
    if (phase === STATE.DISPOSED) {
      throw new Error(`${action} is unavailable after Phase 0 disposal.`);
    }
  }

  function requirePhase(expected, action) {
    requireHealthy(action);
    const accepted = Array.isArray(expected) ? expected : [expected];
    if (!accepted.includes(phase)) {
      throw new Error(`${action} requires state ${accepted.join(' or ')}, not ${phase}.`);
    }
  }

  function allBundlesPrimed() {
    return IMMEDIATE_AIF_PHASE0_LANES.every((lane) => bundlePrimed[lane]);
  }

  function coverageSnapshot() {
    return Object.freeze(Object.fromEntries(
      [...laneOrderCoverage.entries()].sort(([left], [right]) => left.localeCompare(right)),
    ));
  }

  function snapshot() {
    const laneOrderCoverage = coverageSnapshot();
    const scenarioSequenceExact = exactSequence(
      scenarioLoadSequence,
      EXPECTED_SCENARIO_SEQUENCE,
    );
    const laneOrderCoverageExact = Object.keys(laneOrderCoverage).length
      === EXPECTED_LANE_ORDER_COVERAGE_KEYS.length
      && EXPECTED_LANE_ORDER_COVERAGE_KEYS.every(
        (key) => laneOrderCoverage[key] === 1,
      );
    return Object.freeze({
      phase,
      scenarioId,
      scenarioSerial,
      activeScheduleId,
      immediateRecordedScheduleId,
      immediateRerecordRequired:
        phase === STATE.IMMEDIATE_RERECORD_REQUIRED
        || phase === STATE.IMMEDIATE_RERECORDING,
      validationOrder: validationOrder === null ? null : freezeCopy(validationOrder),
      validationIndex,
      validatedLanes: freezeCopy(validatedLanes),
      inFlightLane,
      primeOrder: primeOrder === null ? null : freezeCopy(primeOrder),
      primeIndex,
      bundlePrimed: Object.freeze({ ...bundlePrimed }),
      activeLaneOrder: activeLaneOrder === null ? null : freezeCopy(activeLaneOrder),
      laneOrderIndex,
      scenarioLoadSequence: freezeCopy(scenarioLoadSequence),
      expectedLaneOrderCoverageCount: EXPECTED_LANE_ORDER_COVERAGE_KEYS.length,
      laneOrderCoverage,
      scenarioSequenceExact,
      laneOrderCoverageExact,
      runtimeCoverageComplete: phase === STATE.READY
        && scenarioSequenceExact
        && laneOrderCoverageExact
        && scenarioId === EXPECTED_SCENARIO_SEQUENCE.at(-1)
        && activeScheduleId === CANONICAL_SCHEDULE
        && immediateRecordedScheduleId === CANONICAL_SCHEDULE,
      failure,
    });
  }

  function fail(reason) {
    if (phase === STATE.DISPOSED) return snapshot();
    failure = reason instanceof Error ? reason.message : String(reason);
    phase = STATE.FAILED;
    inFlightLane = null;
    return snapshot();
  }

  function loadScenario(nextScenarioId) {
    validateScenarioId(nextScenarioId);
    requirePhase([STATE.CONSTRUCTED, STATE.READY], 'Scenario load');
    if (phase === STATE.READY
      && (activeScheduleId !== CANONICAL_SCHEDULE
        || (allBundlesPrimed()
          && immediateRecordedScheduleId !== CANONICAL_SCHEDULE))) {
      throw new Error('A new scenario may load only after restoring the canonical schedule.');
    }
    scenarioId = nextScenarioId;
    scenarioLoadSequence.push(nextScenarioId);
    scenarioSerial += 1;
    activeScheduleId = CANONICAL_SCHEDULE;
    validationOrder = null;
    validationIndex = 0;
    inFlightLane = null;
    validatedLanes.length = 0;
    activeLaneOrder = null;
    laneOrderIndex = 0;
    phase = STATE.SCENARIO_LOADED;
    return snapshot();
  }

  function beginLiveValidation(order) {
    requirePhase(STATE.SCENARIO_LOADED, 'Live validation');
    validationOrder = validateLaneOrder(order, 'live validation order');
    validationIndex = 0;
    validatedLanes.length = 0;
    phase = STATE.LIVE_VALIDATION;
    return snapshot();
  }

  function beginLaneSubmission(lane) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.LIVE_VALIDATION, 'Lane submission');
    if (inFlightLane !== null) {
      throw new Error(`Lane ${inFlightLane} is already in flight.`);
    }
    const expected = validationOrder[validationIndex];
    if (lane !== expected) {
      throw new Error(`Live lane ${expected} must run next, not ${lane}.`);
    }
    inFlightLane = lane;
    return snapshot();
  }

  function finishLaneSubmission(lane, pass) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.LIVE_VALIDATION, 'Lane submission completion');
    if (inFlightLane !== lane) {
      throw new Error(`Cannot complete lane ${lane}; ${inFlightLane ?? 'no lane'} is in flight.`);
    }
    inFlightLane = null;
    if (pass !== true) return fail(`live ${lane} validation did not pass`);
    validatedLanes.push(lane);
    validationIndex += 1;
    if (validationIndex === validationOrder.length) phase = STATE.LIVE_VALIDATED;
    return snapshot();
  }

  function freezeVisibleIds(pass = true) {
    requirePhase(STATE.LIVE_VALIDATED, 'Visible-ID freeze');
    if (pass !== true) return fail('ordered visible-ID freeze did not pass');
    phase = allBundlesPrimed() ? STATE.READY : STATE.VISIBLE_IDS_FROZEN;
    return snapshot();
  }

  function beginPriming(order = IMMEDIATE_AIF_PHASE0_LANES) {
    requirePhase(STATE.VISIBLE_IDS_FROZEN, 'Bundle priming');
    if (Object.values(bundlePrimed).some(Boolean)) {
      throw new Error('Bundle priming is one-shot and cannot replace a primed lane bundle.');
    }
    primeOrder = validateLaneOrder(order, 'bundle prime order');
    primeIndex = 0;
    phase = STATE.PRIMING;
    return snapshot();
  }

  function beginBundlePrime(lane) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.PRIMING, 'Bundle lane prime');
    if (inFlightLane !== null) throw new Error(`Lane ${inFlightLane} is already in flight.`);
    const expected = primeOrder[primeIndex];
    if (lane !== expected) throw new Error(`Bundle lane ${expected} must prime next, not ${lane}.`);
    if (bundlePrimed[lane]) throw new Error(`Bundle lane ${lane} was already primed.`);
    inFlightLane = lane;
    return snapshot();
  }

  function finishBundlePrime(lane, pass) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.PRIMING, 'Bundle lane prime completion');
    if (inFlightLane !== lane) {
      throw new Error(`Cannot complete bundle lane ${lane}; ${inFlightLane ?? 'no lane'} is in flight.`);
    }
    inFlightLane = null;
    if (pass !== true) return fail(`bundle lane ${lane} did not record exactly once`);
    bundlePrimed[lane] = true;
    primeIndex += 1;
    if (primeIndex === primeOrder.length) {
      immediateRecordedScheduleId = activeScheduleId;
      phase = STATE.READY;
    }
    return snapshot();
  }

  function transitionSchedule(nextScheduleId) {
    validateScheduleId(nextScheduleId);
    requirePhase(STATE.READY, 'Schedule transition');
    if (nextScheduleId === activeScheduleId) {
      return Object.freeze({ changed: false, ...snapshot() });
    }
    if (!allBundlesPrimed()) throw new Error('Schedule transition requires all three bundles.');
    activeScheduleId = nextScheduleId;
    phase = STATE.IMMEDIATE_RERECORD_REQUIRED;
    return Object.freeze({ changed: true, ...snapshot() });
  }

  function beginImmediateRerecord() {
    requirePhase(STATE.IMMEDIATE_RERECORD_REQUIRED, 'Immediate bundle re-record');
    if (inFlightLane !== null) throw new Error(`Lane ${inFlightLane} is already in flight.`);
    inFlightLane = IMMEDIATE_LANE;
    phase = STATE.IMMEDIATE_RERECORDING;
    return snapshot();
  }

  function finishImmediateRerecord(pass) {
    requirePhase(STATE.IMMEDIATE_RERECORDING, 'Immediate bundle re-record completion');
    if (inFlightLane !== IMMEDIATE_LANE) {
      throw new Error('Only the I lane may satisfy an immediate bundle re-record.');
    }
    inFlightLane = null;
    if (pass !== true) return fail('I bundle did not re-record exactly once');
    immediateRecordedScheduleId = activeScheduleId;
    phase = STATE.READY;
    return snapshot();
  }

  function beginLaneOrder(order) {
    requirePhase(STATE.READY, 'Lane-order challenge');
    if (!allBundlesPrimed() || immediateRecordedScheduleId !== activeScheduleId) {
      throw new Error('Lane-order challenge requires three current, primed bundles.');
    }
    activeLaneOrder = validateLaneOrder(order, 'challenge lane order');
    laneOrderIndex = 0;
    phase = STATE.LANE_ORDER;
    return snapshot();
  }

  function beginLaneSelection(lane) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.LANE_ORDER, 'Lane selection');
    if (inFlightLane !== null) throw new Error(`Lane ${inFlightLane} is already in flight.`);
    const expected = activeLaneOrder[laneOrderIndex];
    if (lane !== expected) throw new Error(`Challenge lane ${expected} must run next, not ${lane}.`);
    inFlightLane = lane;
    return snapshot();
  }

  function finishLaneSelection(lane, pass) {
    validateImmediateAifPhase0Lane(lane);
    requirePhase(STATE.LANE_ORDER, 'Lane selection completion');
    if (inFlightLane !== lane) {
      throw new Error(`Cannot complete challenge lane ${lane}; ${inFlightLane ?? 'no lane'} is in flight.`);
    }
    inFlightLane = null;
    if (pass !== true) return fail(`challenge lane ${lane} did not pass`);
    laneOrderIndex += 1;
    if (laneOrderIndex === activeLaneOrder.length) {
      const key = `${scenarioId}/${activeScheduleId}/${activeLaneOrder.join('')}`;
      laneOrderCoverage.set(key, (laneOrderCoverage.get(key) ?? 0) + 1);
      activeLaneOrder = null;
      laneOrderIndex = 0;
      phase = STATE.READY;
    }
    return snapshot();
  }

  function dispose() {
    if (phase === STATE.DISPOSED) return snapshot();
    phase = STATE.DISPOSED;
    inFlightLane = null;
    return snapshot();
  }

  return Object.freeze({
    snapshot,
    fail,
    loadScenario,
    beginLiveValidation,
    beginLaneSubmission,
    finishLaneSubmission,
    freezeVisibleIds,
    beginPriming,
    beginBundlePrime,
    finishBundlePrime,
    transitionSchedule,
    beginImmediateRerecord,
    finishImmediateRerecord,
    beginLaneOrder,
    beginLaneSelection,
    finishLaneSelection,
    dispose,
  });
}

function requireObject(value, label) {
  if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
    throw new TypeError(`${label} must be an object.`);
  }
  return value;
}

function requireScenarioSet(scenarios) {
  if (!scenarios || typeof scenarios !== 'object') {
    throw new TypeError('Phase 0 scenarios must provide v99 and v20.');
  }
  for (const scenarioId of ['v99', 'v20']) {
    validateImmediateAifPhase0Scenario(scenarios[scenarioId]);
  }
  const topologyFields = ['objectBuckets', 'bucketBases', 'bucketCounts', 'cullOrder'];
  for (const field of topologyFields) {
    if (!exactSequence(scenarios.v99[field], scenarios.v20[field])) {
      throw new Error(`Phase 0 scenario topology differs at ${field}.`);
    }
  }
  return scenarios;
}

function markAttributeUpdated(attribute) {
  attribute.needsUpdate = true;
}

function copyAttribute(attribute, values, label) {
  if (!attribute?.array || attribute.array.constructor !== values.constructor
    || attribute.array.length !== values.length) {
    throw new Error(`${label} cannot replace the fixed Phase 0 buffer shape.`);
  }
  attribute.array.set(values);
  markAttributeUpdated(attribute);
}

function setVisibleIdsToSentinel(attribute) {
  if (!(attribute?.array instanceof Uint32Array)
    || attribute.array.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount) {
    throw new Error('Phase 0 visible-ID buffer has the wrong shape.');
  }
  attribute.array.fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
  markAttributeUpdated(attribute);
}

function setOverflowToSentinel(attribute) {
  if (!(attribute?.array instanceof Uint32Array) || attribute.array.length !== 1) {
    throw new Error('Phase 0 overflow buffer has the wrong shape.');
  }
  attribute.array[0] = IMMEDIATE_AIF_UNUSED_ADDRESS;
  markAttributeUpdated(attribute);
}

function copyCommands(lane, commands) {
  if (!(commands instanceof Uint32Array)
    || commands.byteLength !== IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
    || lane.indirectAttribute.array.length !== commands.length) {
    throw new Error(`${lane.id} command update is not the frozen 640-byte shape.`);
  }
  lane.indirectAttribute.array.set(commands);
  markAttributeUpdated(lane.indirectAttribute);
}

function setAttributeLaneBases(lane, sourceGeometries, bases) {
  const attribute = lane.geometry.getAttribute('bucketBase');
  if (!(attribute?.array instanceof Uint32Array)
    || bases.length !== sourceGeometries.length) {
    throw new Error('A lane bucket-base vertex stream is unavailable or malformed.');
  }
  let vertexCursor = 0;
  for (let draw = 0; draw < sourceGeometries.length; draw += 1) {
    const vertexCount = sourceGeometries[draw]?.getAttribute('position')?.count;
    if (!Number.isInteger(vertexCount) || vertexCount <= 0) {
      throw new Error(`Source geometry ${draw} has no nonempty position stream.`);
    }
    attribute.array.fill(bases[draw], vertexCursor, vertexCursor + vertexCount);
    vertexCursor += vertexCount;
  }
  if (vertexCursor !== attribute.array.length) {
    throw new Error('A lane bucket-base stream does not match merged vertex spans.');
  }
  markAttributeUpdated(attribute);
}

function paddingSentinelsExact(visibleIds, scenario) {
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const base = scenario.bucketBases[bucket];
    for (let slot = scenario.visibleCounts[bucket]; slot < scenario.bucketCounts[bucket]; slot += 1) {
      if (visibleIds[base + slot] !== IMMEDIATE_AIF_UNUSED_ADDRESS) return false;
    }
  }
  return true;
}

/**
 * Builds the correctness-only, three-lane Phase 0 runtime. The returned API
 * intentionally contains no benchmark timer or timestamp-registration seam.
 */
export function buildImmediateAifPhase0Strategy({
  renderer,
  camera,
  sourceGeometries,
  scenarios = null,
} = {}) {
  requireObject(renderer, 'renderer');
  requireObject(camera, 'camera');
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount) {
    throw new RangeError(
      `sourceGeometries must contain ${IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount} geometries.`,
    );
  }
  const exactWorkload = validateExactImmediateAifPhase0Workload({
    scenarios,
    sourceGeometries,
  });
  // From this point onward no caller-owned geometry or scenario object is
  // retained. In particular, later command-oracle rebuilds must not observe a
  // caller mutating index.count after the merged payload was constructed.
  const ownedSourceGeometries = exactWorkload.ownedGeometries;
  let constructionCompleted = false;
  let shared = null;
  let root = null;
  let machine = null;
  const partialLaneStates = {};
  try {
  const canonicalScenarioSet = exactWorkload.canonicalScenarios;
  const scenarioSet = requireScenarioSet(exactWorkload.ownedScenarios);
  const sharedScenario = cloneImmediateAifPhase0ScenarioSet(scenarioSet).v99;
  shared = createFixedSliceSharedResources(
    { scenario: sharedScenario, sourceGeometries: ownedSourceGeometries },
    { addressModes: [
      ADDRESS_MODE_BY_LANE.A,
      ADDRESS_MODE_BY_LANE.I,
      ADDRESS_MODE_BY_LANE.F,
    ] },
  );
  const canonicalImmediateBases = scenarioSet.v99.bucketBases.slice();
  partialLaneStates.A = createFixedSliceAddressLane(shared, {
      addressMode: ADDRESS_MODE_BY_LANE.A,
      id: 'immediate-aif-phase0-A',
    });
  partialLaneStates.I = createFixedSliceAddressLane(shared, {
      addressMode: ADDRESS_MODE_BY_LANE.I,
      id: 'immediate-aif-phase0-I',
      indirectImmediateBases: canonicalImmediateBases,
    });
  partialLaneStates.F = createFixedSliceAddressLane(shared, {
      addressMode: ADDRESS_MODE_BY_LANE.F,
      id: 'immediate-aif-phase0-F',
    });
  const laneStates = Object.freeze(partialLaneStates);
  root = new Group();
  root.name = 'immediate-aif-phase0-root';
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const state = laneStates[lane];
    if (state.commandLayout.commands.byteLength
      !== IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
      || state.meshes.length !== 1
      || state.root.static !== true) {
      throw new Error(`Lane ${lane} does not have the frozen Phase 0 topology.`);
    }
    state.root.visible = false;
    root.add(state.root);
  }
  if (new Set(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => laneStates[lane].indirectAttribute,
  )).size !== 3
    || new Set(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => laneStates[lane].geometry,
    )).size !== 3) {
    throw new Error('Phase 0 requires three private command buffers and geometry shells.');
  }

  machine = createImmediateAifPhase0CorrectnessStateMachine();
  const liveValidations = new Map();
  let activeScenario = null;
  let activeScenarioId = null;
  let activeLane = null;
  let visibleResetSerial = 0;
  let immediateSourceDetachmentSerial = 0;
  let frozenVisibleIds = null;
  let activeOracles = null;

  function buildOracles(scheduleId) {
    if (activeScenario === null) throw new Error('Phase 0 scenario has not been loaded.');
    return Object.freeze(Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane,
      Object.freeze({
        command: createImmediateAifCommandOracle({
          lane,
          scenario: activeScenario,
          sourceGeometries: ownedSourceGeometries,
          firstIndexes: shared.firstIndexes,
          scheduleId,
        }),
        address: createImmediateAifAddressOracle({
          lane,
          scenario: activeScenario,
          scheduleId,
        }),
      }),
    ])));
  }

  function selectActiveLane(lane) {
    validateImmediateAifPhase0Lane(lane);
    for (const candidate of IMMEDIATE_AIF_PHASE0_LANES) {
      laneStates[candidate].root.visible = candidate === lane;
    }
    activeLane = lane;
  }

  function resetVisibleIds(reason) {
    setVisibleIdsToSentinel(shared.attributes.visibleIds);
    visibleResetSerial += 1;
    return Object.freeze({
      kind: 'immediate-aif-phase0-full-visible-id-sentinel-reset',
      reason,
      serial: visibleResetSerial,
      sentinel: IMMEDIATE_AIF_UNUSED_ADDRESS,
      valueCount: shared.attributes.visibleIds.array.length,
      byteLength: shared.attributes.visibleIds.array.byteLength,
      attributeId: shared.attributes.visibleIds.id,
      attributeVersion: shared.attributes.visibleIds.version,
    });
  }

  function loadScenario(scenarioId) {
    validateScenarioId(scenarioId);
    validateExactImmediateAifPhase0ScenarioSet(scenarioSet, canonicalScenarioSet);
    machine.loadScenario(scenarioId);
    try {
      activeScenarioId = scenarioId;
      activeScenario = scenarioSet[scenarioId];
      copyAttribute(shared.attributes.matrix, activeScenario.matrices, 'matrix buffer');
      copyAttribute(shared.attributes.bounds, activeScenario.bounds, 'bounds buffer');
      copyAttribute(
        shared.attributes.objectBucket,
        activeScenario.objectBuckets,
        'object-bucket buffer',
      );
      copyAttribute(shared.attributes.bucketBase, activeScenario.bucketBases, 'bucket-base buffer');
      copyAttribute(
        shared.attributes.bucketCapacity,
        activeScenario.bucketCounts,
        'bucket-capacity buffer',
      );
      copyAttribute(shared.attributes.cullOrder, activeScenario.cullOrder, 'cull-order buffer');
      const visibleReset = resetVisibleIds(`load-${scenarioId}`);
      setOverflowToSentinel(shared.attributes.overflow);
      activeOracles = buildOracles(CANONICAL_SCHEDULE);
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        copyCommands(laneStates[lane], activeOracles[lane].command.initialCommands);
      }
      setAttributeLaneBases(
        laneStates.A,
        ownedSourceGeometries,
        activeOracles.A.command.sourceBaseByDraw,
      );
      laneStates.I.indirectImmediateBases.set(
        activeOracles.I.command.sourceBaseByDraw,
      );
      liveValidations.clear();
      frozenVisibleIds = null;
      activeLane = null;
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) laneStates[lane].root.visible = false;
      return Object.freeze({
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-scenario-load',
        scenarioId,
        visibilityFraction: activeScenario.visibilityFraction,
        expectedVisibleCount: activeScenario.expectedVisibleCount,
        visibleReset,
        state: machine.snapshot(),
      });
    } catch (error) {
      machine.fail(error);
      throw error;
    }
  }

  function beginLiveValidation(order) {
    liveValidations.clear();
    return machine.beginLiveValidation(order);
  }

  async function submitAndValidateLane(
    lane,
    { renderer: activeRenderer = renderer, camera: activeCamera = camera } = {},
  ) {
    validateImmediateAifPhase0Lane(lane);
    if (activeRenderer !== renderer || activeCamera !== camera) {
      throw new Error('Phase 0 live validation requires the construction renderer and camera.');
    }
    if (typeof activeRenderer.compute !== 'function'
      || typeof activeRenderer.getArrayBufferAsync !== 'function') {
      throw new TypeError('Phase 0 live validation requires compute and GPU readback APIs.');
    }
    machine.beginLaneSubmission(lane);
    try {
      const laneState = laneStates[lane];
      const visibleReset = resetVisibleIds(`pre-live-${activeScenarioId}-${lane}`);
      setOverflowToSentinel(shared.attributes.overflow);
      copyCommands(laneState, activeOracles[lane].command.initialCommands);
      updateFrustumPlaneState(shared.planeState, activeCamera, activeRenderer);
      activeRenderer.compute(laneState.computeNodes);
      const snapshot = await readFixedSliceLaneSnapshot(activeRenderer, shared, laneState);
      const correctness = await validateFixedSliceLaneSnapshot({
        shared,
        lane: laneState,
        expectedIds: activeScenario.expectedVisibleIds,
        snapshot,
      });
      const commandOracle = activeOracles[lane].command;
      const addressOracle = activeOracles[lane].address;
      const commandExact = exactSequence(snapshot.commands, commandOracle.postCullCommands);
      const observedCanonicalOrder = exactSequence(
        snapshot.visibleIds,
        addressOracle.canonicalVisibleIds,
      );
      const paddingExact = paddingSentinelsExact(snapshot.visibleIds, activeScenario);
      const pass = correctness.pass === true
        && commandExact
        && paddingExact
        && snapshot.overflow === 0;
      const evidence = Object.freeze({
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-live-lane-validation',
        pass,
        scenarioId: activeScenarioId,
        lane,
        scheduleId: CANONICAL_SCHEDULE,
        visibleReset,
        commandExact,
        // Atomic compaction order is deliberately diagnostic only. WebGPU does
        // not guarantee invocation/workgroup execution order, so the live gate
        // validates membership and uploads one deterministic packing below.
        observedCanonicalOrder,
        paddingSentinelsExact: paddingExact,
        overflow: snapshot.overflow,
        correctness,
        snapshot: Object.freeze({
          commands: snapshot.commands.slice(),
          visibleIds: snapshot.visibleIds.slice(),
          overflow: snapshot.overflow,
        }),
      });
      liveValidations.set(lane, evidence);
      machine.finishLaneSubmission(lane, pass);
      return evidence;
    } catch (error) {
      machine.fail(error);
      throw error;
    }
  }

  function freezeValidatedVisibleIds() {
    const state = machine.snapshot();
    if (state.phase !== STATE.LIVE_VALIDATED) {
      throw new Error(`Visible-ID freeze requires state ${STATE.LIVE_VALIDATED}, not ${state.phase}.`);
    }
    const ordered = state.validationOrder.map((lane) => liveValidations.get(lane));
    const allLiveMembershipValidationsPass = ordered.length === IMMEDIATE_AIF_PHASE0_LANES.length
      && ordered.every((evidence) => evidence?.pass === true
        && evidence.correctness?.membership?.pass === true
        && evidence.correctness?.membershipDigests?.pass === true
        && evidence.commandExact === true
        && evidence.paddingSentinelsExact === true
        && evidence.overflow === 0);
    const canonicalPacking = activeOracles.A.address.canonicalVisibleIds;
    const canonicalPackingExact = canonicalPacking instanceof Uint32Array
      && canonicalPacking.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount;
    const pass = allLiveMembershipValidationsPass && canonicalPackingExact;
    if (pass) {
      frozenVisibleIds = canonicalPacking.slice();
      copyAttribute(
        shared.attributes.visibleIds,
        frozenVisibleIds,
        'frozen visible-ID buffer',
      );
    }
    machine.freezeVisibleIds(pass);
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-ordered-visible-id-freeze',
      pass,
      scenarioId: activeScenarioId,
      source: pass ? 'deterministic-cpu-canonical-packing' : null,
      sourceLane: null,
      allLiveMembershipValidationsPass,
      canonicalPackingExact,
      observedLiveOrdersEqual: ordered.length === IMMEDIATE_AIF_PHASE0_LANES.length
        && ordered.every((evidence) => evidence?.snapshot?.visibleIds instanceof Uint32Array)
        && ordered.slice(1).every((evidence) => exactSequence(
          evidence.snapshot.visibleIds,
          ordered[0].snapshot.visibleIds,
        )),
      valueCount: canonicalPacking?.length ?? null,
      byteLength: canonicalPacking?.byteLength ?? null,
      visibleIds: pass ? frozenVisibleIds.slice() : null,
      state: machine.snapshot(),
    });
  }

  function bundleCounts() {
    return Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => [lane, laneStates[lane].bundleRecordCallbackCount],
    ));
  }

  function bundleUuids() {
    return Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => [lane, laneStates[lane].root.uuid],
    ));
  }

  async function primeBundles({
    renderLane,
    order = IMMEDIATE_AIF_PHASE0_LANES,
  } = {}) {
    if (typeof renderLane !== 'function') {
      throw new TypeError('Bundle priming requires a renderLane callback.');
    }
    const resolvedOrder = validateLaneOrder(order, 'bundle prime order');
    const initialCounts = bundleCounts();
    if (IMMEDIATE_AIF_PHASE0_LANES.some((lane) => initialCounts[lane] !== 0)) {
      machine.fail('bundle priming did not begin from zero record callbacks');
      throw new Error('Bundle priming requires zero prior lane record callbacks.');
    }
    machine.beginPriming(resolvedOrder);
    const identities = bundleUuids();
    const records = [];
    try {
      for (const lane of resolvedOrder) {
        machine.beginBundlePrime(lane);
        selectActiveLane(lane);
        const before = bundleCounts();
        const callbackEvidence = await renderLane(getLaneExecutionContext(lane));
        const after = bundleCounts();
        const pass = after[lane] === before[lane] + 1
          && IMMEDIATE_AIF_PHASE0_LANES.every(
            (candidate) => candidate === lane || after[candidate] === before[candidate],
          )
          && laneStates[lane].root.uuid === identities[lane]
          && hasExplicitCallbackPass(callbackEvidence);
        records.push(Object.freeze({ lane, pass, before, after, callback: callbackEvidence ?? null }));
        machine.finishBundlePrime(lane, pass);
        if (!pass) break;
      }
    } catch (error) {
      machine.fail(error);
      throw error;
    }
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-bundle-prime',
      pass: records.length === resolvedOrder.length && records.every((record) => record.pass),
      order: freezeCopy(resolvedOrder),
      initialCounts: Object.freeze(initialCounts),
      records: Object.freeze(records),
      bundleUuids: Object.freeze(identities),
      state: machine.snapshot(),
    });
  }

  function transitionSchedule(scheduleId) {
    validateScheduleId(scheduleId);
    const beforeState = machine.snapshot();
    const transition = machine.transitionSchedule(scheduleId);
    if (!transition.changed) {
      return Object.freeze({
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-schedule-transition',
        pass: true,
        changed: false,
        fromScheduleId: scheduleId,
        toScheduleId: scheduleId,
        state: machine.snapshot(),
      });
    }
    try {
      const nextOracles = buildOracles(scheduleId);
      const before = {
        aAttributeVersion: laneStates.A.geometry.getAttribute('bucketBase').version,
        iBundleVersion: laneStates.I.root.version,
        fCommandVersion: laneStates.F.indirectAttribute.version,
        bundleCounts: bundleCounts(),
      };
      setAttributeLaneBases(
        laneStates.A,
        ownedSourceGeometries,
        nextOracles.A.command.sourceBaseByDraw,
      );
      const priorImmediateSource = laneStates.I.indirectImmediateBases;
      const immediateSourceWasDetached = priorImmediateSource.byteLength === 0;
      if (immediateSourceWasDetached) {
        const replacement = nextOracles.I.command.sourceBaseByDraw.slice();
        laneStates.I.geometry.setIndirect(
          laneStates.I.indirectAttribute,
          Array.from(laneStates.I.commandLayout.offsets),
          replacement,
        );
        laneStates.I.indirectImmediateBases = replacement;
      } else {
        priorImmediateSource.set(nextOracles.I.command.sourceBaseByDraw);
      }
      copyCommands(laneStates.F, nextOracles.F.command.postCullCommands);
      laneStates.I.root.needsUpdate = true;
      activeOracles = nextOracles;
      const after = {
        aAttributeVersion: laneStates.A.geometry.getAttribute('bucketBase').version,
        iBundleVersion: laneStates.I.root.version,
        fCommandVersion: laneStates.F.indirectAttribute.version,
        bundleCounts: bundleCounts(),
      };
      const pass = after.aAttributeVersion === before.aAttributeVersion + 1
        && after.iBundleVersion === before.iBundleVersion + 1
        && after.fCommandVersion === before.fCommandVersion + 1
        && exactSequence(
          laneStates.I.indirectImmediateBases,
          nextOracles.I.command.sourceBaseByDraw,
        )
        && exactSequence(
          laneStates.F.indirectAttribute.array,
          nextOracles.F.command.postCullCommands,
        )
        && IMMEDIATE_AIF_PHASE0_LANES.every(
          (lane) => after.bundleCounts[lane] === before.bundleCounts[lane],
        );
      if (!pass) machine.fail(`schedule ${scheduleId} resource transition failed`);
      return Object.freeze({
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-schedule-transition',
        pass,
        changed: true,
        fromScheduleId: beforeState.activeScheduleId,
        toScheduleId: scheduleId,
        immediateRerecordRequired: true,
        immediateSourceWasDetached,
        immediateSourceReplaced: immediateSourceWasDetached,
        before: Object.freeze(before),
        after: Object.freeze(after),
        state: machine.snapshot(),
      });
    } catch (error) {
      machine.fail(error);
      throw error;
    }
  }

  function mutateAndDetachImmediateBaseSource(mutatedBases) {
    const state = machine.snapshot();
    const insideImmediateRerecord = state.phase === STATE.IMMEDIATE_RERECORDING
      && state.inFlightLane === IMMEDIATE_LANE;
    if (!insideImmediateRerecord) {
      throw new Error(
        'Immediate-source detachment is allowed only from the native I bundle finish hook.',
      );
    }
    if (!(mutatedBases instanceof Uint32Array)
      || mutatedBases.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount) {
      throw new RangeError(
        `mutatedBases must be a Uint32Array of length ${IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount}.`,
      );
    }
    const source = laneStates.I.indirectImmediateBases;
    if (!(source instanceof Uint32Array)
      || source.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
      || source.byteLength === 0) {
      throw new Error('The active I immediate source is unavailable or already detached.');
    }
    const recordedBases = source.slice();
    if (exactSequence(recordedBases, mutatedBases)) {
      throw new Error('The post-finish mutation must change the I immediate source.');
    }
    source.set(mutatedBases);
    const mutatedCopy = source.slice();
    structuredClone(source, { transfer: [source.buffer] });
    if (source.byteLength !== 0 || source.length !== 0) {
      throw new Error('The I immediate source did not detach exactly.');
    }
    immediateSourceDetachmentSerial += 1;
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-post-finish-source-detachment',
      pass: true,
      scenarioId: activeScenarioId,
      scheduleId: state.activeScheduleId,
      detachedDuringImmediateRerecord: insideImmediateRerecord,
      serial: immediateSourceDetachmentSerial,
      recordedBases,
      mutatedBases: mutatedCopy,
      detachedByteLength: source.byteLength,
      detachedLength: source.length,
      state: machine.snapshot(),
    });
  }

  async function rerecordImmediateBundle({ renderLane } = {}) {
    if (typeof renderLane !== 'function') {
      throw new TypeError('I bundle re-record requires a renderLane callback.');
    }
    machine.beginImmediateRerecord();
    selectActiveLane(IMMEDIATE_LANE);
    const identity = laneStates.I.root.uuid;
    const version = laneStates.I.root.version;
    const before = bundleCounts();
    try {
      const callbackEvidence = await renderLane(getLaneExecutionContext(IMMEDIATE_LANE));
      const after = bundleCounts();
      const pass = laneStates.I.root.uuid === identity
        && laneStates.I.root.version === version
        && after.I === before.I + 1
        && after.A === before.A
        && after.F === before.F
        && hasExplicitCallbackPass(callbackEvidence);
      machine.finishImmediateRerecord(pass);
      return Object.freeze({
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-immediate-bundle-rerecord',
        pass,
        lane: IMMEDIATE_LANE,
        scheduleId: machine.snapshot().activeScheduleId,
        sameBundleIdentity: laneStates.I.root.uuid === identity,
        sameBundleVersion: laneStates.I.root.version === version,
        before,
        after,
        callback: callbackEvidence ?? null,
        state: machine.snapshot(),
      });
    } catch (error) {
      machine.fail(error);
      throw error;
    }
  }

  function getLaneExecutionContext(lane) {
    validateImmediateAifPhase0Lane(lane);
    if (activeOracles === null || frozenVisibleIds === null) {
      throw new Error('Lane execution requires a frozen Phase 0 survivor buffer.');
    }
    return Object.freeze({
      lane,
      scenarioId: activeScenarioId,
      scheduleId: machine.snapshot().activeScheduleId,
      root: laneStates[lane].root,
      mesh: laneStates[lane].meshes[0],
      geometry: laneStates[lane].geometry,
      material: laneStates[lane].material,
      indirectAttribute: laneStates[lane].indirectAttribute,
      visibleIdsAttribute: shared.attributes.visibleIds,
      commandOracle: activeOracles[lane].command,
      addressOracle: activeOracles[lane].address,
    });
  }

  async function runLaneOrder(order, { renderLane } = {}) {
    if (typeof renderLane !== 'function') {
      throw new TypeError('Lane-order challenge requires a renderLane callback.');
    }
    const resolvedOrder = validateLaneOrder(order, 'challenge lane order');
    machine.beginLaneOrder(resolvedOrder);
    const stableUuids = bundleUuids();
    const stableRecordCounts = bundleCounts();
    const records = [];
    try {
      for (const lane of resolvedOrder) {
        machine.beginLaneSelection(lane);
        selectActiveLane(lane);
        const callbackEvidence = await renderLane(getLaneExecutionContext(lane));
        const observedCounts = bundleCounts();
        const bundleStable = IMMEDIATE_AIF_PHASE0_LANES.every(
          (candidate) => laneStates[candidate].root.uuid === stableUuids[candidate]
            && observedCounts[candidate] === stableRecordCounts[candidate],
        );
        const pass = bundleStable && hasExplicitCallbackPass(callbackEvidence);
        records.push(Object.freeze({
          lane,
          pass,
          bundleStable,
          callback: callbackEvidence ?? null,
        }));
        machine.finishLaneSelection(lane, pass);
        if (!pass) break;
      }
    } catch (error) {
      machine.fail(error);
      throw error;
    }
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-lane-order-challenge',
      pass: records.length === resolvedOrder.length && records.every((record) => record.pass),
      scenarioId: activeScenarioId,
      scheduleId: machine.snapshot().activeScheduleId,
      order: freezeCopy(resolvedOrder),
      records: Object.freeze(records),
      state: machine.snapshot(),
    });
  }

  async function runAllLaneOrders({ renderLane } = {}) {
    const results = [];
    for (const order of IMMEDIATE_AIF_PHASE0_LANE_ORDERS) {
      results.push(await runLaneOrder(order, { renderLane }));
    }
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-six-lane-orders',
      pass: results.length === 6 && results.every((result) => result.pass),
      scenarioId: activeScenarioId,
      scheduleId: machine.snapshot().activeScheduleId,
      results: Object.freeze(results),
      state: machine.snapshot(),
    });
  }

  function getLaneValidationResources(lane) {
    validateImmediateAifPhase0Lane(lane);
    return Object.freeze({
      ...getLaneExecutionContext(lane),
      overflowAttribute: shared.attributes.overflow,
      matrixAttribute: shared.attributes.matrix,
      commandByteLength: laneStates[lane].indirectAttribute.array.byteLength,
      commandOffsets: freezeCopy(laneStates[lane].commandLayout.offsets),
      latestLiveValidation: liveValidations.get(lane) ?? null,
    });
  }

  function diagnostics() {
    const commandAttributes = IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => laneStates[lane].indirectAttribute,
    );
    const geometryShells = IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => laneStates[lane].geometry,
    );
    const activeRoots = IMMEDIATE_AIF_PHASE0_LANES.filter(
      (lane) => laneStates[lane].root.visible,
    );
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-correctness-runtime',
      pass: commandAttributes.every(
        (attribute) => attribute.array.byteLength
          === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength,
      )
        && new Set(commandAttributes).size === 3
        && new Set(geometryShells).size === 3
        && IMMEDIATE_AIF_PHASE0_LANES.every(
          (lane) => laneStates[lane].root.static === true
            && laneStates[lane].meshes.length === 1,
        )
        && activeRoots.length <= 1,
      state: machine.snapshot(),
      activeLane,
      activeRootCount: activeRoots.length,
      visibleResetSerial,
      immediateSourceDetachmentSerial,
      frozenVisibleIds: frozenVisibleIds !== null,
      sharedVisibleIdsAttributeId: shared.attributes.visibleIds.id,
      commandBufferIds: Object.freeze(Object.fromEntries(
        IMMEDIATE_AIF_PHASE0_LANES.map(
          (lane) => [lane, laneStates[lane].indirectAttribute.id],
        ),
      )),
      commandBufferByteLengths: Object.freeze(Object.fromEntries(
        IMMEDIATE_AIF_PHASE0_LANES.map(
          (lane) => [lane, laneStates[lane].indirectAttribute.array.byteLength],
        ),
      )),
      geometryUuids: Object.freeze(Object.fromEntries(
        IMMEDIATE_AIF_PHASE0_LANES.map(
          (lane) => [lane, laneStates[lane].geometry.uuid],
        ),
      )),
      bundleUuids: Object.freeze(bundleUuids()),
      bundleVersions: Object.freeze(Object.fromEntries(
        IMMEDIATE_AIF_PHASE0_LANES.map(
          (lane) => [lane, laneStates[lane].root.version],
        ),
      )),
      bundleRecordCounts: Object.freeze(bundleCounts()),
      bucketBaseVertexStreams: Object.freeze(Object.fromEntries(
        IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
          lane,
          laneStates[lane].geometry.getAttribute('bucketBase') !== undefined,
        ]),
      )),
      configuredDrawCommands: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount,
      configuredRenderObjectsPerLane: 1,
      configuredComputeDispatchesPerLane: 2,
      analysisEligible: false,
      numericalDecision: null,
      runtimeCoverageComplete: machine.snapshot().runtimeCoverageComplete,
      exactWorkload: {
        schemaVersion: exactWorkload.schemaVersion,
        kind: exactWorkload.kind,
        pass: exactWorkload.pass,
        geometry: exactWorkload.geometry,
      },
    });
  }

  function dispose() {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      laneStates[lane].geometry.setIndirect(null);
      laneStates[lane].root.removeFromParent();
    }
    root.removeFromParent();
    for (const geometry of ownedSourceGeometries) geometry.dispose();
    machine.dispose();
  }

  const publicSharedResources = Object.freeze({
    kind: shared.kind,
    // Public copies may be inspected or even mutated without reaching the
    // private oracle inputs retained by the strategy closure.
    firstIndexes: shared.firstIndexes.slice(),
    commandCapacities: shared.commandCapacities.slice(),
    commandRecordCount: shared.commandRecordCount,
    geometriesByAddressMode: shared.geometriesByAddressMode,
    ownedGeometries: shared.ownedGeometries,
    attributes: shared.attributes,
    storageAttributes: shared.storageAttributes,
    storageNodes: shared.storageNodes,
    drawStruct: shared.drawStruct,
    planeState: shared.planeState,
  });

  const strategy = {
    id: IMMEDIATE_AIF_PHASE0_STRATEGY_ID,
    root,
    geometries: shared.ownedGeometries,
    materials: IMMEDIATE_AIF_PHASE0_LANES.map((lane) => laneStates[lane].material),
    storageAttributes: [
      ...shared.storageAttributes,
      ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => laneStates[lane].indirectAttribute),
    ],
    computeNodes: IMMEDIATE_AIF_PHASE0_LANES.flatMap(
      (lane) => laneStates[lane].computeNodes,
    ),
    usesCompute: true,
    configuredDrawCommands: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    laneIds: [...IMMEDIATE_AIF_PHASE0_LANES],
    laneStates,
    sharedResources: publicSharedResources,
    exactWorkload: Object.freeze({
      schemaVersion: exactWorkload.schemaVersion,
      kind: exactWorkload.kind,
      pass: exactWorkload.pass,
      geometry: exactWorkload.geometry,
    }),
    get scenarios() {
      return cloneImmediateAifPhase0ScenarioSet(scenarioSet);
    },
    get activeLane() {
      return activeLane;
    },
    get activeScenarioId() {
      return activeScenarioId;
    },
    get state() {
      return machine.snapshot();
    },
    get frozenVisibleIds() {
      return frozenVisibleIds?.slice() ?? null;
    },
    selectActiveLane,
    loadScenario,
    beginLiveValidation,
    submitAndValidateLane,
    freezeValidatedVisibleIds,
    primeBundles,
    transitionSchedule,
    mutateAndDetachImmediateBaseSource,
    rerecordImmediateBundle,
    runLaneOrder,
    runAllLaneOrders,
    getLaneValidationResources,
    diagnostics,
    dispose,
  };
  constructionCompleted = true;
  return strategy;
  } finally {
    if (!constructionCompleted) {
      const partialLanes = Object.values(partialLaneStates);
      for (const lane of partialLanes) {
        try { lane.geometry?.setIndirect?.(null); } catch {}
        try { lane.root?.removeFromParent?.(); } catch {}
      }
      try { root?.removeFromParent?.(); } catch {}
      for (const node of new Set(partialLanes.flatMap((lane) => lane.computeNodes ?? []))) {
        try { node?.dispose?.(); } catch {}
      }
      for (const material of new Set(partialLanes.map((lane) => lane.material))) {
        try { material?.dispose?.(); } catch {}
      }
      for (const attribute of new Set([
        ...(shared?.storageAttributes ?? []),
        ...partialLanes.map((lane) => lane.indirectAttribute),
      ])) {
        try { attribute?.dispose?.(); } catch {}
      }
      for (const geometry of new Set(shared?.ownedGeometries ?? [])) {
        try { geometry?.dispose?.(); } catch {}
      }
      try { machine?.dispose?.(); } catch {}
      for (const geometry of ownedSourceGeometries) geometry.dispose();
    }
  }
}

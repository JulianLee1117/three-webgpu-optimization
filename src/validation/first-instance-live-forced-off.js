import { Scene } from 'three/webgpu';
import {
  buildFixedSliceDeploymentStrategy,
  readFixedSliceLaneSnapshot,
  selectFixedSliceDeployment,
  validateFixedSliceLaneSnapshot,
} from '../strategies/fixed-slice.js';
import { disposeStrategyResources } from '../strategies/resources.js';

const PORTABLE_LANE = 'portable';

function skippedGate(kind, callbackName) {
  return {
    kind,
    pass: false,
    available: false,
    reason: `${callbackName} was not supplied.`,
  };
}

function inspectPortableConstruction(strategy, scenario) {
  const geometry = strategy?.geometries?.[0];
  const geometryModes = Object.keys(
    strategy?.sharedResources?.geometriesByAddressMode ?? {},
  );
  const pass = strategy?.id === 'fixed-slice'
    && strategy?.laneState?.lane === PORTABLE_LANE
    && strategy?.materials?.length === 1
    && strategy?.geometries?.length === 1
    && strategy?.computeNodes?.length === 2
    && geometry?.getAttribute?.('bucketBase') !== undefined
    && geometryModes.length === 1
    && geometryModes[0] === 'bucket-base'
    && strategy?.laneState?.addressMode === 'bucket-base'
    && strategy?.configuredDrawCommands === scenario.bucketCount
    && strategy?.configuredComputeDispatches === 2
    && strategy?.configuredComputeSubmissions === 1;
  return {
    pass,
    selectedStrategyId: strategy?.id ?? null,
    constructedLane: strategy?.laneState?.lane ?? null,
    addressMode: strategy?.laneState?.addressMode ?? null,
    materialCount: strategy?.materials?.length ?? null,
    geometryCount: strategy?.geometries?.length ?? null,
    computeNodeCount: strategy?.computeNodes?.length ?? null,
    geometryModes,
    portableBucketBasePresent: geometry?.getAttribute?.('bucketBase') !== undefined,
    bucketBaseRemovalAbsent: geometry?.getAttribute?.('bucketBase') !== undefined,
    featureLaneConstructed: strategy?.laneState?.lane === 'feature'
      || geometryModes.includes('indirect-first-instance'),
    configuredDrawCommands: strategy?.configuredDrawCommands ?? null,
    configuredComputeDispatches: strategy?.configuredComputeDispatches ?? null,
    configuredComputeSubmissions: strategy?.configuredComputeSubmissions ?? null,
  };
}

function inspectPortableCommands(strategy, scenario) {
  const commands = strategy?.laneState?.indirectAttribute?.array;
  const expectedLength = Math.max(2, scenario.bucketCount) * 5;
  let nonzeroFirstInstanceCount = 0;
  const firstInstanceWords = [];
  if (commands instanceof Uint32Array) {
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const value = commands[bucket * 5 + 4];
      firstInstanceWords.push(value);
      if (value !== 0) nonzeroFirstInstanceCount += 1;
    }
  }
  return {
    pass: commands instanceof Uint32Array
      && commands.length === expectedLength
      && nonzeroFirstInstanceCount === 0,
    commandArrayType: commands?.constructor?.name ?? null,
    commandWordCount: commands?.length ?? null,
    commandRecordCount: commands instanceof Uint32Array ? commands.length / 5 : null,
    drawCommandCount: scenario.bucketCount,
    commandByteOffset: 0,
    firstInstanceWords,
    nonzeroFirstInstanceCount,
    fifthCommandWordsAllZero: nonzeroFirstInstanceCount === 0,
  };
}

/**
 * Runs the forced-off deployment gate on a caller-owned disposable renderer.
 * Address and output callbacks are deliberately injected so the page can use
 * its production oracle/capture path while this helper owns construction,
 * compute/readback validation, and guaranteed cleanup.
 */
export async function runFirstInstanceLiveForcedFeatureOffGate({
  scenario,
  sourceGeometries,
  renderer,
  camera,
  expectedIds = scenario?.expectedVisibleIds,
  addressChallenge = null,
  captureOutput = null,
}) {
  if (!(expectedIds instanceof Uint32Array)) {
    throw new TypeError('Forced feature-off expectedIds must be a Uint32Array.');
  }
  if (addressChallenge !== null && typeof addressChallenge !== 'function') {
    throw new TypeError('addressChallenge must be a function when supplied.');
  }
  if (captureOutput !== null && typeof captureOutput !== 'function') {
    throw new TypeError('captureOutput must be a function when supplied.');
  }

  const actualFeatureAvailable = renderer?.hasFeature?.('indirect-first-instance') === true;
  const selection = selectFixedSliceDeployment({ renderer, featureAvailable: false });
  let strategy = null;
  let scene = null;
  let evidence = null;
  let disposal = {
    pass: false,
    attempted: false,
    rootDetached: false,
    indirectDetached: false,
  };
  try {
    strategy = buildFixedSliceDeploymentStrategy(
      { scenario, sourceGeometries, renderer },
      { featureAvailable: false },
    );
    scene = new Scene();
    scene.name = 'first-instance-live-forced-feature-off-disposable-scene';
    scene.add(strategy.root);
    const construction = inspectPortableConstruction(strategy, scenario);
    const commands = inspectPortableCommands(strategy, scenario);

    strategy.update(camera, renderer);
    strategy.submitCompute(renderer);
    const snapshot = await readFixedSliceLaneSnapshot(
      renderer,
      strategy.sharedResources,
      strategy.laneState,
    );
    const correctness = await validateFixedSliceLaneSnapshot({
      shared: strategy.sharedResources,
      lane: strategy.laneState,
      expectedIds,
      snapshot,
    });
    const callbackContext = {
      strategy,
      scene,
      renderer,
      camera,
      expectedIds,
      correctness,
      snapshot,
      resources: {
        matrixAttribute: strategy.sharedResources.attributes.matrix,
        visibleIdsAttribute: strategy.sharedResources.attributes.visibleIds,
        overflowAttribute: strategy.sharedResources.attributes.overflow,
        indirectAttribute: strategy.laneState.indirectAttribute,
        geometry: strategy.laneState.geometry,
        material: strategy.laneState.material,
        addressMode: strategy.laneState.addressMode,
        commandByteOffset: 0,
        commandOffsets: Array.from(strategy.laneState.commandLayout.offsets),
        firstIndexes: strategy.sharedResources.firstIndexes,
        bucketBases: scenario.bucketBases,
        bucketCounts: scenario.bucketCounts,
      },
    };
    const address = addressChallenge
      ? await addressChallenge(callbackContext)
      : skippedGate('first-instance-live-forced-off-address-gate', 'addressChallenge');
    const output = captureOutput
      ? await captureOutput(callbackContext)
      : skippedGate('first-instance-live-forced-off-output-gate', 'captureOutput');
    evidence = {
      schemaVersion: 1,
      kind: 'first-instance-live-forced-feature-off-deployment-gate',
      passBeforeDisposal: selection.lane === PORTABLE_LANE
        && selection.featureAvailable === false
        && construction.pass
        && construction.featureLaneConstructed === false
        && commands.pass
        && correctness.pass === true
        && address?.pass === true
        && output?.pass === true,
      actualFeatureAvailable,
      forcedFeatureAvailable: false,
      separateDisposableRendererRequired: true,
      timingContaminationBoundary: 'caller-owned-disposable-renderer-device',
      selection,
      construction,
      commands,
      correctness,
      address,
      output,
    };
  } finally {
    if (strategy) {
      disposal.attempted = true;
      disposeStrategyResources(renderer, strategy);
      disposal = {
        pass: strategy.root.parent === null
          && strategy.geometries.every((geometry) => geometry.indirect === null),
        attempted: true,
        rootDetached: strategy.root.parent === null,
        indirectDetached: strategy.geometries.every((geometry) => geometry.indirect === null),
      };
    }
    scene?.clear();
  }
  return {
    ...evidence,
    pass: evidence.passBeforeDisposal && disposal.pass,
    disposal,
  };
}

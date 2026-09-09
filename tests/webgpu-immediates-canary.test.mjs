import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BUNDLE_IMMEDIATE_SEQUENCE,
  BUNDLE_MUTATED_IMMEDIATE_SEQUENCE,
  DIRECT_IMMEDIATE_SEQUENCE,
  IMMEDIATE_CANARY_VALUES,
  INSTANCE_INDEX_SEQUENCE,
  assessImmediateCapabilities,
  assessThreeLaneEligibility,
  buildImmediateCanaryShader,
  isDistinctNonMonotonicSequence,
  validateImmediateCanaryResult,
} from '../src/immediate-canary.js';
import {
  WEBGPU_IMMEDIATES_BROWSER_ARGS,
  WEBGPU_IMMEDIATES_DEFAULT_BROWSER_ARGS,
  WEBGPU_IMMEDIATES_EXPOSURE_MODES,
  assessDefaultExposureBrowserRecord,
  findForbiddenDefaultExposureArguments,
  parseWebGpuImmediatesProbeArguments,
  rawWebGpuImmediatesArtifactModeIsReadOnly,
} from '../scripts/probe-webgpu-immediates.mjs';

function passingCapabilities() {
  return {
    navigatorGpu: true,
    wgslLanguageFeatures: ['immediate_address_space'],
    wgslImmediateAddressSpace: true,
    adapterAvailable: true,
    nonFallbackAdapter: true,
    indirectFirstInstanceAdapter: true,
    indirectFirstInstanceDevice: true,
    adapterMaxImmediateSize: 4,
    deviceMaxImmediateSize: 4,
    renderPassSetImmediates: true,
    renderBundleSetImmediates: true,
  };
}

function expectedValues(addresses) {
  return addresses.map((address) => IMMEDIATE_CANARY_VALUES[address]);
}

function addressesFor(immediates) {
  return immediates.map((base, index) => base + INSTANCE_INDEX_SEQUENCE[index]);
}

function passingNegativeControl(name) {
  return {
    name,
    pass: true,
    rejected: true,
    synchronousException: null,
    scopedErrors: [{
      filter: 'validation',
      name: 'GPUValidationError',
      message: 'expected validation rejection',
    }],
    unexpectedScopedErrors: [],
  };
}

function passingPageResult() {
  const capabilities = passingCapabilities();
  return {
    schemaVersion: 2,
    kind: 'webgpu-immediates-correctness-canary',
    status: 'pass',
    scope: 'raw WebGPU immediate-data capability and correctness only',
    executionMode: 'technical-canary',
    phase: 'phase-0-raw-capability-canary',
    fullPhase0Pass: false,
    integratedCanaryInvoked: false,
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    efficacyEvaluated: false,
    userAgent: 'HeadlessChrome/test',
    secureContext: true,
    crossOriginIsolated: true,
    adapterInfo: { isFallbackAdapter: false },
    adapterFeatures: ['indirect-first-instance'],
    requestedDeviceFeatures: ['indirect-first-instance'],
    deviceFeatures: ['indirect-first-instance'],
    capabilities,
    capabilityAssessment: assessImmediateCapabilities(capabilities),
    threeLaneEligibility: assessThreeLaneEligibility(capabilities),
    shaderCompilationMessages: [],
    pipelineLayoutDescriptor: {
      bindGroupLayoutCount: 1,
      immediateSize: 4,
    },
    canaries: {
      directRenderPass: {
        pass: true,
        renderPassSetImmediates: true,
        addressExpression: 'immediateBase + instanceIndex',
        immediateU32Sequence: [...DIRECT_IMMEDIATE_SEQUENCE],
        instanceIndexSequence: [...INSTANCE_INDEX_SEQUENCE],
        addressSequence: addressesFor(DIRECT_IMMEDIATE_SEQUENCE),
        expectedU32Sequence: expectedValues(addressesFor(DIRECT_IMMEDIATE_SEQUENCE)),
        observedU32Sequence: expectedValues(addressesFor(DIRECT_IMMEDIATE_SEQUENCE)),
      },
      renderBundleSnapshot: {
        pass: true,
        renderBundleSetImmediates: true,
        addressExpression: 'immediateBase + instanceIndex',
        immediateU32SequenceAtEncoding: [...BUNDLE_IMMEDIATE_SEQUENCE],
        sourceU32SequenceAfterFinish: [...BUNDLE_MUTATED_IMMEDIATE_SEQUENCE],
        sourceDetachedAfterMutation: true,
        sourceByteLengthAfterDetach: 0,
        instanceIndexSequence: [...INSTANCE_INDEX_SEQUENCE],
        addressSequenceAtEncoding: addressesFor(BUNDLE_IMMEDIATE_SEQUENCE),
        expectedU32Sequence: expectedValues(addressesFor(BUNDLE_IMMEDIATE_SEQUENCE)),
        mutatedSourceWouldProduceU32Sequence: expectedValues(
          addressesFor(BUNDLE_MUTATED_IMMEDIATE_SEQUENCE),
        ),
        observedU32Sequence: expectedValues(addressesFor(BUNDLE_IMMEDIATE_SEQUENCE)),
        snapshotPreserved: true,
      },
    },
    negativeControls: {
      unsetRequiredSlotAfterBundle: {
        ...passingNegativeControl('unset-required-slot-after-bundle'),
        omittedRange: { byteOffset: 0, byteLength: 4 },
        priorBundleSetSameRange: true,
        directStateRestoredAfterExecuteBundles: {
          pipeline: true,
          bindGroup0: true,
        },
      },
      misalignedRange: {
        ...passingNegativeControl('misaligned-range'),
        requestedRange: { byteOffset: 2, byteLength: 4 },
        deviceMaxImmediateSize: 4,
      },
      rangeBeyondMaxImmediateSize: {
        ...passingNegativeControl('range-beyond-max-immediate-size'),
        requestedRange: { byteOffset: 4, byteLength: 4 },
        deviceMaxImmediateSize: 4,
      },
    },
    deviceErrors: {
      uncapturedErrors: [],
      unexpectedDeviceLosses: [],
      lifecycleScopedErrors: [],
      finalDeviceLoss: { reason: 'destroyed', message: 'destroyed for canary' },
    },
    failure: null,
  };
}

test('WGSL canary requires immediate address space and uses its u32 as an address', () => {
  const shader = buildImmediateCanaryShader();
  assert.match(shader, /^requires immediate_address_space;/u);
  assert.match(shader, /var<immediate> immediateBase\s*:\s*u32;/u);
  assert.match(shader, /values\[immediateBase \+ instanceIndex\]/u);
});

test('raw address schedules contain three distinct non-monotonic values', () => {
  assert.equal(isDistinctNonMonotonicSequence(DIRECT_IMMEDIATE_SEQUENCE), true);
  assert.equal(isDistinctNonMonotonicSequence(BUNDLE_IMMEDIATE_SEQUENCE), true);
  assert.equal(isDistinctNonMonotonicSequence([0, 1, 2]), false);
  assert.equal(isDistinctNonMonotonicSequence([2, 1, 0]), false);
  assert.equal(isDistinctNonMonotonicSequence([2, 0, 2]), false);
});

test('immediate capability is independent from three-lane firstInstance eligibility', () => {
  assert.deepEqual(assessImmediateCapabilities(passingCapabilities()), {
    pass: true,
    requiredImmediateBytes: 4,
    reasons: [],
  });

  for (const mutation of [
    (capabilities) => { capabilities.wgslImmediateAddressSpace = false; },
    (capabilities) => { capabilities.nonFallbackAdapter = false; },
    (capabilities) => { capabilities.adapterMaxImmediateSize = 3; },
    (capabilities) => { capabilities.deviceMaxImmediateSize = 3; },
    (capabilities) => { capabilities.renderPassSetImmediates = false; },
    (capabilities) => { capabilities.renderBundleSetImmediates = false; },
  ]) {
    const capabilities = passingCapabilities();
    mutation(capabilities);
    assert.equal(assessImmediateCapabilities(capabilities).pass, false);
  }

  for (const mutation of [
    (capabilities) => { capabilities.indirectFirstInstanceAdapter = false; },
    (capabilities) => { capabilities.indirectFirstInstanceDevice = false; },
  ]) {
    const capabilities = passingCapabilities();
    mutation(capabilities);
    assert.equal(assessImmediateCapabilities(capabilities).pass, true);
    assert.equal(assessThreeLaneEligibility(capabilities).pass, false);
  }
});

test('page validator accepts exact direct, snapshot, and negative-control semantics', () => {
  assert.deepEqual(validateImmediateCanaryResult(passingPageResult()), {
    pass: true,
    reasons: [],
    capabilities: {
      pass: true,
      requiredImmediateBytes: 4,
      reasons: [],
    },
    threeLaneEligibility: {
      pass: true,
      reasons: [],
      immediateCapabilityPass: true,
      indirectFirstInstanceAdapter: true,
      indirectFirstInstanceDevice: true,
    },
  });
});

test('page validator rejects a bundle that observes post-finish source mutation', () => {
  const result = passingPageResult();
  result.canaries.renderBundleSnapshot.observedU32Sequence =
    expectedValues(addressesFor(BUNDLE_MUTATED_IMMEDIATE_SEQUENCE));
  result.canaries.renderBundleSnapshot.snapshotPreserved = false;
  const validation = validateImmediateCanaryResult(result);
  assert.equal(validation.pass, false);
  assert.ok(validation.reasons.includes('render-bundle non-monotonic snapshot canary failed'));
});

test('page validator rejects a silently accepted negative control', () => {
  const result = passingPageResult();
  result.negativeControls.misalignedRange.pass = false;
  result.negativeControls.misalignedRange.rejected = false;
  result.negativeControls.misalignedRange.scopedErrors = [];
  const validation = validateImmediateCanaryResult(result);
  assert.equal(validation.pass, false);
  assert.ok(validation.reasons.includes(
    'misalignedRange negative control was not cleanly rejected',
  ));
});

test('page validator rejects synchronous or wrongly typed negative-control errors', () => {
  const synchronous = passingPageResult();
  synchronous.negativeControls.misalignedRange.synchronousException = {
    name: 'TypeError',
    message: 'unrelated JavaScript failure',
  };
  synchronous.negativeControls.misalignedRange.scopedErrors = [];
  assert.equal(validateImmediateCanaryResult(synchronous).pass, false);

  const wrongType = passingPageResult();
  wrongType.negativeControls.misalignedRange.scopedErrors[0].name = 'GPUInternalError';
  assert.equal(validateImmediateCanaryResult(wrongType).pass, false);

  const duplicate = passingPageResult();
  duplicate.negativeControls.misalignedRange.scopedErrors.push({
    filter: 'validation',
    name: 'GPUValidationError',
    message: 'second error',
  });
  assert.equal(validateImmediateCanaryResult(duplicate).pass, false);
});

test('page validator rejects mismatched control metadata and derived capability fields', () => {
  const wrongOffset = passingPageResult();
  wrongOffset.negativeControls.rangeBeyondMaxImmediateSize.requestedRange.byteOffset = 8;
  assert.equal(validateImmediateCanaryResult(wrongOffset).pass, false);

  const wrongFeatureInventory = passingPageResult();
  wrongFeatureInventory.capabilities.wgslLanguageFeatures = [];
  assert.equal(validateImmediateCanaryResult(wrongFeatureInventory).pass, false);

  const retainedFailure = passingPageResult();
  retainedFailure.failure = { name: 'UnexpectedError' };
  assert.equal(validateImmediateCanaryResult(retainedFailure).pass, false);
});

test('page validator requires lifecycle scope and final destroyed-device completion', () => {
  const scopedError = passingPageResult();
  scopedError.deviceErrors.lifecycleScopedErrors.push({
    filter: 'validation',
    name: 'GPUValidationError',
    message: 'late error',
  });
  assert.equal(validateImmediateCanaryResult(scopedError).pass, false);

  const unexpectedLoss = passingPageResult();
  unexpectedLoss.deviceErrors.finalDeviceLoss.reason = 'unknown';
  assert.equal(validateImmediateCanaryResult(unexpectedLoss).pass, false);
});

test('page validator rejects performance or efficacy evidence', () => {
  const result = passingPageResult();
  result.timingCaptured = true;
  assert.equal(validateImmediateCanaryResult(result).pass, false);
});

test('developer-enabled probe preserves its established command and launch arguments', () => {
  assert.deepEqual(WEBGPU_IMMEDIATES_BROWSER_ARGS, [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
  ]);
  assert.equal(Object.isFrozen(WEBGPU_IMMEDIATES_BROWSER_ARGS), true);
  assert.deepEqual(parseWebGpuImmediatesProbeArguments([]), {
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED,
  });
});

test('default-exposure probe uses no custom launch flags and has no developer fallback', () => {
  assert.deepEqual(WEBGPU_IMMEDIATES_DEFAULT_BROWSER_ARGS, []);
  assert.equal(Object.isFrozen(WEBGPU_IMMEDIATES_DEFAULT_BROWSER_ARGS), true);
  assert.deepEqual(parseWebGpuImmediatesProbeArguments(['--default-exposure']), {
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
  });
  assert.throws(
    () => parseWebGpuImmediatesProbeArguments(['--default-exposure', '--retry-with-flags']),
    /accepts only --default-exposure/u,
  );
});

test('default-exposure effective-argument validator rejects capability-expanding flags', () => {
  const cleanRecord = {
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
    launchArguments: [],
    effectiveArgumentsCapture: {
      api: 'Browser.getBrowserCommandLine',
      available: true,
      arguments: ['chrome.exe', '--headless', '--enable-automation'],
      forbiddenArguments: [],
    },
  };
  assert.deepEqual(assessDefaultExposureBrowserRecord(cleanRecord), {
    pass: true,
    reasons: [],
  });

  for (const forbiddenArgument of [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--enable-dawn-features=allow_unsafe_apis',
    '--disable-dawn-features=some_toggle',
    '--use-webgpu-adapter=swiftshader',
    '--force-webgpu-compat',
    '--ignore-gpu-blocklist',
    '--force-high-performance-gpu',
  ]) {
    const record = structuredClone(cleanRecord);
    record.effectiveArgumentsCapture.arguments.push(forbiddenArgument);
    record.effectiveArgumentsCapture.forbiddenArguments = [forbiddenArgument];
    assert.equal(assessDefaultExposureBrowserRecord(record).pass, false);
    assert.deepEqual(
      findForbiddenDefaultExposureArguments(record.effectiveArgumentsCapture.arguments),
      [forbiddenArgument],
    );
  }
});

test('raw canary artifact mode must be read-only', () => {
  assert.equal(rawWebGpuImmediatesArtifactModeIsReadOnly({ mode: 0o100444 }), true);
  assert.equal(rawWebGpuImmediatesArtifactModeIsReadOnly({ mode: 0o100644 }), false);
  assert.equal(rawWebGpuImmediatesArtifactModeIsReadOnly({ mode: 0o100666 }), false);
  assert.equal(rawWebGpuImmediatesArtifactModeIsReadOnly(null), false);
});

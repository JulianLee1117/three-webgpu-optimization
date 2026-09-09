import { createHash } from 'node:crypto';
import { execFile as execFileCallback } from 'node:child_process';
import {
  chmod,
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  realpath,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import {
  DEFAULT_THREE_DRAW_INDEX_OUTPUT_ROOT,
  EXPECTED_THREE_DRAW_INDEX_COMMIT,
  EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES,
  EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256,
  EXPECTED_THREE_DRAW_INDEX_REVISION,
  EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT,
  EXPECTED_THREE_DRAW_INDEX_SRC_TREE,
  EXPECTED_THREE_DRAW_INDEX_TREE,
  THREE_DRAW_INDEX_TARGET_PATHS,
  buildThreeDrawIndexOverlay,
} from './build-three-draw-index-overlay.mjs';

export const THREE_DRAW_INDEX_RUNNER_KIND =
  'three-webgpu-draw-index-default-exposure-runner';
export const THREE_DRAW_INDEX_REPORT_KIND =
  'three-webgpu-draw-index-default-exposure-report';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runnerPath = fileURLToPath(import.meta.url);
const protocolPath = path.join(repoRoot, 'protocols', 'three-draw-index-canary-v1.json');
const overlayManifestFilename = 'THREE_DRAW_INDEX_OVERLAY.json';
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const EXPECTED_PAGE_KIND = 'three-webgpu-draw-index-correctness-canary';
const EXPECTED_EXECUTION_MODE = 'timing-free-one-shot-correctness';
const EXPECTED_PROTOCOL_BYTES = 3_772;
const EXPECTED_PROTOCOL_SHA256 =
  '760b9a8e62bdeb454f208e8fa30fe63ffac16d2be832b818bda51d0e9879933e';
const EXPECTED_PROTOCOL_COMMAND =
  'npm.cmd run probe:three-draw-index -- --browser '
  + '"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"';
const EXPECTED_CLAIM_BOUNDARY =
  'Correctness and integration on one standards-capable implementation; '
  + 'no performance or cross-device universality claim.';
const EXPECTED_REPORT_PATH =
  'results/development/three-draw-index-default-exposure-canary/'
  + 'canary-draw-index-default-exposure-994260a-v1/report.json';
const EXPECTED_SIDECAR_PATH = `${EXPECTED_REPORT_PATH}.sha256`;
const EXPECTED_OVERLAY_TOTAL_BYTES = 4_648_406;
const EXPECTED_OVERLAY_INVENTORY_SHA256 =
  '9359deb388c8351001f54058335233d0ddea9d3cd950adaa3ff7c4799e0968ae';
const EXPECTED_FORBIDDEN_ARGUMENT_FRAGMENTS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--enable-dawn-features',
  '--disable-dawn-features',
  '--use-angle',
  '--use-vulkan',
  '--use-webgpu-adapter',
  '--force-webgpu-compat',
  '--ignore-gpu-blocklist',
  '--force-high-performance-gpu',
]);
const EXPECTED_CAPABILITY_ORDER = Object.freeze([
  'secure context and navigator.gpu',
  'WGSL immediate_address_space language feature',
  'navigator.gpu.requestAdapter() with no descriptor',
  'adapter.requestDevice() with no descriptor',
  'actual integer device.limits.maxImmediateSize >= 4',
  'render-pass and render-bundle setImmediates surfaces',
  'renderer construction and shader generation',
]);
const EXPECTED_ASSERTIONS = Object.freeze([
  'direct fragment draw produces drawIndex zero through a flat u32 varying',
  'nonmonotonic indexed-indirect order 2,0,3,1 receives compact ordinals 0,1,2,3',
  'cached render bundle is reused without re-recording',
  'explicit invalidation produces a distinct correctly recorded bundle',
  'BatchedMesh order 2,0,3,1 distinguishes compact drawIndex from original batchIndirectIndex',
  'default BatchedMesh remains immediate-free with layout size zero',
  'both ArrayCamera layers independently reset drawIndex to zero',
  'immediate shaders use one requirement and declaration; fragment code has no direct immediate access',
  'every requiring physical draw has an immediately adjacent exact four-byte setter',
  'all Three.js errors, browser errors, GPU errors, scope results, and unexpected losses are empty',
  'source, served runtime, browser executable, overlay, and artifact identities are exact',
]);
const EXECUTION_FAILURE_STAGES = Object.freeze([
  'server-create',
  'server-listen',
  'server-address',
  'browser-launch',
  'browser-version',
  'browser-command-line',
  'browser-policy',
  'context-create',
  'route-install',
  'page-create',
  'page-observers',
  'init-script',
  'navigation',
  'page-ready',
  'page-result',
  'page-diagnostics',
  'page-resources',
  'page-validation',
  'served-source-validation',
  'browser-observations',
]);
const DRAW_COUNT = 4;
const INDIRECT_WORD_COUNT = 5;
const PHYSICAL_DRAW_METHODS = Object.freeze([
  'draw',
  'drawIndexed',
  'drawIndirect',
  'drawIndexedIndirect',
]);
const BATCH_ORDER = Object.freeze([2, 0, 3, 1]);
const INDIRECT_OFFSETS = Object.freeze([40, 0, 60, 20]);
const RERECORD_OFFSETS = Object.freeze([20, 60, 0, 40]);
const BOUND_SOURCE_PATHS = Object.freeze([
  'scripts/probe-three-draw-index.mjs',
  'scripts/build-three-draw-index-overlay.mjs',
  'src/three-draw-index-canary.js',
  'src/three-draw-index-canary-contract.js',
  'three-draw-index-canary.html',
  'protocols/three-draw-index-canary-v1.json',
  'docs/WEBGPU_DRAW_INDEX_IMMEDIATES_PROTOCOL.md',
  'patches/three-webgpu-draw-index-immediates.patch',
  'patches/three-webgpu-draw-index-immediates.json',
  'package.json',
  'package-lock.json',
]);
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonicalJsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 8_192),
  };
}

function exactArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function addReasonUnless(condition, reasons, message) {
  if (!condition) reasons.push(message);
}

function isStrictJsonData(value, seen = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || seen.has(value)) return false;
  seen.add(value);
  const valid = Array.isArray(value)
    ? value.every((entry) => isStrictJsonData(entry, seen))
    : Object.getPrototypeOf(value) === Object.prototype
      && Object.entries(value).every(([key, entry]) => (
        typeof key === 'string' && isStrictJsonData(entry, seen)
      ));
  seen.delete(value);
  return valid;
}

function countToken(source, token) {
  return typeof source === 'string' ? source.split(token).length - 1 : 0;
}

function rgbaPixels(redValues, greenValues = redValues.map(() => 0)) {
  return redValues.flatMap((red, index) => [red, greenValues[index], 0, 255]);
}

function expectedIndirectOutput(offsets = INDIRECT_OFFSETS) {
  const drawOrdinalByGeometry = Array(DRAW_COUNT).fill(-1);
  offsets.forEach((offset, drawOrdinal) => {
    drawOrdinalByGeometry[offset / 20] = drawOrdinal;
  });
  return rgbaPixels(drawOrdinalByGeometry.map((drawOrdinal) => drawOrdinal + 1));
}

function expectedBatchedOutput() {
  const drawOrdinalByInstance = Array(DRAW_COUNT).fill(-1);
  BATCH_ORDER.forEach((instanceId, drawOrdinal) => {
    drawOrdinalByInstance[instanceId] = drawOrdinal;
  });
  return rgbaPixels(
    drawOrdinalByInstance.map((drawOrdinal) => drawOrdinal + 1),
    [1, 2, 3, 4],
  );
}

function expectedCommands() {
  const words = [];
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    words.push(3, 1, draw * 3, 0, 0);
  }
  return words;
}

function validateOutput(record, expected, reasons, label) {
  const expectedHash = sha256(Buffer.from(expected));
  const observed = record?.observed;
  const observedHash = Array.isArray(observed)
    ? sha256(Buffer.from(observed))
    : null;
  const mismatchCount = Array.isArray(observed) && observed.length === expected.length
    ? observed.reduce(
      (count, value, index) => count + Number(value !== expected[index]),
      0,
    )
    : null;
  addReasonUnless(
    record?.exact === true
      && exactArray(record?.expected, expected)
      && exactArray(observed, expected)
      && record?.expectedSha256 === expectedHash
      && record?.observedSha256 === observedHash
      && observedHash === expectedHash
      && record?.mismatchCount === 0
      && mismatchCount === 0,
    reasons,
    `${label} output bytes or hashes are invalid`,
  );
}

function phaseTraces(result, collectionName, phase) {
  const traces = result?.instrumentation?.[collectionName];
  if (!Array.isArray(traces)) return [];
  return traces.filter((trace) => trace?.creationPhase === phase
    || trace?.events?.some((event) => event?.phase === phase));
}

function traceDrawEvents(trace, method) {
  return trace?.events?.filter((event) => event?.method === method) ?? [];
}

function validateImmediateTrace(
  trace,
  drawMethod,
  expectedOrdinals,
  reasons,
  label,
  expectedOffsets = null,
) {
  const events = trace?.events;
  if (!Array.isArray(events)) {
    reasons.push(`${label} trace is absent`);
    return;
  }
  const draws = [];
  const setters = events.filter((event) => event?.method === 'setImmediates');
  const physicalDraws = events.filter((event) => PHYSICAL_DRAW_METHODS.includes(event?.method));
  for (let index = 0; index < events.length; index += 1) {
    if (events[index]?.method === drawMethod) draws.push({ event: events[index], index });
  }
  addReasonUnless(
    trace?.callableSetImmediates === true
      && draws.length === expectedOrdinals.length
      && physicalDraws.length === expectedOrdinals.length
      && physicalDraws.every((event) => event.method === drawMethod)
      && setters.length === expectedOrdinals.length,
    reasons,
    `${label} setter/draw counts are invalid`,
  );
  for (let draw = 0; draw < expectedOrdinals.length; draw += 1) {
    const current = draws[draw];
    const setter = current === undefined ? null : events[current.index - 1];
    addReasonUnless(
      setter?.method === 'setImmediates'
        && setter?.rangeOffset === 0
        && setter?.sourceType === 'Uint32Array'
        && setter?.sourceElementCount === 1
        && setter?.dataOffset === 0
        && setter?.dataSize === 1
        && exactArray(setter?.selectedValues, [expectedOrdinals[draw]]),
      reasons,
      `${label} draw ${draw} lacks its exact immediately adjacent setter`,
    );
    if (expectedOffsets !== null) {
      addReasonUnless(
        current?.event?.indirectOffset === expectedOffsets[draw],
        reasons,
        `${label} draw ${draw} indirect offset is invalid`,
      );
    }
  }
}

function validateControlTrace(trace, reasons, label) {
  const setters = trace?.events?.filter((event) => event?.method === 'setImmediates') ?? [];
  const draws = traceDrawEvents(trace, 'drawIndexed');
  const physicalDraws = trace?.events?.filter(
    (event) => PHYSICAL_DRAW_METHODS.includes(event?.method),
  ) ?? [];
  addReasonUnless(
    setters.length === 0
      && draws.length === DRAW_COUNT
      && physicalDraws.length === DRAW_COUNT
      && physicalDraws.every((event) => event.method === 'drawIndexed')
      && draws.every((event, index) => event?.arguments?.[4] === index),
    reasons,
    `${label} must contain four compact batched draws and no immediate setter`,
  );
}

function pipelineEvidenceForTrace(result, trace) {
  const pipelineEvents = trace?.events?.filter((event) => event?.method === 'setPipeline') ?? [];
  const pipelineEvent = pipelineEvents[0];
  const pipeline = result?.instrumentation?.renderPipelines?.find(
    (record) => record?.pipelineId === pipelineEvent?.pipelineId,
  );
  const layout = result?.instrumentation?.pipelineLayouts?.find(
    (record) => record?.layoutId === pipeline?.layoutId,
  );
  const vertex = result?.instrumentation?.shaderModules?.find(
    (record) => record?.moduleId === pipeline?.vertexModuleId,
  );
  const fragment = result?.instrumentation?.shaderModules?.find(
    (record) => record?.moduleId === pipeline?.fragmentModuleId,
  );
  return { pipelineEvents, pipelineEvent, pipeline, layout, vertex, fragment };
}

function validatePipelineForTrace(result, trace, immediateSize, reasons, label) {
  const evidence = pipelineEvidenceForTrace(result, trace);
  addReasonUnless(
    evidence.pipelineEvents.length === 1
      && typeof evidence.pipelineEvent?.pipelineId === 'string'
      && evidence.pipeline?.pipelineId === evidence.pipelineEvent.pipelineId
      && evidence.layout?.hasOwnImmediateSize === true
      && evidence.layout?.immediateSize === immediateSize,
    reasons,
    `${label} pipeline/layout identity or immediate size is invalid`,
  );
  const vertex = evidence.vertex?.code;
  const fragment = evidence.fragment?.code;
  if (immediateSize === 4) {
    addReasonUnless(
      typeof vertex === 'string'
        && typeof fragment === 'string'
        && countToken(vertex, 'requires immediate_address_space;') === 1
        && countToken(vertex, 'var<immediate> nodeDrawIndex : u32;') === 1
        && /varyings\.nodeVarying\d+\s*=\s*nodeDrawIndex\s*;/u.test(vertex)
        && /@interpolate\(flat, either\)\s+nodeVarying\d+\s*:\s*u32/u.test(fragment)
        && /f32\(\s*nodeVarying\d+\s*\)/u.test(fragment)
        && !fragment?.includes('nodeDrawIndex'),
      reasons,
      `${label} WGSL requirement/declaration/flat-varying evidence is invalid`,
    );
  } else {
    addReasonUnless(
      typeof vertex === 'string'
        && typeof fragment === 'string'
        && !vertex.includes('immediate_address_space')
        && !vertex?.includes('nodeDrawIndex')
        && !fragment?.includes('nodeDrawIndex'),
      reasons,
      `${label} immediate-free WGSL contains an immediate token`,
    );
  }
}

function bundleExecutionsForPhase(result, phase) {
  return phaseTraces(result, 'renderPasses', phase).flatMap((trace) => (
    trace.events
      ?.filter((event) => event?.method === 'executeBundles')
      .flatMap((event) => event.bundleIds ?? [])
      ?? []
  ));
}

function tracesWithDrawMethod(traces, method) {
  return traces.filter((trace) => traceDrawEvents(trace, method).length > 0);
}

function validateGlobalDrawLedger(result, reasons) {
  const instrumentation = result?.instrumentation;
  const passes = instrumentation?.renderPasses;
  const bundles = instrumentation?.renderBundles;
  const pipelines = instrumentation?.renderPipelines;
  const layouts = instrumentation?.pipelineLayouts;
  const modules = instrumentation?.shaderModules;
  if (![passes, bundles, pipelines, layouts, modules].every(Array.isArray)) {
    reasons.push('global draw ledger collections are absent');
    return;
  }
  const uniqueRecords = (records, key) => records.every((record) => (
    typeof record?.[key] === 'string'
  )) && new Set(records.map((record) => record[key])).size === records.length;
  let valid = uniqueRecords(pipelines, 'pipelineId')
    && uniqueRecords(layouts, 'layoutId')
    && uniqueRecords(modules, 'moduleId');
  const pipelineById = new Map(pipelines.map((record) => [record.pipelineId, record]));
  const layoutById = new Map(layouts.map((record) => [record.layoutId, record]));
  const moduleById = new Map(modules.map((record) => [record.moduleId, record]));
  valid &&= pipelines.every((pipeline) => (
    layoutById.has(pipeline.layoutId)
      && moduleById.has(pipeline.vertexModuleId)
      && moduleById.has(pipeline.fragmentModuleId)
  ));

  const expectedCounts = new Map([
    ['pass\0direct-fragment\0drawIndexed', 1],
    ['bundle\0indirect-bundle-first\0drawIndexedIndirect', 4],
    ['bundle\0indirect-bundle-rerecord\0drawIndexedIndirect', 4],
    ['pass\0batched-explicit\0drawIndexed', 4],
    ['pass\0batched-control\0drawIndexed', 4],
    ['bundle\0array-camera\0drawIndexedIndirect', 8],
  ]);
  const observedCounts = new Map();
  let immediateSetterCount = 0;
  for (const trace of [...passes, ...bundles]) {
    let activePipelineId = null;
    const events = trace?.events;
    if (!Array.isArray(events)) {
      valid = false;
      continue;
    }
    for (let index = 0; index < events.length; index += 1) {
      const event = events[index];
      if (event?.method === 'setPipeline') {
        activePipelineId = event.pipelineId;
        valid &&= pipelineById.has(activePipelineId);
        continue;
      }
      const pipeline = pipelineById.get(activePipelineId);
      const layout = layoutById.get(pipeline?.layoutId);
      if (event?.method === 'setImmediates') {
        immediateSetterCount += 1;
        valid &&= layout?.immediateSize === 4
          && PHYSICAL_DRAW_METHODS.includes(events[index + 1]?.method);
      }
      if (PHYSICAL_DRAW_METHODS.includes(event?.method)) {
        const key = `${trace.type}\0${event.phase}\0${event.method}`;
        observedCounts.set(key, (observedCounts.get(key) ?? 0) + 1);
        valid &&= event.phase === trace.creationPhase
          && (layout?.immediateSize === 0 || layout?.immediateSize === 4)
          && (layout?.immediateSize === 4
            ? events[index - 1]?.method === 'setImmediates'
            : events[index - 1]?.method !== 'setImmediates');
      }
    }
  }
  valid &&= immediateSetterCount === 21
    && observedCounts.size === expectedCounts.size
    && [...expectedCounts].every(([key, count]) => observedCounts.get(key) === count);
  addReasonUnless(valid, reasons,
    'global draw/pipeline ledger contains an unclassified or inconsistent event');
}

export function validateThreeDrawIndexPageResult(result) {
  const reasons = [];
  addReasonUnless(
    result?.schemaVersion === 1
      && result?.kind === EXPECTED_PAGE_KIND
      && result?.executionMode === EXPECTED_EXECUTION_MODE
      && result?.timingCaptured === false
      && result?.benchmarkClaim === false
      && result?.claimBoundary === EXPECTED_CLAIM_BOUNDARY,
    reasons,
    'page result header or timing boundary is invalid',
  );

  if (result?.status === 'unsupported') {
    const preflight = result?.preflight;
    const stage = preflight?.unsupportedStage;
    const beforeAdapter = ['secure-context', 'navigator-gpu', 'wgsl-language-feature']
      .includes(stage);
    const adapterUnavailable = stage === 'adapter-request';
    const afterDevice = ['device-limit', 'encoder-surfaces'].includes(stage);
    const adapterWasRequested = preflight?.adapterRequestAttempts === 1
      && preflight?.requestAdapterArgumentsCount === 0
      && preflight?.requestAdapterDescriptorSupplied === false;
    const adapterWasNotRequested = preflight?.adapterRequestAttempts === 0
      && preflight?.adapterAcquisitions === 0
      && preflight?.requestAdapterArgumentsCount === null
      && preflight?.requestAdapterDescriptorSupplied === null;
    const deviceWasRequested = preflight?.deviceRequestAttempts === 1
      && preflight?.deviceAcquisitions === 1
      && preflight?.requestDeviceArgumentsCount === 0
      && preflight?.requestDeviceDescriptorSupplied === false;
    const deviceWasNotRequested = preflight?.deviceRequestAttempts === 0
      && preflight?.deviceAcquisitions === 0
      && preflight?.requestDeviceArgumentsCount === null
      && preflight?.requestDeviceDescriptorSupplied === null;
    const stageEvidenceIsExact = (beforeAdapter
      && adapterWasNotRequested
      && deviceWasNotRequested)
      || (adapterUnavailable
        && adapterWasRequested
        && preflight?.adapterAcquisitions === 0
        && deviceWasNotRequested)
      || (afterDevice
        && adapterWasRequested
        && preflight?.adapterAcquisitions === 1
        && deviceWasRequested);
    const stageCapabilityIsExact = (stage === 'secure-context'
      && preflight?.secureContext === false)
      || (stage === 'navigator-gpu'
        && preflight?.secureContext === true
        && result?.capabilities?.wgslLanguageFeatures?.length === 0)
      || (stage === 'wgsl-language-feature'
        && preflight?.secureContext === true
        && result?.capabilities?.immediateAddressSpace === false)
      || (stage === 'adapter-request'
        && result?.capabilities?.immediateAddressSpace === true
        && result?.adapterInfo === null)
      || (stage === 'device-limit'
        && (!Number.isInteger(result?.capabilities?.deviceMaxImmediateSize)
          || result.capabilities.deviceMaxImmediateSize < 4))
      || (stage === 'encoder-surfaces'
        && Number.isInteger(result?.capabilities?.deviceMaxImmediateSize)
        && result.capabilities.deviceMaxImmediateSize >= 4
        && (result?.capabilities?.renderPassSetImmediates !== true
          || result?.capabilities?.renderBundleSetImmediates !== true));
    const instrumentationIsEmpty = result?.instrumentation !== null
      && result?.instrumentation?.shaderModules?.length === 0
      && result?.instrumentation?.pipelineLayouts?.length === 0
      && result?.instrumentation?.renderPipelines?.length === 0
      && result?.instrumentation?.renderPasses?.length === 0
      && result?.instrumentation?.renderBundles?.length === 0
      && result?.instrumentation?.errorScopes?.length === 0
      && Array.isArray(result?.instrumentation?.patchedDeviceMethods)
      && result.instrumentation.patchedDeviceMethods.length > 0;
    const instrumentationStageIsExact = afterDevice
      ? (preflight?.instrumentationInstalledBeforeRenderer === true
        && instrumentationIsEmpty)
      : (preflight?.instrumentationInstalledBeforeRenderer === false
        && result?.instrumentation === null);
    addReasonUnless(
      result?.failure === null
        && preflight?.capabilityDecision === 'unsupported'
        && preflight?.currentStage === stage
        && typeof preflight?.unsupportedReason === 'string'
        && preflight.unsupportedReason.length > 0
        && stageEvidenceIsExact
        && stageCapabilityIsExact
        && instrumentationStageIsExact
        && preflight?.shaderModulesAtCapabilityDecision === 0
        && preflight?.rendererConstructed === false
        && result?.runtimeSignature === null
        && result?.scenarios === null
        && Array.isArray(result?.shaderCompilation)
        && result.shaderCompilation.length === 0
        && result?.threeErrors?.length === 0
        && result?.threeConsoleErrors?.length === 0
        && result?.threeDeviceLosses?.length === 0
        && result?.gpuErrors?.uncaptured?.length === 0
        && result?.gpuErrors?.unexpectedDeviceLosses?.length === 0
        && result?.cleanupFailures?.length === 0
        && (afterDevice
          ? result?.errorObservation?.intentionalDeviceLoss?.reason === 'destroyed'
          : result?.errorObservation?.intentionalDeviceLoss === null),
      reasons,
      'unsupported result crossed the capability boundary or contains errors',
    );
    return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
  }

  addReasonUnless(result?.status === 'passed' && result?.failure === null,
    reasons, 'page did not report a clean pass');
  addReasonUnless(
    result?.preflight?.secureContext === true
      && result?.preflight?.capabilityDecision === 'supported'
      && result?.preflight?.currentStage === 'renderer-and-shaders'
      && result?.preflight?.unsupportedStage === null
      && result?.preflight?.adapterRequestAttempts === 1
      && result?.preflight?.adapterAcquisitions === 1
      && result?.preflight?.requestAdapterArgumentsCount === 0
      && result?.preflight?.requestAdapterDescriptorSupplied === false
      && result?.preflight?.deviceRequestAttempts === 1
      && result?.preflight?.deviceAcquisitions === 1
      && result?.preflight?.requestDeviceArgumentsCount === 0
      && result?.preflight?.requestDeviceDescriptorSupplied === false
      && result?.preflight?.shaderModulesAtCapabilityDecision === 0
      && result?.preflight?.instrumentationInstalledBeforeRenderer === true
      && result?.preflight?.rendererConstructed === true,
    reasons,
    'capability-before-shader or default-device preflight evidence is invalid',
  );
  addReasonUnless(
    result?.capabilities?.immediateAddressSpace === true
      && result?.capabilities?.wgslLanguageFeatures?.includes('immediate_address_space')
      && Number.isInteger(result?.capabilities?.deviceMaxImmediateSize)
      && result.capabilities.deviceMaxImmediateSize >= 4
      && result?.capabilities?.renderPassSetImmediates === true
      && result?.capabilities?.renderBundleSetImmediates === true,
    reasons,
    'required WebGPU capability evidence is invalid',
  );
  addReasonUnless(
    result?.runtimeSignature?.revision === EXPECTED_THREE_DRAW_INDEX_REVISION
      && result?.runtimeSignature?.drawIndexNode === true
      && result?.runtimeSignature?.backendHasDrawIndexSetter === true
      && result?.runtimeSignature?.backendDrawUsesPipelineMetadata === true
      && result?.runtimeSignature?.builderReturnsImmediateSymbol === true
      && result?.runtimeSignature?.builderChecksDrawIndexAvailability === true,
    reasons,
    'runtime overlay signature is invalid',
  );
  validateGlobalDrawLedger(result, reasons);

  validateOutput(
    result?.scenarios?.directFragment?.output,
    [1, 0, 0, 255],
    reasons,
    'direct fragment',
  );
  validateOutput(
    result?.scenarios?.indirectBundle?.first?.output,
    expectedIndirectOutput(),
    reasons,
    'first indirect bundle',
  );
  validateOutput(
    result?.scenarios?.indirectBundle?.cached?.output,
    expectedIndirectOutput(),
    reasons,
    'cached indirect bundle',
  );
  validateOutput(
    result?.scenarios?.indirectBundle?.rerecorded?.output,
    expectedIndirectOutput(RERECORD_OFFSETS),
    reasons,
    're-recorded indirect bundle',
  );
  validateOutput(
    result?.scenarios?.batched?.explicit?.output,
    expectedBatchedOutput(),
    reasons,
    'explicit BatchedMesh',
  );
  validateOutput(
    result?.scenarios?.batched?.control?.output,
    rgbaPixels([0, 0, 0, 0], [1, 2, 3, 4]),
    reasons,
    'control BatchedMesh',
  );
  for (let layer = 0; layer < 2; layer += 1) {
    validateOutput(
      result?.scenarios?.arrayCamera?.layers?.[layer],
      expectedIndirectOutput(),
      reasons,
      `ArrayCamera layer ${layer}`,
    );
  }
  addReasonUnless(
    exactArray(result?.scenarios?.indirectBundle?.commands, expectedCommands())
      && exactArray(result?.scenarios?.indirectBundle?.initialOffsets, INDIRECT_OFFSETS)
      && exactArray(result?.scenarios?.indirectBundle?.rerecordOffsets, RERECORD_OFFSETS)
      && result?.scenarios?.indirectBundle?.rootVersion === 1,
    reasons,
    'indirect commands, offsets, or bundle invalidation version are invalid',
  );
  addReasonUnless(
    exactArray(result?.scenarios?.batched?.requestedOrder, BATCH_ORDER)
      && exactArray(result?.scenarios?.batched?.observedIndirectTextureOrder, BATCH_ORDER),
    reasons,
    'BatchedMesh did not realize the forced nonidentity order',
  );
  addReasonUnless(
    result?.scenarios?.arrayCamera?.cameraCount === 2
      && result?.scenarios?.arrayCamera?.targetDepth === 2
      && result?.scenarios?.arrayCamera?.layers?.length === 2,
    reasons,
    'ArrayCamera topology is invalid',
  );

  const ordinals = [0, 1, 2, 3];
  const directTraces = tracesWithDrawMethod(
    phaseTraces(result, 'renderPasses', 'direct-fragment'),
    'drawIndexed',
  );
  addReasonUnless(directTraces.length === 1, reasons,
    'direct phase must contain exactly one indexed draw trace');
  if (directTraces.length === 1) {
    validateImmediateTrace(directTraces[0], 'drawIndexed', [0], reasons, 'direct');
    validatePipelineForTrace(result, directTraces[0], 4, reasons, 'direct');
  }

  const firstBundles = tracesWithDrawMethod(
    phaseTraces(result, 'renderBundles', 'indirect-bundle-first'),
    'drawIndexedIndirect',
  );
  const cachedBundles = tracesWithDrawMethod(
    phaseTraces(result, 'renderBundles', 'indirect-bundle-cached'),
    'drawIndexedIndirect',
  );
  const rerecordedBundles = tracesWithDrawMethod(
    phaseTraces(result, 'renderBundles', 'indirect-bundle-rerecord'),
    'drawIndexedIndirect',
  );
  addReasonUnless(firstBundles.length === 1 && cachedBundles.length === 0
    && rerecordedBundles.length === 1, reasons,
  'bundle record/cache/re-record encoder counts are invalid');
  if (firstBundles.length === 1) {
    validateImmediateTrace(
      firstBundles[0],
      'drawIndexedIndirect',
      ordinals,
      reasons,
      'first indirect bundle',
      INDIRECT_OFFSETS,
    );
    validatePipelineForTrace(result, firstBundles[0], 4, reasons, 'first indirect bundle');
  }
  if (rerecordedBundles.length === 1) {
    validateImmediateTrace(
      rerecordedBundles[0],
      'drawIndexedIndirect',
      ordinals,
      reasons,
      're-recorded indirect bundle',
      RERECORD_OFFSETS,
    );
    validatePipelineForTrace(
      result,
      rerecordedBundles[0],
      4,
      reasons,
      're-recorded indirect bundle',
    );
  }
  const firstExecutionIds = bundleExecutionsForPhase(result, 'indirect-bundle-first');
  const cachedExecutionIds = bundleExecutionsForPhase(result, 'indirect-bundle-cached');
  const rerecordedExecutionIds = bundleExecutionsForPhase(result, 'indirect-bundle-rerecord');
  addReasonUnless(
    firstExecutionIds.length === 1
      && cachedExecutionIds.length === 1
      && rerecordedExecutionIds.length === 1
      && firstExecutionIds[0] === cachedExecutionIds[0]
      && firstExecutionIds[0] !== rerecordedExecutionIds[0]
      && exactArray(
        result?.scenarios?.indirectBundle?.first?.executedBundleIds,
        firstExecutionIds,
      )
      && exactArray(
        result?.scenarios?.indirectBundle?.cached?.executedBundleIds,
        cachedExecutionIds,
      )
      && exactArray(
        result?.scenarios?.indirectBundle?.rerecorded?.executedBundleIds,
        rerecordedExecutionIds,
      ),
    reasons,
    'executed bundle identities do not prove reuse followed by re-recording',
  );

  const explicitBatchTraces = tracesWithDrawMethod(
    phaseTraces(result, 'renderPasses', 'batched-explicit'),
    'drawIndexed',
  );
  const controlBatchTraces = tracesWithDrawMethod(
    phaseTraces(result, 'renderPasses', 'batched-control'),
    'drawIndexed',
  );
  addReasonUnless(explicitBatchTraces.length === 1 && controlBatchTraces.length === 1,
    reasons, 'BatchedMesh phases must each contain one pass trace');
  if (explicitBatchTraces.length === 1) {
    validateImmediateTrace(
      explicitBatchTraces[0],
      'drawIndexed',
      ordinals,
      reasons,
      'explicit BatchedMesh',
    );
    addReasonUnless(
      traceDrawEvents(explicitBatchTraces[0], 'drawIndexed')
        .every((event, index) => event?.arguments?.[4] === index),
      reasons,
      'explicit BatchedMesh firstInstance values are not compact',
    );
    validatePipelineForTrace(result, explicitBatchTraces[0], 4, reasons,
      'explicit BatchedMesh');
  }
  if (controlBatchTraces.length === 1) {
    validateControlTrace(controlBatchTraces[0], reasons, 'control BatchedMesh');
    validatePipelineForTrace(result, controlBatchTraces[0], 0, reasons,
      'control BatchedMesh');
  }

  const arrayBundles = tracesWithDrawMethod(
    phaseTraces(result, 'renderBundles', 'array-camera'),
    'drawIndexedIndirect',
  );
  addReasonUnless(arrayBundles.length === 2, reasons,
    'ArrayCamera must create exactly two drawing bundle encoders');
  for (let layer = 0; layer < arrayBundles.length; layer += 1) {
    validateImmediateTrace(
      arrayBundles[layer],
      'drawIndexedIndirect',
      ordinals,
      reasons,
      `ArrayCamera bundle ${layer}`,
      INDIRECT_OFFSETS,
    );
    validatePipelineForTrace(result, arrayBundles[layer], 4, reasons,
      `ArrayCamera bundle ${layer}`);
  }
  const arrayExecutionIds = bundleExecutionsForPhase(result, 'array-camera');
  addReasonUnless(
    arrayExecutionIds.length === 2
      && new Set(arrayExecutionIds).size === 2
      && arrayBundles.every((trace) => arrayExecutionIds.includes(trace.bundleId)),
    reasons,
    'ArrayCamera layer bundle executions are invalid',
  );

  const scopes = result?.instrumentation?.errorScopes;
  const pushes = scopes?.filter((record) => record?.operation === 'push') ?? [];
  const pops = scopes?.filter((record) => record?.operation === 'pop') ?? [];
  const scopeStack = [];
  let scopeLedgerValid = Array.isArray(scopes);
  let previousScopeSequence = -1;
  if (Array.isArray(scopes)) {
    for (const record of scopes) {
      scopeLedgerValid &&= Number.isSafeInteger(record?.sequence)
        && record.sequence > previousScopeSequence;
      previousScopeSequence = record?.sequence;
      if (record?.operation === 'push') {
        scopeStack.push(record.filter);
        scopeLedgerValid &&= typeof record.filter === 'string'
          && record.depthAfterCall === scopeStack.length;
      } else if (record?.operation === 'pop') {
        const expectedFilter = scopeStack.at(-1) ?? null;
        scopeLedgerValid &&= scopeStack.length > 0
          && record.depthBeforeCall === scopeStack.length
          && record.poppedFilter === expectedFilter;
        scopeStack.pop();
        scopeLedgerValid &&= record.depthAfterCall === scopeStack.length
          && record.settled === true
          && record.resolvedError === null
          && record.rejected === null;
      } else {
        scopeLedgerValid = false;
      }
    }
  }
  const expectedOuterFilters = ['internal', 'out-of-memory', 'validation'];
  const firstOuterPushes = scopes?.slice(0, 3) ?? [];
  const finalOuterPops = scopes?.slice(-3) ?? [];
  addReasonUnless(
    Array.isArray(scopes)
      && scopeLedgerValid
      && scopeStack.length === 0
      && pushes.length === pops.length
      && pushes.length >= 3
      && exactArray(firstOuterPushes.map((record) => record?.filter), expectedOuterFilters)
      && firstOuterPushes.every((record) => record?.operation === 'push')
      && exactArray(
        finalOuterPops.map((record) => record?.poppedFilter),
        [...expectedOuterFilters].reverse(),
      )
      && finalOuterPops.every((record) => record?.operation === 'pop'),
    reasons,
    'instrumented WebGPU error scopes are unbalanced, unsettled, or nonempty',
  );
  addReasonUnless(
    exactArray(result?.errorObservation?.outerFilters, expectedOuterFilters)
      && result?.errorObservation?.includesRendererInitialization === true
      && result?.errorObservation?.includesAllRendersAndReadbacks === true
      && result?.errorObservation?.resourcesDisposedWithinScopes === true
      && result?.errorObservation?.indirectGpuAttributesDeleted === 2
      && result?.errorObservation?.depthBeforeOuterPops === 3
      && result?.errorObservation?.finalDepth === 0
      && result?.errorObservation?.outerResults?.length === 3
      && exactArray(
        result.errorObservation.outerResults.map((record) => record?.filter),
        [...expectedOuterFilters].reverse(),
      )
      && result.errorObservation.outerResults.every((record) => record?.error === null)
      && result?.errorObservation?.intentionalDeviceLoss?.reason === 'destroyed',
    reasons,
    'outer lifecycle scopes, disposal coverage, or intentional loss are invalid',
  );
  const shaderModules = result?.instrumentation?.shaderModules;
  const shaderCompilation = result?.shaderCompilation;
  const shaderModuleIds = Array.isArray(shaderModules)
    ? shaderModules.map((record) => record?.moduleId)
    : [];
  const compilationIds = Array.isArray(shaderCompilation)
    ? shaderCompilation.map((record) => record?.moduleId)
    : [];
  const compilationEvidenceIsComplete = Array.isArray(shaderModules)
    && shaderModules.length > 0
    && shaderModuleIds.every((moduleId) => typeof moduleId === 'string')
    && new Set(shaderModuleIds).size === shaderModuleIds.length
    && Array.isArray(shaderCompilation)
    && shaderCompilation.length === shaderModules.length
    && new Set(compilationIds).size === compilationIds.length
    && shaderModuleIds.every((moduleId) => compilationIds.includes(moduleId))
    && shaderCompilation.every((record) => record?.available === true
      && Array.isArray(record?.messages)
      && record.messages.every((message) => message?.type !== 'error'));
  addReasonUnless(
    result?.threeErrors?.length === 0
      && result?.threeConsoleErrors?.length === 0
      && result?.threeDeviceLosses?.length === 0
      && result?.gpuErrors?.uncaptured?.length === 0
      && result?.gpuErrors?.unexpectedDeviceLosses?.length === 0
      && result?.cleanupFailures?.length === 0
      && compilationEvidenceIsComplete,
    reasons,
    'Three.js, shader compilation, GPU error, or unexpected-loss evidence is nonempty',
  );

  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

async function readProtocol() {
  const bytes = await readFile(protocolPath);
  const value = JSON.parse(bytes.toString('utf8'));
  const canonical = canonicalJsonBytes(value);
  if (!bytes.equals(canonical)
      || bytes.length !== EXPECTED_PROTOCOL_BYTES
      || sha256(bytes) !== EXPECTED_PROTOCOL_SHA256) {
    throw new Error('Frozen protocol bytes are noncanonical or differ from the pinned identity.');
  }
  return {
    bytes,
    sha256: sha256(bytes),
    value,
  };
}

export function validateThreeDrawIndexProtocol(protocol) {
  const reasons = [];
  const canonical = protocol === undefined ? null : canonicalJsonBytes(protocol);
  addReasonUnless(
    canonical !== null
      && canonical.length === EXPECTED_PROTOCOL_BYTES
      && sha256(canonical) === EXPECTED_PROTOCOL_SHA256,
    reasons,
    'protocol canonical byte identity is invalid',
  );
  addReasonUnless(
    protocol?.schemaVersion === 1
      && protocol?.kind === 'three-webgpu-draw-index-default-exposure-protocol'
      && protocol?.executionId === 'draw-index-default-exposure-994260a-v1'
      && protocol?.executionMode === EXPECTED_EXECUTION_MODE
      && protocol?.claimBoundary === EXPECTED_CLAIM_BOUNDARY
      && protocol?.command === EXPECTED_PROTOCOL_COMMAND,
    reasons,
    'protocol header is invalid',
  );
  addReasonUnless(
    protocol?.upstream?.commit === EXPECTED_THREE_DRAW_INDEX_COMMIT
      && protocol?.upstream?.tree === EXPECTED_THREE_DRAW_INDEX_TREE
      && protocol?.upstream?.srcTree === EXPECTED_THREE_DRAW_INDEX_SRC_TREE
      && protocol?.upstream?.runtimeRevision === EXPECTED_THREE_DRAW_INDEX_REVISION,
    reasons,
    'protocol upstream identity is invalid',
  );
  addReasonUnless(
    protocol?.patch?.bytes === EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES
      && protocol?.patch?.sha256 === EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256,
    reasons,
    'protocol patch identity is invalid',
  );
  addReasonUnless(
    protocol?.browser?.requiredExecutablePath
      === 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
      && protocol?.browser?.requiredProductVersion === '152.0.7977.82'
      && protocol?.browser?.requiredByteLength === 4_461_720
      && protocol?.browser?.requiredSha256
        === '8cd23aec3a30479b9b8db2063e70526c88a7ec2c99cd744603eb26ab598733ed'
      && exactArray(protocol?.browser?.playwrightArguments, [])
      && exactArray(
        protocol?.browser?.forbiddenArgumentPrefixes,
        EXPECTED_FORBIDDEN_ARGUMENT_FRAGMENTS,
      ),
    reasons,
    'protocol browser identity or argument policy is invalid',
  );
  addReasonUnless(
    protocol?.output?.reportPath === EXPECTED_REPORT_PATH
      && protocol?.output?.sha256Path === EXPECTED_SIDECAR_PATH
      && protocol?.output?.exclusiveDirectory === true
      && protocol?.output?.overwrite === false
      && protocol?.output?.encoding
        === 'canonical two-space UTF-8 JSON with one trailing LF',
    reasons,
    'protocol immutable output policy is invalid',
  );
  addReasonUnless(
    protocol?.upstream?.repository === 'https://github.com/mrdoob/three.js.git'
      && protocol?.patch?.path === 'patches/three-webgpu-draw-index-immediates.patch'
      && protocol?.page?.path === 'three-draw-index-canary.html'
      && protocol?.page?.module === 'src/three-draw-index-canary.js'
      && protocol?.page?.contract === 'src/three-draw-index-canary-contract.js'
      && protocol?.page?.timeoutMilliseconds === 120_000,
    reasons,
    'protocol repository, patch, or page binding is invalid',
  );
  addReasonUnless(
    exactArray(protocol?.capabilityOrder, EXPECTED_CAPABILITY_ORDER)
      && exactArray(protocol?.assertions, EXPECTED_ASSERTIONS),
    reasons,
    'protocol capability order or assertion set is invalid',
  );
  addReasonUnless(
    protocol?.timingPolicy?.captureTimings === false
      && protocol?.timingPolicy?.captureGpuTimestamps === false
      && protocol?.timingPolicy?.performanceClaim === false
      && protocol?.timingPolicy?.retries === 0
      && protocol?.timingPolicy?.browserLaunches === 1
      && protocol?.timingPolicy?.devices === 1,
    reasons,
    'protocol timing or one-shot policy is invalid',
  );
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

async function collectBoundSourceFiles() {
  const files = await Promise.all(BOUND_SOURCE_PATHS.map(async (relativePath) => {
    const bytes = await readFile(path.join(repoRoot, ...relativePath.split('/')));
    return {
      path: relativePath,
      byteLength: bytes.length,
      sha256: sha256(bytes),
    };
  }));
  return {
    schemaVersion: 1,
    files,
    aggregateSha256: sha256(Buffer.from(JSON.stringify(files), 'utf8')),
  };
}

function boundSourceSnapshotValid(snapshot) {
  const files = snapshot?.files;
  return snapshot?.schemaVersion === 1
    && Array.isArray(files)
    && files.length === BOUND_SOURCE_PATHS.length
    && files.every((record, index) => (
      record?.path === BOUND_SOURCE_PATHS[index]
        && Number.isSafeInteger(record?.byteLength)
        && record.byteLength > 0
        && SHA256_PATTERN.test(record?.sha256)
    ))
    && snapshot?.aggregateSha256
      === sha256(Buffer.from(JSON.stringify(files), 'utf8'));
}

function boundSourcesMatch(start, end) {
  return boundSourceSnapshotValid(start)
    && boundSourceSnapshotValid(end)
    && start?.aggregateSha256 === end?.aggregateSha256
    && JSON.stringify(start?.files) === JSON.stringify(end?.files);
}

async function runGit(args, options = {}) {
  const { stdout } = await execFile('git', args, {
    cwd: repoRoot,
    encoding: options.encoding ?? 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    windowsHide: true,
  });
  return stdout;
}

async function collectGitIdentity() {
  const [commit, tree, branch, statusText, trackedRaw] = await Promise.all([
    runGit(['rev-parse', 'HEAD']),
    runGit(['rev-parse', 'HEAD^{tree}']),
    runGit(['branch', '--show-current']),
    runGit(['status', '--porcelain=v1', '--untracked-files=all']),
    runGit(['ls-files', '-z'], { encoding: 'buffer' }),
  ]);
  const trackedPaths = trackedRaw.toString('utf8').split('\0').filter(Boolean).sort();
  const aggregate = createHash('sha256');
  for (const relativePath of trackedPaths) {
    const bytes = await readFile(path.join(repoRoot, ...relativePath.split('/')));
    aggregate.update(relativePath, 'utf8');
    aggregate.update('\0');
    aggregate.update(bytes);
    aggregate.update('\0');
  }
  return {
    commit: commit.trim(),
    tree: tree.trim(),
    branch: branch.trim(),
    statusPorcelain: statusText,
    clean: statusText.length === 0,
    trackedFileCount: trackedPaths.length,
    trackedWorkingBytesSha256: aggregate.digest('hex'),
  };
}

async function executableIdentity(executablePath) {
  const resolved = path.resolve(executablePath);
  const [metadata, bytes] = await Promise.all([stat(resolved), readFile(resolved)]);
  return {
    path: resolved.replaceAll('\\', '/'),
    byteLength: metadata.size,
    sha256: sha256(bytes),
  };
}

function exactExecutableIdentity(identity, protocol) {
  if (!executableSnapshotValid(identity)
      || typeof protocol?.browser?.requiredExecutablePath !== 'string') return false;
  return identity.path.toLowerCase()
      === protocol.browser.requiredExecutablePath.replaceAll('\\', '/').toLowerCase()
    && identity?.byteLength === protocol.browser.requiredByteLength
    && identity?.sha256 === protocol.browser.requiredSha256;
}

function executableSnapshotValid(identity) {
  return typeof identity?.path === 'string'
    && identity.path.length > 0
    && Number.isSafeInteger(identity?.byteLength)
    && identity.byteLength > 0
    && SHA256_PATTERN.test(identity?.sha256);
}

export function validateThreeDrawIndexOverlayManifest(manifest) {
  const reasons = [];
  addReasonUnless(
    manifest?.schemaVersion === 1
      && manifest?.overlay === 'three-webgpu-draw-index-immediates'
      && manifest?.source?.commit === EXPECTED_THREE_DRAW_INDEX_COMMIT
      && manifest?.source?.tree === EXPECTED_THREE_DRAW_INDEX_TREE
      && manifest?.source?.srcTree === EXPECTED_THREE_DRAW_INDEX_SRC_TREE
      && manifest?.source?.runtimeRevision === EXPECTED_THREE_DRAW_INDEX_REVISION
      && manifest?.source?.cleanTrackedTree === true
      && manifest?.source?.cleanFullWorktree === true
      && manifest?.source?.statusPorcelain === '',
    reasons,
    'overlay source identity is invalid',
  );
  addReasonUnless(
    manifest?.patch?.bytes === EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES
      && manifest?.patch?.sha256 === EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256
      && exactArray(manifest?.patch?.changedPaths, THREE_DRAW_INDEX_TARGET_PATHS),
    reasons,
    'overlay patch identity is invalid',
  );
  const files = manifest?.output?.files;
  const validFiles = Array.isArray(files)
    && files.every((record) => typeof record?.path === 'string'
      && /^(?:100644|100755)$/u.test(record?.gitMode)
      && Number.isSafeInteger(record?.bytes)
      && record.bytes >= 0
      && SHA256_PATTERN.test(record?.sha256));
  const uniquePaths = new Set(validFiles ? files.map((record) => record.path) : []);
  const totalBytes = validFiles
    ? files.reduce((total, record) => total + record.bytes, 0)
    : null;
  const inventoryCommitment = validFiles
    ? files
      .map((file) => `${file.gitMode}\0${file.path}\0${file.bytes}\0${file.sha256}\n`)
      .join('')
    : null;
  const inventorySha256 = inventoryCommitment === null
    ? null
    : sha256(Buffer.from(inventoryCommitment, 'utf8'));
  addReasonUnless(
    manifest?.contract?.wgslRequirement === 'immediate_address_space'
      && manifest?.contract?.immediateSizeBytes === 4
      && manifest?.contract?.shaderSymbol === 'nodeDrawIndex'
      && manifest?.output?.fileCount === EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT
      && validFiles
      && files.length === EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT
      && uniquePaths.size === EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT
      && manifest?.output?.totalBytes === totalBytes
      && totalBytes === EXPECTED_OVERLAY_TOTAL_BYTES
      && manifest?.output?.inventorySha256 === inventorySha256
      && inventorySha256 === EXPECTED_OVERLAY_INVENTORY_SHA256,
    reasons,
    'overlay contract or output inventory is invalid',
  );
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

async function listOverlayFiles(root, relativeDirectory = '') {
  const absoluteDirectory = path.join(root, ...relativeDirectory.split('/').filter(Boolean));
  const entries = await readdir(absoluteDirectory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
  const files = [];
  for (const entry of entries) {
    const relativePath = relativeDirectory === ''
      ? entry.name
      : `${relativeDirectory}/${entry.name}`;
    if (entry.isSymbolicLink()) {
      throw new Error(`Generated overlay contains a symbolic link: ${relativePath}`);
    }
    if (entry.isDirectory()) {
      files.push(...await listOverlayFiles(root, relativePath));
    } else if (entry.isFile()) {
      files.push(relativePath);
    } else {
      throw new Error(`Generated overlay contains a non-file entry: ${relativePath}`);
    }
  }
  return files;
}

async function collectOverlayIdentity(overlayRoot, manifest) {
  const reasons = [];
  const manifestValidation = validateThreeDrawIndexOverlayManifest(manifest);
  addReasonUnless(manifestValidation.valid, reasons,
    `overlay manifest is invalid: ${manifestValidation.reasons.join('; ')}`);
  const expectedPaths = [
    ...manifest.output.files.map((record) => record.path),
    overlayManifestFilename,
  ].sort((left, right) => left.localeCompare(right, 'en'));
  let actualPaths = [];
  try {
    actualPaths = (await listOverlayFiles(overlayRoot))
      .sort((left, right) => left.localeCompare(right, 'en'));
  } catch (error) {
    reasons.push(error.message);
  }
  addReasonUnless(exactArray(actualPaths, expectedPaths), reasons,
    'overlay path inventory differs from its manifest');

  const files = [];
  for (const expected of manifest.output.files) {
    const absolutePath = path.join(overlayRoot, ...expected.path.split('/'));
    try {
      const [metadata, bytes] = await Promise.all([
        lstat(absolutePath),
        readFile(absolutePath),
      ]);
      const record = {
        path: expected.path,
        gitMode: expected.gitMode,
        bytes: bytes.length,
        sha256: sha256(bytes),
      };
      files.push(record);
      addReasonUnless(metadata.isFile() && !metadata.isSymbolicLink(), reasons,
        `overlay path is not a regular file: ${expected.path}`);
      addReasonUnless(
        record.bytes === expected.bytes && record.sha256 === expected.sha256,
        reasons,
        `overlay file identity differs from manifest: ${expected.path}`,
      );
    } catch (error) {
      reasons.push(`cannot read overlay file ${expected.path}: ${error.message}`);
    }
  }
  const commitment = files
    .map((file) => `${file.gitMode}\0${file.path}\0${file.bytes}\0${file.sha256}\n`)
    .join('');
  const inventorySha256 = sha256(Buffer.from(commitment, 'utf8'));
  const totalBytes = files.reduce((total, file) => total + file.bytes, 0);
  addReasonUnless(
    files.length === manifest.output.fileCount
      && totalBytes === manifest.output.totalBytes
      && inventorySha256 === manifest.output.inventorySha256,
    reasons,
    'overlay disk aggregate differs from its manifest',
  );

  let manifestSha256 = null;
  try {
    const manifestBytes = await readFile(path.join(overlayRoot, overlayManifestFilename));
    manifestSha256 = sha256(manifestBytes);
    addReasonUnless(manifestBytes.equals(canonicalJsonBytes(manifest)), reasons,
      'overlay manifest bytes are not canonical or differ from the parsed manifest');
  } catch (error) {
    reasons.push(`cannot read overlay manifest: ${error.message}`);
  }
  return {
    valid: reasons.length === 0,
    reasons: [...new Set(reasons)],
    fileCount: files.length,
    totalBytes,
    inventorySha256,
    manifestSha256,
    pathCount: actualPaths.length,
  };
}

function normalizedModuleId(id) {
  let value = id.split('?')[0];
  if (value.startsWith('/@fs/')) value = value.slice('/@fs/'.length);
  if (/^\/[A-Za-z]:\//u.test(value)) value = value.slice(1);
  return path.normalize(path.resolve(value));
}

function comparablePath(value) {
  const normalized = path.normalize(path.resolve(value));
  return process.platform === 'win32' ? normalized.toLowerCase() : normalized;
}

function sourceObserverPlugin(overlayRoot, manifest, observations) {
  const normalizedOverlayRoot = path.normalize(path.resolve(overlayRoot));
  const pageTargets = new Map([
    [comparablePath(path.join(repoRoot, 'src', 'three-draw-index-canary.js')), 'page'],
    [comparablePath(path.join(repoRoot, 'src', 'three-draw-index-canary-contract.js')), 'contract'],
  ]);
  const overlayTargets = new Map(manifest.output.files.map((record) => [
    comparablePath(path.join(normalizedOverlayRoot, ...record.path.split('/'))),
    record.path,
  ]));
  return {
    name: 'three-draw-index-source-observer',
    enforce: 'pre',
    transform(code, id) {
      const normalized = normalizedModuleId(id);
      const identityKey = comparablePath(normalized);
      let key = pageTargets.get(identityKey);
      const overlayPath = overlayTargets.get(identityKey) ?? null;
      if (overlayPath !== null) {
        key = `overlay:${overlayPath}`;
      }
      if (key === undefined) return null;
      const previous = observations.get(key);
      observations.set(key, {
        key,
        resolvedId: normalized.replaceAll('\\', '/'),
        overlayPath,
        inputSha256: sha256(Buffer.from(code, 'utf8')),
        transformCount: (previous?.transformCount ?? 0) + 1,
      });
      return null;
    },
  };
}

function validateServedSource({
  observations,
  manifest,
  resources,
  sourceFiles,
  expectedOrigin,
}) {
  const reasons = [];
  const inventory = new Map(manifest.output.files.map((record) => [record.path, record]));
  const records = [...observations.values()].sort((left, right) => (
    left.key.localeCompare(right.key)
  ));
  const overlayRecords = records.filter((record) => record.overlayPath !== null);
  const sourceHash = (relativePath) => sourceFiles.files.find(
    (record) => record.path === relativePath,
  )?.sha256;
  addReasonUnless(
    overlayRecords.length > 0
      && overlayRecords.every((record) => record.transformCount === 1
        && inventory.get(record.overlayPath)?.sha256 === record.inputSha256)
      && overlayRecords.some((record) => record.overlayPath === 'src/Three.WebGPU.js')
      && overlayRecords.some((record) => record.overlayPath === 'src/Three.TSL.js'),
    reasons,
    'served overlay module inputs do not match the generated inventory',
  );
  const pageRecord = observations.get('page');
  const contractRecord = observations.get('contract');
  addReasonUnless(
    pageRecord?.transformCount === 1
      && pageRecord?.inputSha256 === sourceHash('src/three-draw-index-canary.js')
      && contractRecord?.transformCount === 1
      && contractRecord?.inputSha256 === sourceHash('src/three-draw-index-canary-contract.js'),
    reasons,
    'served page or contract input does not match its bound source',
  );
  addReasonUnless(
    Array.isArray(resources)
      && resources.length > 0
      && resources.every((url) => {
        const parsed = new URL(url);
        return parsed.origin === expectedOrigin;
      })
      && resources.every((url) => !/node_modules[/\\]three|upstream-three|three-immediate-/iu
        .test(decodeURIComponent(url))),
    reasons,
    'loaded resource URLs are non-loopback or include a foreign Three.js runtime',
  );
  return {
    exact: reasons.length === 0,
    reasons: [...new Set(reasons)],
    observedModules: records,
    resourceUrls: resources,
  };
}

function installPageObservers(page) {
  const observations = {
    pageErrors: [],
    consoleErrors: [],
    requestFailures: [],
    httpErrors: [],
    crashes: [],
    unhandledRejections: [],
  };
  page.on('pageerror', (error) => observations.pageErrors.push(serializeError(error)));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      observations.consoleErrors.push({
        text: message.text().slice(0, 4_096),
        location: message.location(),
      });
    }
  });
  page.on('requestfailed', (request) => observations.requestFailures.push({
    url: request.url(),
    method: request.method(),
    failure: request.failure(),
  }));
  page.on('response', (response) => {
    if (response.status() >= 400) {
      observations.httpErrors.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText(),
      });
    }
  });
  page.on('crash', () => observations.crashes.push({ crashed: true }));
  return observations;
}

function pageObservationsAreStructured(observations) {
  const expectedKeys = [
    'pageErrors',
    'consoleErrors',
    'requestFailures',
    'httpErrors',
    'crashes',
    'unhandledRejections',
  ];
  return observations !== null
    && typeof observations === 'object'
    && exactArray(Object.keys(observations).sort(), [...expectedKeys].sort())
    && expectedKeys.every((key) => Array.isArray(observations[key]));
}

function pageObservationsAreClean(observations) {
  return pageObservationsAreStructured(observations)
    && Object.values(observations).every((records) => records.length === 0);
}

async function captureEffectiveBrowserArguments(browser) {
  try {
    const session = await browser.newBrowserCDPSession();
    const result = await session.send('Browser.getBrowserCommandLine');
    await session.detach();
    return {
      api: 'Browser.getBrowserCommandLine',
      available: true,
      arguments: result.arguments ?? [],
    };
  } catch (error) {
    return {
      api: 'Browser.getBrowserCommandLine',
      available: false,
      arguments: [],
      failure: serializeError(error),
    };
  }
}

export function findForbiddenThreeDrawIndexBrowserArguments(arguments_, protocol) {
  if (!Array.isArray(arguments_)) return ['<effective-arguments-unavailable>'];
  return arguments_.filter((argument) => {
    if (typeof argument !== 'string') return true;
    const normalized = argument.toLowerCase();
    return protocol.browser.forbiddenArgumentPrefixes.some((prefix) => (
      normalized === prefix || normalized.startsWith(`${prefix}=`)
    ));
  });
}

function structuralGitSnapshotValid(snapshot) {
  return /^[a-f0-9]{40}$/u.test(snapshot?.commit)
    && !/^0{40}$/u.test(snapshot.commit)
    && /^[a-f0-9]{40}$/u.test(snapshot?.tree)
    && !/^0{40}$/u.test(snapshot.tree)
    && typeof snapshot?.branch === 'string'
    && typeof snapshot?.statusPorcelain === 'string'
    && typeof snapshot?.clean === 'boolean'
    && snapshot.clean === (snapshot.statusPorcelain.length === 0)
    && Number.isSafeInteger(snapshot?.trackedFileCount)
    && snapshot.trackedFileCount > 0
    && SHA256_PATTERN.test(snapshot?.trackedWorkingBytesSha256);
}

function gitSnapshotValid(snapshot) {
  return structuralGitSnapshotValid(snapshot)
    && snapshot?.branch === 'experiment/webgpu-draw-index-immediates'
    && snapshot?.statusPorcelain === ''
    && snapshot?.clean === true;
}

function exactGitIdentity(left, right) {
  return gitSnapshotValid(left)
    && gitSnapshotValid(right)
    && left?.commit === right?.commit
    && left?.tree === right?.tree
    && left?.branch === right?.branch
    && left?.statusPorcelain === right?.statusPorcelain
    && left?.clean === true
    && right?.clean === true
    && left?.trackedFileCount === right?.trackedFileCount
    && left?.trackedWorkingBytesSha256 === right?.trackedWorkingBytesSha256;
}

function overlayIdentityMatchesManifest(identity, manifest, manifestSha256) {
  return identity?.valid === true
    && exactArray(identity?.reasons, [])
    && identity?.fileCount === manifest?.output?.fileCount
    && identity?.totalBytes === manifest?.output?.totalBytes
    && identity?.inventorySha256 === manifest?.output?.inventorySha256
    && identity?.manifestSha256 === manifestSha256
    && identity?.pathCount === manifest?.output?.fileCount + 1;
}

function structuralOverlayIdentityValid(identity) {
  const hashesAreStructured = (identity?.inventorySha256 === null
    || SHA256_PATTERN.test(identity?.inventorySha256))
    && (identity?.manifestSha256 === null
      || SHA256_PATTERN.test(identity?.manifestSha256));
  return typeof identity?.valid === 'boolean'
    && Array.isArray(identity?.reasons)
    && Number.isSafeInteger(identity?.fileCount)
    && identity.fileCount >= 0
    && Number.isSafeInteger(identity?.totalBytes)
    && identity.totalBytes >= 0
    && Number.isSafeInteger(identity?.pathCount)
    && identity.pathCount >= 0
    && hashesAreStructured
    && (identity.valid ? identity.reasons.length === 0 : identity.reasons.length > 0);
}

export function validateThreeDrawIndexReport(report) {
  const reasons = [];
  addReasonUnless(
    report?.schemaVersion === 1
      && report?.kind === THREE_DRAW_INDEX_REPORT_KIND
      && ['passed', 'unsupported', 'failed'].includes(report?.status)
      && report?.executionMode === EXPECTED_EXECUTION_MODE
      && report?.timingCaptured === false
      && report?.benchmarkClaim === false
      && report?.claimBoundary === EXPECTED_CLAIM_BOUNDARY,
    reasons,
    'report header or claim boundary is invalid',
  );
  const protocolValue = report?.protocol?.value;
  const protocolValidation = validateThreeDrawIndexProtocol(protocolValue);
  const protocolBytes = protocolValue === undefined
    ? null
    : canonicalJsonBytes(protocolValue);
  addReasonUnless(
    protocolValidation.valid
      && protocolBytes !== null
      && report?.protocol?.byteLength === protocolBytes.length
      && report?.protocol?.sha256 === sha256(protocolBytes)
      && report?.protocol?.path === 'protocols/three-draw-index-canary-v1.json',
    reasons,
    'bound protocol is invalid',
  );
  addReasonUnless(
    report?.executionPolicy?.attemptCount === 1
      && report?.executionPolicy?.retryCount === 0
      && report?.executionPolicy?.browserLaunchCount === 1
      && report?.executionPolicy?.deviceCountLimit === 1
      && report?.executionPolicy?.customBrowserArguments?.length === 0
      && report?.executionPolicy?.timingCaptured === false,
    reasons,
    'one-shot execution policy is invalid',
  );
  addReasonUnless(
    report?.browser?.launchApi === 'playwright-core.chromium.launch'
      && report?.browser?.headless === true
      && exactArray(report?.browser?.customArguments, [])
      && report?.server?.implementation === 'vite.createServer'
      && report?.server?.configFile === false
      && report?.server?.host === '127.0.0.1'
      && report?.server?.localOnly === true
      && JSON.stringify(report?.server?.isolationHeaders) === JSON.stringify(ISOLATION_HEADERS),
    reasons,
    'runner browser/server static policy is invalid',
  );
  const lifecycle = report?.observedLifecycle;
  addReasonUnless(
    Number.isInteger(lifecycle?.serverStartAttempts)
      && lifecycle.serverStartAttempts >= 0
      && lifecycle.serverStartAttempts <= 1
      && Number.isInteger(lifecycle?.serverInstancesCreated)
      && lifecycle.serverInstancesCreated >= 0
      && lifecycle.serverInstancesCreated <= lifecycle.serverStartAttempts
      && Number.isInteger(lifecycle?.serverStartSuccesses)
      && lifecycle.serverStartSuccesses >= 0
      && lifecycle.serverStartSuccesses <= lifecycle.serverInstancesCreated
      && Number.isInteger(lifecycle?.browserLaunchAttempts)
      && lifecycle.browserLaunchAttempts >= 0
      && lifecycle.browserLaunchAttempts <= 1
      && lifecycle.browserLaunchAttempts <= lifecycle.serverStartSuccesses
      && Number.isInteger(lifecycle?.browserLaunchSuccesses)
      && lifecycle.browserLaunchSuccesses >= 0
      && lifecycle.browserLaunchSuccesses <= lifecycle.browserLaunchAttempts
      && Number.isInteger(lifecycle?.contextsCreated)
      && lifecycle.contextsCreated >= 0
      && lifecycle.contextsCreated <= lifecycle.browserLaunchSuccesses
      && Number.isInteger(lifecycle?.pagesCreated)
      && lifecycle.pagesCreated >= 0
      && lifecycle.pagesCreated <= lifecycle.contextsCreated
      && lifecycle?.postflightChecksAttempted === 4
      && Number.isInteger(lifecycle?.postflightChecksSucceeded)
      && lifecycle.postflightChecksSucceeded >= 0
      && lifecycle.postflightChecksSucceeded <= 4,
    reasons,
    'observed lifecycle counts are invalid',
  );
  addReasonUnless(
    report?.artifactPolicy?.reportPath === EXPECTED_REPORT_PATH
      && report?.artifactPolicy?.sha256Path === EXPECTED_SIDECAR_PATH
      && report?.artifactPolicy?.exclusiveDirectory === true
      && report?.artifactPolicy?.overwrite === false,
    reasons,
    'artifact policy differs from the frozen protocol',
  );
  const overlayManifest = report?.overlay?.manifest;
  const overlayValidation = validateThreeDrawIndexOverlayManifest(overlayManifest);
  const overlayManifestBytes = overlayManifest === undefined
    ? null
    : canonicalJsonBytes(overlayManifest);
  const overlayManifestSha256 = overlayManifestBytes === null
    ? null
    : sha256(overlayManifestBytes);
  addReasonUnless(
    overlayValidation.valid
      && JSON.stringify(report?.overlay?.validation) === JSON.stringify(overlayValidation)
      && report?.overlay?.manifestSha256 === overlayManifestSha256
      && overlayIdentityMatchesManifest(
        report?.overlay?.identityStart,
        overlayManifest,
        overlayManifestSha256,
      ),
    reasons,
    'generated overlay preflight evidence is invalid',
  );
  if (report?.status === 'failed') {
    const runnerFailures = report?.runnerFailures;
    const allowedFailureStages = new Set([
      ...EXECUTION_FAILURE_STAGES,
      'context-close',
      'browser-close',
      'server-close',
      'postflight-bound-sources',
      'postflight-git',
      'postflight-browser-executable',
      'postflight-overlay',
      'postflight-identity-mismatch',
      'final-report-validation',
    ]);
    const hasFailureStage = (stage) => Array.isArray(runnerFailures)
      && runnerFailures.some((record) => record?.stage === stage);
    const mainFailureStage = runnerFailures?.find(
      (record) => EXECUTION_FAILURE_STAGES.includes(record?.stage),
    )?.stage ?? null;
    const mainFailureRank = mainFailureStage === null
      ? EXECUTION_FAILURE_STAGES.length
      : EXECUTION_FAILURE_STAGES.indexOf(mainFailureStage);
    const runnerFailuresAreExact = Array.isArray(runnerFailures)
      && runnerFailures.length > 0
      && runnerFailures.every((record) => allowedFailureStages.has(record?.stage)
        && typeof record?.failure?.name === 'string'
        && typeof record?.failure?.message === 'string')
      && JSON.stringify(report?.failure) === JSON.stringify(runnerFailures[0]?.failure);
    const filesEnd = report?.sourceIdentity?.filesEnd;
    const gitEnd = report?.sourceIdentity?.gitEnd;
    const executableEnd = report?.browser?.executableEnd;
    const overlayEnd = report?.overlay?.identityEnd;
    const filesMatch = filesEnd !== null
      && boundSourcesMatch(report?.sourceIdentity?.filesStart, filesEnd);
    const gitMatch = gitEnd !== null
      && exactGitIdentity(report?.sourceIdentity?.gitStart, gitEnd);
    const executableMatch = executableEnd !== null
      && JSON.stringify(report?.browser?.executableStart) === JSON.stringify(executableEnd)
      && exactExecutableIdentity(executableEnd, protocolValue);
    const overlayMatch = overlayEnd !== null
      && JSON.stringify(report?.overlay?.identityStart) === JSON.stringify(overlayEnd);
    const successfulPostflightRecords = [filesEnd, gitEnd, executableEnd, overlayEnd]
      .filter((record) => record !== null).length;
    const postflightRecordsAreExact = (filesEnd !== null
      ? boundSourceSnapshotValid(filesEnd)
      : hasFailureStage('postflight-bound-sources'))
      && (gitEnd !== null
        ? structuralGitSnapshotValid(gitEnd)
        : hasFailureStage('postflight-git'))
      && (executableEnd !== null
        ? executableSnapshotValid(executableEnd)
        : hasFailureStage('postflight-browser-executable'))
      && (overlayEnd !== null
        ? structuralOverlayIdentityValid(overlayEnd)
        : hasFailureStage('postflight-overlay'))
      && report?.sourceIdentity?.filesMatch === filesMatch
      && report?.sourceIdentity?.gitMatch === gitMatch
      && report?.browser?.identityMatchesStart === executableMatch
      && report?.overlay?.identityMatch === overlayMatch
      && lifecycle?.postflightChecksSucceeded === successfulPostflightRecords;
    const lifecyclePrefixIsExact = lifecycle?.serverStartAttempts === 1
      && lifecycle?.serverInstancesCreated === Number(mainFailureRank > 0)
      && lifecycle?.serverStartSuccesses === Number(mainFailureRank > 1)
      && lifecycle?.browserLaunchAttempts === Number(mainFailureRank >= 3)
      && lifecycle?.browserLaunchSuccesses === Number(mainFailureRank > 3)
      && lifecycle?.contextsCreated === Number(mainFailureRank > 7)
      && lifecycle?.pagesCreated === Number(mainFailureRank > 9);
    const browserLaunched = lifecycle?.browserLaunchSuccesses === 1;
    const browserStageIsExact = browserLaunched
      ? ((mainFailureStage === 'browser-version'
        ? report?.browser?.productVersion === null
        : typeof report?.browser?.productVersion === 'string')
        && (['browser-version', 'browser-command-line'].includes(mainFailureStage)
          ? (report?.browser?.effectiveArguments === null
            && report?.browser?.forbiddenEffectiveArguments === null)
          : (report?.browser?.effectiveArguments?.api === 'Browser.getBrowserCommandLine'
            && typeof report?.browser?.effectiveArguments?.available === 'boolean'
            && Array.isArray(report?.browser?.effectiveArguments?.arguments)
            && exactArray(
              report?.browser?.forbiddenEffectiveArguments,
              findForbiddenThreeDrawIndexBrowserArguments(
                report.browser.effectiveArguments.arguments,
                protocolValue,
              ),
            )))
        && report?.browser?.closedCleanly === !hasFailureStage('browser-close'))
      : (report?.browser?.productVersion === null
        && report?.browser?.effectiveArguments === null
        && report?.browser?.forbiddenEffectiveArguments === null
        && report?.browser?.closedCleanly === false);
    const contextStageIsExact = lifecycle?.contextsCreated === 1
      ? report?.browser?.contextClosedCleanly === !hasFailureStage('context-close')
      : report?.browser?.contextClosedCleanly === false;
    const serverStageIsExact = (lifecycle?.serverStartSuccesses === 1
      ? (mainFailureStage === 'server-address'
        ? report?.server?.port === null
        : (Number.isInteger(report?.server?.port) && report.server.port > 0))
      : report?.server?.port === null)
      && report?.server?.closedCleanly === (
        lifecycle?.serverInstancesCreated === 1 && !hasFailureStage('server-close')
      );
    const observationsStageIsExact = lifecycle?.pagesCreated === 0
      ? (report?.browserObservations === null
        && report?.browserObservationsClean === false)
      : (mainFailureStage === 'page-observers'
        ? (report?.browserObservations === null
          && report?.browserObservationsClean === false)
        : (pageObservationsAreStructured(report?.browserObservations)
          && report?.browserObservationsClean
            === pageObservationsAreClean(report.browserObservations)));
    let recomputedPageValidation = null;
    let partialPageEvidenceIsExact = true;
    if (report?.pageResult === null) {
      partialPageEvidenceIsExact = report?.pageValidation === null
        && report?.servedSource === null;
    } else if (report?.pageValidation !== null) {
      try {
        recomputedPageValidation = validateThreeDrawIndexPageResult(report.pageResult);
        partialPageEvidenceIsExact = JSON.stringify(report.pageValidation)
          === JSON.stringify(recomputedPageValidation);
      } catch {
        partialPageEvidenceIsExact = false;
      }
    }
    if (report?.servedSource !== null) {
      partialPageEvidenceIsExact &&= typeof report.servedSource?.exact === 'boolean'
        && Array.isArray(report.servedSource?.reasons)
        && Array.isArray(report.servedSource?.observedModules)
        && Array.isArray(report.servedSource?.resourceUrls)
        && (report.servedSource.exact
          ? report.servedSource.reasons.length === 0
          : report.servedSource.reasons.length > 0);
    }
    addReasonUnless(
      typeof report?.failure?.name === 'string'
        && typeof report?.failure?.message === 'string'
        && runnerFailuresAreExact
        && boundSourceSnapshotValid(report?.sourceIdentity?.filesStart)
        && gitSnapshotValid(report?.sourceIdentity?.gitStart)
        && exactExecutableIdentity(report?.browser?.executableStart, protocolValue)
        && lifecyclePrefixIsExact
        && postflightRecordsAreExact
        && browserStageIsExact
        && contextStageIsExact
        && serverStageIsExact
        && observationsStageIsExact
        && partialPageEvidenceIsExact,
      reasons,
      'failed report lacks exact stage-aware failure or pre/postflight evidence',
    );
    return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
  }
  addReasonUnless(
    boundSourcesMatch(report?.sourceIdentity?.filesStart, report?.sourceIdentity?.filesEnd)
      && report?.sourceIdentity?.filesMatch === true
      && exactGitIdentity(report?.sourceIdentity?.gitStart, report?.sourceIdentity?.gitEnd)
      && report?.sourceIdentity?.gitMatch === true,
    reasons,
    'repository or bound source identity changed during execution',
  );
  addReasonUnless(
    lifecycle?.serverStartAttempts === 1
      && lifecycle?.serverInstancesCreated === 1
      && lifecycle?.serverStartSuccesses === 1
      && lifecycle?.browserLaunchAttempts === 1
      && lifecycle?.browserLaunchSuccesses === 1
      && lifecycle?.contextsCreated === 1
      && lifecycle?.pagesCreated === 1
      && lifecycle?.postflightChecksAttempted === 4
      && lifecycle?.postflightChecksSucceeded === 4
      && exactArray(report?.runnerFailures, []),
    reasons,
    'completed report does not prove exactly one server/browser/context/page lifecycle',
  );
  addReasonUnless(
    overlayIdentityMatchesManifest(
      report?.overlay?.identityEnd,
      overlayManifest,
      overlayManifestSha256,
    )
      && report?.overlay?.identityMatch === true
      && JSON.stringify(report?.overlay?.identityStart)
        === JSON.stringify(report?.overlay?.identityEnd),
    reasons,
    'generated overlay identity changed during execution',
  );
  const protocol = protocolValue;
  addReasonUnless(
    exactExecutableIdentity(report?.browser?.executableStart, protocol)
      && exactExecutableIdentity(report?.browser?.executableEnd, protocol)
      && report?.browser?.identityMatchesStart === true
      && report?.browser?.productVersion === protocol?.browser?.requiredProductVersion
      && report?.browser?.launchApi === 'playwright-core.chromium.launch'
      && report?.browser?.headless === true
      && exactArray(report?.browser?.customArguments, [])
      && report?.browser?.effectiveArguments?.available === true
      && exactArray(
        report?.browser?.forbiddenEffectiveArguments,
        findForbiddenThreeDrawIndexBrowserArguments(
          report?.browser?.effectiveArguments?.arguments,
          protocol,
        ),
      )
      && report?.browser?.forbiddenEffectiveArguments?.length === 0
      && report?.browser?.contextClosedCleanly === true
      && report?.browser?.closedCleanly === true,
    reasons,
    'browser identity, arguments, or teardown are invalid',
  );
  addReasonUnless(
    report?.server?.host === '127.0.0.1'
      && Number.isInteger(report?.server?.port)
      && report.server.port > 0
      && report?.server?.configFile === false
      && report?.server?.localOnly === true
      && report?.server?.closedCleanly === true,
    reasons,
    'loopback server evidence is invalid',
  );
  addReasonUnless(
    pageObservationsAreClean(report?.browserObservations)
      && report?.browserObservationsClean === true,
    reasons,
    'browser observations contain an error',
  );
  const pageValidation = validateThreeDrawIndexPageResult(report?.pageResult);
  addReasonUnless(
    pageValidation.valid
      && JSON.stringify(report?.pageValidation) === JSON.stringify(pageValidation)
      && report?.servedSource?.exact === true
      && exactArray(report?.servedSource?.reasons, []),
    reasons,
    'page or served-source validation is invalid',
  );

  if (report?.status === 'passed') {
    addReasonUnless(
      report?.failure === null && report?.pageResult?.status === 'passed',
      reasons,
      'passing report contains a failure or non-passing page',
    );
  } else if (report?.status === 'unsupported') {
    addReasonUnless(
      report?.failure === null && report?.pageResult?.status === 'unsupported',
      reasons,
      'unsupported report is not a clean capability exit',
    );
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function resolvedArtifactPaths(protocol) {
  const reportPath = path.resolve(repoRoot, ...protocol.output.reportPath.split('/'));
  const sidecarPath = path.resolve(repoRoot, ...protocol.output.sha256Path.split('/'));
  const runDirectory = path.dirname(reportPath);
  const relativeReport = path.relative(repoRoot, reportPath).replaceAll('\\', '/');
  const relativeSidecar = path.relative(repoRoot, sidecarPath).replaceAll('\\', '/');
  if (relativeReport !== protocol.output.reportPath
      || relativeSidecar !== protocol.output.sha256Path
      || path.dirname(sidecarPath) !== runDirectory
      || path.basename(reportPath) !== 'report.json'
      || path.basename(sidecarPath) !== 'report.json.sha256') {
    throw new Error('Protocol artifact paths escape or violate the frozen run directory.');
  }
  return { reportPath, sidecarPath, runDirectory };
}

function samePath(left, right) {
  const resolvedLeft = path.resolve(left);
  const resolvedRight = path.resolve(right);
  return process.platform === 'win32'
    ? resolvedLeft.toLowerCase() === resolvedRight.toLowerCase()
    : resolvedLeft === resolvedRight;
}

function pathIsWithin(candidate, allowedRoot) {
  const relative = path.relative(path.resolve(allowedRoot), path.resolve(candidate));
  return relative !== ''
    && relative !== '..'
    && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative);
}

function sameFilesystemObject(left, right) {
  return left?.isFile() === true
    && right?.isFile() === true
    && left.size === right.size
    && (left.dev === 0 || right.dev === 0 || left.dev === right.dev)
    && (left.ino === 0 || right.ino === 0 || left.ino === right.ino);
}

async function assertUnaliasedPathWithin(filename, allowedRoot) {
  const resolvedRoot = path.resolve(allowedRoot);
  const resolved = path.resolve(filename);
  if (!pathIsWithin(resolved, resolvedRoot)) {
    throw new Error(`Artifact path escapes its allowed root: ${resolved}`);
  }
  const components = path.relative(resolvedRoot, resolved).split(path.sep);
  for (let index = 0, current = resolvedRoot; index <= components.length; index += 1) {
    if (index > 0) current = path.join(current, components[index - 1]);
    const metadata = await lstat(current);
    if (metadata.isSymbolicLink()) {
      throw new Error(`Artifact path contains a symbolic link or junction: ${current}`);
    }
    const canonical = await realpath(current);
    if (!samePath(canonical, current)) {
      throw new Error(`Artifact path contains an aliased ancestor: ${current}`);
    }
  }
}

async function readRegularFileSecurely(filename, allowedRoot) {
  const resolved = path.resolve(filename);
  await assertUnaliasedPathWithin(resolved, allowedRoot);
  const before = await lstat(resolved);
  if (!before.isFile() || before.isSymbolicLink()) {
    throw new Error(`Expected one regular non-symlink file: ${resolved}`);
  }
  const handle = await open(resolved, 'r');
  let handleMetadata;
  let bytes;
  try {
    handleMetadata = await handle.stat();
    bytes = await handle.readFile();
  } finally {
    await handle.close();
  }
  const after = await lstat(resolved);
  await assertUnaliasedPathWithin(resolved, allowedRoot);
  if (!sameFilesystemObject(before, handleMetadata)
      || !sameFilesystemObject(handleMetadata, after)
      || bytes.length !== after.size) {
    throw new Error(`File identity changed while it was being read: ${resolved}`);
  }
  return { resolved, bytes, metadata: after };
}

async function assertArtifactDirectory(paths, expectedNames) {
  await assertUnaliasedPathWithin(paths.runDirectory, repoRoot);
  const directoryMetadata = await lstat(paths.runDirectory);
  if (!directoryMetadata.isDirectory() || directoryMetadata.isSymbolicLink()) {
    throw new Error('Artifact run path is not a regular directory.');
  }
  const entries = await readdir(paths.runDirectory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
  const sortedExpected = [...expectedNames].sort((left, right) => (
    left.localeCompare(right, 'en')
  ));
  if (!exactArray(entries.map((entry) => entry.name), sortedExpected)
      || entries.some((entry) => !entry.isFile() || entry.isSymbolicLink())) {
    throw new Error('Artifact directory entries differ from the exact regular-file set.');
  }
}

async function reserveArtifactDirectory(protocol) {
  const paths = resolvedArtifactPaths(protocol);
  await mkdir(path.dirname(paths.runDirectory), { recursive: true });
  await assertUnaliasedPathWithin(path.dirname(paths.runDirectory), repoRoot);
  await mkdir(paths.runDirectory, { recursive: false });
  await assertArtifactDirectory(paths, []);
  return paths;
}

async function writeImmutableArtifact(paths, report) {
  await assertArtifactDirectory(paths, []);
  if (!isStrictJsonData(report)) {
    throw new Error('Report contains non-finite, non-plain, undefined, or cyclic data.');
  }
  const bytes = canonicalJsonBytes(report);
  const roundTripped = JSON.parse(bytes.toString('utf8'));
  const roundTripValidation = validateThreeDrawIndexReport(roundTripped);
  if (!roundTripValidation.valid
      || JSON.stringify(roundTripped) !== JSON.stringify(report)) {
    throw new Error(
      `JSON-round-tripped report failed before publication: ${roundTripValidation.reasons.join('; ')}`,
    );
  }
  await writeFile(paths.reportPath, bytes, { flag: 'wx', mode: 0o444 });
  await chmod(paths.reportPath, 0o444);
  const firstRead = await readRegularFileSecurely(paths.reportPath, repoRoot);
  if (!firstRead.bytes.equals(bytes)) {
    throw new Error('Report reread differs from written bytes.');
  }
  const digest = sha256(bytes);
  const sidecarBytes = Buffer.from(`${digest}  report.json\n`, 'utf8');
  await writeFile(paths.sidecarPath, sidecarBytes, { flag: 'wx', mode: 0o444 });
  await chmod(paths.sidecarPath, 0o444);
  await assertArtifactDirectory(paths, ['report.json', 'report.json.sha256']);
  const [finalReport, finalSidecar] = await Promise.all([
    readRegularFileSecurely(paths.reportPath, repoRoot),
    readRegularFileSecurely(paths.sidecarPath, repoRoot),
  ]);
  if (!finalReport.bytes.equals(bytes) || !finalSidecar.bytes.equals(sidecarBytes)) {
    throw new Error('Immutable artifact changed during final reread.');
  }
  return {
    reportPath: paths.reportPath,
    sidecarPath: paths.sidecarPath,
    byteLength: bytes.length,
    sha256: digest,
    reportWritableBits: finalReport.metadata.mode & 0o222,
    sidecarWritableBits: finalSidecar.metadata.mode & 0o222,
  };
}

export async function verifyThreeDrawIndexArtifact(
  reportPath = null,
  suppliedProtocol = null,
) {
  try {
    const protocolRecord = suppliedProtocol === null ? await readProtocol() : suppliedProtocol;
    const protocol = protocolRecord?.value;
    const protocolBytes = protocol === undefined ? null : canonicalJsonBytes(protocol);
    if (protocolBytes === null
        || !Buffer.isBuffer(protocolRecord?.bytes)
        || !protocolRecord.bytes.equals(protocolBytes)
        || protocolRecord?.sha256 !== sha256(protocolBytes)
        || !validateThreeDrawIndexProtocol(protocol).valid) {
      return { valid: false, reasons: ['supplied/current protocol record is not exact'] };
    }
    const paths = resolvedArtifactPaths(protocol);
    if (reportPath !== null && !samePath(reportPath, paths.reportPath)) {
      return { valid: false, reasons: ['artifact path differs from frozen protocol path'] };
    }
    await assertArtifactDirectory(paths, ['report.json', 'report.json.sha256']);
    const [reportFile, sidecarFile] = await Promise.all([
      readRegularFileSecurely(paths.reportPath, repoRoot),
      readRegularFileSecurely(paths.sidecarPath, repoRoot),
    ]);
    const reportBytes = reportFile.bytes;
    const sidecarBytes = sidecarFile.bytes;
    let report;
    try {
      report = JSON.parse(reportBytes.toString('utf8'));
    } catch (error) {
      return { valid: false, reasons: [`report JSON parse failed: ${error.message}`] };
    }
    const canonical = canonicalJsonBytes(report);
    const digest = sha256(reportBytes);
    const expectedSidecar = `${digest}  report.json\n`;
    const reportValidation = validateThreeDrawIndexReport(report);
    const reasons = [];
    addReasonUnless(reportBytes.equals(canonical), reasons,
      'report is not canonical two-space UTF-8/LF JSON');
    addReasonUnless(sidecarBytes.toString('utf8') === expectedSidecar, reasons,
      'SHA-256 sidecar does not bind the exact report bytes');
    addReasonUnless((reportFile.metadata.mode & 0o222) === 0
      && (sidecarFile.metadata.mode & 0o222) === 0, reasons,
    'artifact files retain writable permission bits');
    addReasonUnless(
      report?.protocol?.byteLength === protocolRecord.bytes.length
        && report?.protocol?.sha256 === protocolRecord.sha256
        && JSON.stringify(report?.protocol?.value) === JSON.stringify(protocolRecord.value),
      reasons,
      'report protocol record differs from the supplied/current protocol bytes',
    );
    addReasonUnless(reportValidation.valid, reasons,
      `report validation failed: ${reportValidation.reasons.join('; ')}`);
    return {
      valid: reasons.length === 0,
      reasons: [...new Set(reasons)],
      report,
      byteLength: reportBytes.length,
      sha256: digest,
      reportPath: paths.reportPath,
      sidecarPath: paths.sidecarPath,
    };
  } catch (error) {
    return {
      valid: false,
      reasons: [`artifact verification I/O/path failure: ${error.message}`],
    };
  }
}

export function parseThreeDrawIndexArguments(arguments_) {
  let browserPath = null;
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === '--browser') {
      const value = arguments_[++index];
      if (value === undefined) throw new Error('--browser requires an exact path.');
      if (browserPath !== null) throw new Error('--browser may be specified only once.');
      browserPath = path.resolve(value);
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  if (browserPath === null) throw new Error('--browser is required.');
  return { browserPath };
}

export async function runThreeDrawIndexProbe({ browserPath } = {}) {
  if (typeof browserPath !== 'string') throw new Error('An exact browserPath is required.');
  const protocolRecord = await readProtocol();
  const protocolValidation = validateThreeDrawIndexProtocol(protocolRecord.value);
  if (!protocolValidation.valid) {
    throw new Error(`Frozen protocol is invalid: ${protocolValidation.reasons.join('; ')}`);
  }
  const protocol = protocolRecord.value;
  if (path.resolve(browserPath).toLowerCase()
      !== path.resolve(protocol.browser.requiredExecutablePath).toLowerCase()) {
    throw new Error('Browser path differs from the frozen protocol identity.');
  }

  const [filesStart, gitStart, executableStart] = await Promise.all([
    collectBoundSourceFiles(),
    collectGitIdentity(),
    executableIdentity(browserPath),
  ]);
  if (!gitStart.clean) throw new Error('The public research repository must be clean.');
  if (!exactExecutableIdentity(executableStart, protocol)) {
    throw new Error('Browser executable does not match the frozen byte identity.');
  }

  // Claim the one-shot execution ID before the shared generated overlay can be touched.
  const artifactPaths = await reserveArtifactDirectory(protocol);
  const build = await buildThreeDrawIndexOverlay();
  const overlayManifestPath = path.join(build.outputRoot, overlayManifestFilename);
  const overlayManifestBytes = await readFile(overlayManifestPath);
  const overlayManifest = JSON.parse(overlayManifestBytes.toString('utf8'));
  const overlayValidation = validateThreeDrawIndexOverlayManifest(overlayManifest);
  if (!overlayValidation.valid) {
    throw new Error(`Overlay manifest is invalid: ${overlayValidation.reasons.join('; ')}`);
  }
  const overlayIdentityStart = await collectOverlayIdentity(build.outputRoot, overlayManifest);
  if (!overlayIdentityStart.valid) {
    throw new Error(
      `Generated overlay preflight failed: ${overlayIdentityStart.reasons.join('; ')}`,
    );
  }

  const report = {
    schemaVersion: 1,
    kind: THREE_DRAW_INDEX_REPORT_KIND,
    status: 'failed',
    executionMode: EXPECTED_EXECUTION_MODE,
    timingCaptured: false,
    benchmarkClaim: false,
    claimBoundary: protocol.claimBoundary,
    executionPolicy: {
      attemptCount: 1,
      retryCount: 0,
      browserLaunchCount: 1,
      deviceCountLimit: 1,
      customBrowserArguments: [],
      timingCaptured: false,
    },
    observedLifecycle: {
      serverStartAttempts: 0,
      serverInstancesCreated: 0,
      serverStartSuccesses: 0,
      browserLaunchAttempts: 0,
      browserLaunchSuccesses: 0,
      contextsCreated: 0,
      pagesCreated: 0,
      postflightChecksAttempted: 0,
      postflightChecksSucceeded: 0,
    },
    artifactPolicy: {
      reportPath: protocol.output.reportPath,
      sha256Path: protocol.output.sha256Path,
      exclusiveDirectory: true,
      overwrite: false,
    },
    protocol: {
      path: 'protocols/three-draw-index-canary-v1.json',
      byteLength: protocolRecord.bytes.length,
      sha256: protocolRecord.sha256,
      value: protocol,
    },
    sourceIdentity: {
      filesStart,
      filesEnd: null,
      filesMatch: false,
      gitStart,
      gitEnd: null,
      gitMatch: false,
    },
    overlay: {
      root: path.relative(repoRoot, build.outputRoot).replaceAll('\\', '/'),
      manifestPath: path.relative(repoRoot, overlayManifestPath).replaceAll('\\', '/'),
      manifestSha256: sha256(overlayManifestBytes),
      manifest: overlayManifest,
      validation: overlayValidation,
      identityStart: overlayIdentityStart,
      identityEnd: null,
      identityMatch: false,
    },
    browser: {
      launchApi: 'playwright-core.chromium.launch',
      headless: true,
      customArguments: [],
      executableStart,
      executableEnd: null,
      identityMatchesStart: false,
      productVersion: null,
      effectiveArguments: null,
      forbiddenEffectiveArguments: null,
      contextClosedCleanly: false,
      closedCleanly: false,
    },
    server: {
      implementation: 'vite.createServer',
      configFile: false,
      host: '127.0.0.1',
      port: null,
      localOnly: true,
      isolationHeaders: ISOLATION_HEADERS,
      closedCleanly: false,
    },
    browserObservations: null,
    browserObservationsClean: false,
    servedSource: null,
    pageResult: null,
    pageValidation: null,
    runnerFailures: [],
    failure: null,
  };

  let server = null;
  let browser = null;
  let context = null;
  let primaryFailure = null;
  let executionStage = 'server-create';
  let pageObservations = null;
  const recordRunnerFailure = (stage, error) => {
    const serialized = serializeError(error);
    report.runnerFailures.push({ stage, failure: serialized });
    if (primaryFailure === null) primaryFailure = error;
  };
  const moduleObservations = new Map();
  try {
    const aliasTargets = {
      three: path.join(build.outputRoot, 'src', 'Three.js'),
      'three/webgpu': path.join(build.outputRoot, 'src', 'Three.WebGPU.js'),
      'three/tsl': path.join(build.outputRoot, 'src', 'Three.TSL.js'),
    };
    executionStage = 'server-create';
    report.observedLifecycle.serverStartAttempts += 1;
    server = await createServer({
      root: repoRoot,
      configFile: false,
      appType: 'spa',
      logLevel: 'error',
      clearScreen: false,
      cacheDir: path.join(
        repoRoot,
        '.generated',
        'three-draw-index-vite-cache',
        protocol.executionId,
      ),
      resolve: {
        alias: Object.entries(aliasTargets).map(([find, replacement]) => ({
          find: new RegExp(`^${find.replace('/', '\\/')}$`, 'u'),
          replacement,
        })),
      },
      optimizeDeps: { noDiscovery: true, include: [] },
      plugins: [sourceObserverPlugin(
        build.outputRoot,
        overlayManifest,
        moduleObservations,
      )],
      server: {
        host: '127.0.0.1',
        port: 0,
        strictPort: false,
        hmr: false,
        headers: ISOLATION_HEADERS,
        fs: { strict: true, allow: [repoRoot, build.outputRoot] },
      },
    });
    report.observedLifecycle.serverInstancesCreated += 1;
    executionStage = 'server-listen';
    await server.listen();
    report.observedLifecycle.serverStartSuccesses += 1;
    executionStage = 'server-address';
    const address = server.httpServer.address();
    if (address === null || typeof address === 'string') {
      throw new Error('Vite did not expose a numeric loopback port.');
    }
    report.server.port = address.port;
    const baseUrl = `http://127.0.0.1:${address.port}`;

    executionStage = 'browser-launch';
    report.observedLifecycle.browserLaunchAttempts += 1;
    browser = await chromium.launch({
      executablePath: path.resolve(browserPath),
      headless: true,
      args: [],
    });
    report.observedLifecycle.browserLaunchSuccesses += 1;
    executionStage = 'browser-version';
    report.browser.productVersion = browser.version();
    executionStage = 'browser-command-line';
    report.browser.effectiveArguments = await captureEffectiveBrowserArguments(browser);
    report.browser.forbiddenEffectiveArguments = findForbiddenThreeDrawIndexBrowserArguments(
      report.browser.effectiveArguments.arguments,
      protocol,
    );
    executionStage = 'browser-policy';
    if (report.browser.productVersion !== protocol.browser.requiredProductVersion) {
      throw new Error('Launched Chrome product version differs from the frozen protocol.');
    }
    if (!report.browser.effectiveArguments.available
        || report.browser.forbiddenEffectiveArguments.length !== 0) {
      throw new Error('Chrome effective arguments violate default WebGPU exposure.');
    }

    executionStage = 'context-create';
    context = await browser.newContext({
      viewport: { width: 320, height: 180 },
      deviceScaleFactor: 1,
      serviceWorkers: 'block',
    });
    report.observedLifecycle.contextsCreated += 1;
    executionStage = 'route-install';
    await context.route('**/*', async (route) => {
      const requestUrl = new URL(route.request().url());
      if (requestUrl.origin === baseUrl) {
        await route.continue();
      } else {
        await route.abort('blockedbyclient');
      }
    });
    executionStage = 'page-create';
    const page = await context.newPage();
    report.observedLifecycle.pagesCreated += 1;
    executionStage = 'page-observers';
    const observations = installPageObservers(page);
    pageObservations = observations;
    report.browserObservations = observations;
    executionStage = 'init-script';
    await page.addInitScript(() => {
      globalThis.__THREE_DRAW_INDEX_RUNNER_DIAGNOSTICS__ = {
        unhandledRejections: [],
      };
      globalThis.addEventListener('unhandledrejection', (event) => {
        globalThis.__THREE_DRAW_INDEX_RUNNER_DIAGNOSTICS__.unhandledRejections.push({
          message: String(event.reason?.message ?? event.reason).slice(0, 4_096),
        });
      });
    });
    executionStage = 'navigation';
    await page.goto(`${baseUrl}/three-draw-index-canary.html`, {
      waitUntil: 'load',
      timeout: protocol.page.timeoutMilliseconds,
    });
    executionStage = 'page-ready';
    await page.waitForFunction(
      () => globalThis.__THREE_DRAW_INDEX_CANARY__?.ready === true,
      null,
      { timeout: protocol.page.timeoutMilliseconds },
    );
    executionStage = 'page-result';
    report.pageResult = await page.evaluate(() => (
      structuredClone(globalThis.__THREE_DRAW_INDEX_CANARY__.result)
    ));
    executionStage = 'page-diagnostics';
    const diagnostics = await page.evaluate(() => (
      structuredClone(globalThis.__THREE_DRAW_INDEX_RUNNER_DIAGNOSTICS__)
    ));
    observations.unhandledRejections.push(...diagnostics.unhandledRejections);
    executionStage = 'page-resources';
    const resourceUrls = await page.evaluate(() => (
      performance.getEntriesByType('resource').map((entry) => entry.name).sort()
    ));
    executionStage = 'page-validation';
    report.pageValidation = validateThreeDrawIndexPageResult(report.pageResult);
    executionStage = 'served-source-validation';
    report.servedSource = validateServedSource({
      observations: moduleObservations,
      manifest: overlayManifest,
      resources: resourceUrls,
      sourceFiles: filesStart,
      expectedOrigin: baseUrl,
    });
    executionStage = 'browser-observations';
    report.browserObservationsClean = pageObservationsAreClean(observations);
    executionStage = 'page-validation';
    if (!report.pageValidation.valid) {
      throw new Error(`Page evidence failed: ${report.pageValidation.reasons.join('; ')}`);
    }
    executionStage = 'served-source-validation';
    if (!report.servedSource.exact) {
      throw new Error(`Served source failed: ${report.servedSource.reasons.join('; ')}`);
    }
    executionStage = 'browser-observations';
    if (!report.browserObservationsClean) {
      throw new Error('Browser observations contain an error.');
    }
  } catch (error) {
    recordRunnerFailure(executionStage, error);
  } finally {
    try {
      await context?.close();
      report.browser.contextClosedCleanly = context !== null;
    } catch (error) {
      recordRunnerFailure('context-close', error);
    }
    try {
      await browser?.close();
      report.browser.closedCleanly = browser !== null;
    } catch (error) {
      recordRunnerFailure('browser-close', error);
    }
    try {
      await server?.close();
      report.server.closedCleanly = server !== null;
    } catch (error) {
      recordRunnerFailure('server-close', error);
    }
    if (pageObservations !== null) {
      report.browserObservationsClean = pageObservationsAreClean(pageObservations);
    }
    const postflightStages = [
      ['bound-sources', collectBoundSourceFiles()],
      ['git', collectGitIdentity()],
      ['browser-executable', executableIdentity(browserPath)],
      ['overlay', collectOverlayIdentity(build.outputRoot, overlayManifest)],
    ];
    report.observedLifecycle.postflightChecksAttempted = postflightStages.length;
    const postflightResults = await Promise.allSettled(
      postflightStages.map(([, promise]) => promise),
    );
    for (let index = 0; index < postflightResults.length; index += 1) {
      const [stage] = postflightStages[index];
      const result = postflightResults[index];
      if (result.status === 'rejected') {
        recordRunnerFailure(`postflight-${stage}`, result.reason);
        continue;
      }
      report.observedLifecycle.postflightChecksSucceeded += 1;
      if (stage === 'bound-sources') report.sourceIdentity.filesEnd = result.value;
      else if (stage === 'git') report.sourceIdentity.gitEnd = result.value;
      else if (stage === 'browser-executable') report.browser.executableEnd = result.value;
      else if (stage === 'overlay') report.overlay.identityEnd = result.value;
    }
    report.sourceIdentity.filesMatch = report.sourceIdentity.filesEnd !== null
      && boundSourcesMatch(filesStart, report.sourceIdentity.filesEnd);
    report.sourceIdentity.gitMatch = report.sourceIdentity.gitEnd !== null
      && exactGitIdentity(gitStart, report.sourceIdentity.gitEnd);
    report.browser.identityMatchesStart = report.browser.executableEnd !== null
      && JSON.stringify(executableStart) === JSON.stringify(report.browser.executableEnd)
      && exactExecutableIdentity(report.browser.executableEnd, protocol);
    report.overlay.identityMatch = report.overlay.identityEnd !== null
      && JSON.stringify(overlayIdentityStart) === JSON.stringify(report.overlay.identityEnd);
    if (postflightResults.every((result) => result.status === 'fulfilled')
        && (!report.sourceIdentity.filesMatch
          || !report.sourceIdentity.gitMatch
          || !report.browser.identityMatchesStart
          || !report.overlay.identityMatch)) {
      recordRunnerFailure(
        'postflight-identity-mismatch',
        new Error('Postflight source, Git, browser, or overlay identity changed.'),
      );
    }
  }

  if (primaryFailure === null) {
    report.status = report.pageResult.status === 'unsupported' ? 'unsupported' : 'passed';
  } else {
    report.status = 'failed';
    report.failure = report.runnerFailures[0].failure;
  }
  let reportValidation = validateThreeDrawIndexReport(report);
  if (!reportValidation.valid && report.status !== 'failed') {
    report.status = 'failed';
    const validationFailure = serializeError(new Error(
      `Final report validation failed: ${reportValidation.reasons.join('; ')}`,
    ));
    report.runnerFailures.push({
      stage: 'final-report-validation',
      failure: validationFailure,
    });
    report.failure = validationFailure;
    reportValidation = validateThreeDrawIndexReport(report);
  }
  const artifact = await writeImmutableArtifact(artifactPaths, report);
  const verification = await verifyThreeDrawIndexArtifact(
    artifactPaths.reportPath,
    protocolRecord,
  );
  if (!verification.valid) {
    throw new Error(`Written artifact verification failed: ${verification.reasons.join('; ')}`);
  }
  return { report, artifact, verification, reportValidation };
}

function isDirectExecution() {
  return process.argv[1] !== undefined
    && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
}

if (isDirectExecution()) {
  let options;
  try {
    options = parseThreeDrawIndexArguments(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 1;
  }
  if (options !== undefined) {
    runThreeDrawIndexProbe(options).then(({ report, artifact, verification }) => {
      process.stdout.write(`${JSON.stringify({
        status: report.status,
        pageStatus: report.pageResult?.status ?? null,
        reportPath: artifact.reportPath,
        sidecarPath: artifact.sidecarPath,
        byteLength: artifact.byteLength,
        sha256: artifact.sha256,
        verified: verification.valid,
        failure: report.failure,
      }, null, 2)}\n`);
      if (report.status === 'unsupported') process.exitCode = 2;
      if (report.status === 'failed') process.exitCode = 1;
    }).catch((error) => {
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    });
  }
}

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { brotliDecompressSync } from 'node:zlib';

import { summarizeCsv, verifyRunDirectory } from '../analysis/summarize.mjs';
import {
  PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST,
  PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST,
  sanitizeLiveEvidence,
  serializeLiveEvidenceSanitizerResult,
} from '../scripts/sanitize-live-evidence.mjs';
import {
  compareComputeProcessIdentitySets,
  createNvidiaTelemetryCoverageAudit,
  summarizeTelemetryRows,
  TELEMETRY_CSV_FIELDS,
} from '../scripts/nvidia-telemetry.mjs';
import {
  liveFirstInstanceValidationSemanticSha256,
  liveFirstInstanceCrossoverScheduleSha256,
  liveFirstInstanceRenderParityIdentity,
  validateLiveFirstInstanceCrossoverRenderParity,
  validateLiveFirstInstanceCrossoverValidation,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  firstInstanceLiveCrossoverFrame,
  firstInstanceLiveCrossoverHistoryCounts,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  createLiveFirstInstanceEnvironmentAudit,
} from '../scripts/live-first-instance-environment-audit.mjs';
import {
  LIVE_SANITIZER_IMPLEMENTATION_PATHS,
} from '../scripts/live-evidence-sanitizer-policy.mjs';
import { collectSourceProvenance } from '../scripts/source-provenance.mjs';
import {
  CANDIDATE_VITE_REQUIRED_MODULE_PATHS,
  CANDIDATE_VITE_RUNTIME_POLICY_ID,
  candidateViteRuntimeModulesSha256,
} from '../scripts/execution-dependency-closure.mjs';

const TEST_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(TEST_DIRECTORY, '..');
const STRICT_FIXTURE_PATH = path.join(
  TEST_DIRECTORY,
  'fixtures',
  'strict-live-run.br.b64',
);
const execFileAsync = promisify(execFile);
const REQUIRED_ARTIFACTS = Object.freeze([
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
  'forced-feature-off-evidence.json',
]);

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function csvCell(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const FIXTURE_RENDERER_MEMORY = Object.freeze({
  attributes: 0,
  attributesSize: 0,
  geometries: 0,
  indexAttributes: 0,
  indexAttributesSize: 0,
  indirectStorageAttributes: 0,
  indirectStorageAttributesSize: 0,
  programs: 0,
  programsSize: 0,
  storageAttributes: 0,
  storageAttributesSize: 0,
  uniformBuffers: 0,
  uniformBuffersSize: 0,
});

function fixtureViewportState(textureUuid) {
  const viewport = { x: 0, y: 0, width: 1280, height: 720, minDepth: 0, maxDepth: 1 };
  const scissor = { x: 0, y: 0, width: 1280, height: 720 };
  return {
    schemaVersion: 1,
    kind: 'three-r185-live-first-instance-viewport-state',
    renderer: {
      viewport: structuredClone(viewport),
      scissor: structuredClone(scissor),
      scissorTest: false,
      activeRenderTargetTextureUuid: null,
    },
    renderTarget: {
      textureUuid,
      width: 1280,
      height: 720,
      viewport: structuredClone(viewport),
      scissor: structuredClone(scissor),
      scissorTest: false,
    },
  };
}

function addCommandAttributeVersions(commitments) {
  for (const lane of ['portable', 'feature']) commitments[lane].attributeVersion = 0;
}

function staticFixtureLifecycle(lifecycle) {
  const value = structuredClone(lifecycle);
  for (const field of [
    'activeLane',
    'activeCommandBufferId',
    'residentPreparedLane',
    'laneSelectionSerial',
    'computeCallSerial',
    'prepareSerial',
  ]) delete value[field];
  return value;
}

function addFixtureShaderObservation(shaderEvidence, lifecycle, request, executionCounters) {
  const semanticEvidence = structuredClone(shaderEvidence);
  delete semanticEvidence.observation;
  const resourceIdentitiesAtStart = {
    render: {
      portable: {
        renderObjectIdentity: 1,
        nodeBuilderStateIdentity: 2,
        bindingArrayIdentity: 3,
        pipelineIdentity: 4,
      },
      feature: {
        renderObjectIdentity: 5,
        nodeBuilderStateIdentity: 6,
        bindingArrayIdentity: 7,
        pipelineIdentity: 8,
      },
    },
    compute: {
      portable: {
        reset: {
          computeNodeId: lifecycle.computeNodeIds.portable[0],
          computeNodeVersion: 0,
          nodeBuilderStateIdentity: 9,
          bindingArrayIdentity: 10,
          pipelineIdentity: 11,
        },
        cull: {
          computeNodeId: lifecycle.computeNodeIds.portable[1],
          computeNodeVersion: 0,
          nodeBuilderStateIdentity: 12,
          bindingArrayIdentity: 13,
          pipelineIdentity: 14,
        },
      },
      feature: {
        reset: {
          computeNodeId: lifecycle.computeNodeIds.feature[0],
          computeNodeVersion: 0,
          nodeBuilderStateIdentity: 15,
          bindingArrayIdentity: 16,
          pipelineIdentity: 17,
        },
        cull: {
          computeNodeId: lifecycle.computeNodeIds.feature[1],
          computeNodeVersion: 0,
          nodeBuilderStateIdentity: 18,
          bindingArrayIdentity: 19,
          pipelineIdentity: 20,
        },
      },
    },
  };
  const inspectionStateBefore = {
    totalPipelineCacheEntries: 11,
    programEntries: { vertex: 2, fragment: 2, compute: 4 },
    memory: structuredClone(FIXTURE_RENDERER_MEMORY),
  };
  const observationCore = {
    schemaVersion: 1,
    kind: 'live-first-instance-fresh-shader-runtime-observation',
    serial: request.captureOrdinal,
    captureMode: 'live-r185-cache-inspection',
    request: structuredClone(request),
    executionCountersAtStart: structuredClone(executionCounters),
    executionCountersAtEnd: structuredClone(executionCounters),
    executionCountersStable: true,
    resourcesPreinitialized: true,
    resourceIdentitiesAtStart,
    resourceIdentitiesAtEnd: structuredClone(resourceIdentitiesAtStart),
    resourceIdentitiesStable: true,
    inspectionStateBefore,
    inspectionStateAfter: structuredClone(inspectionStateBefore),
    inspectionStateStable: true,
    semanticSha256: sha256(JSON.stringify(semanticEvidence)),
  };
  shaderEvidence.observation = {
    ...observationCore,
    observationSha256: sha256(JSON.stringify(observationCore)),
  };
}

function addFixtureComputeTimestampRegistrations(validation, planIndex) {
  const contextBase = planIndex * 2 + 1;
  validation.lifecycle.computeTimestampContextIds = {
    portable: contextBase,
    feature: contextBase + 1,
  };
  validation.lifecycle.computeTimestampRegistrations = Object.fromEntries(
    ['portable', 'feature'].map((lane, index) => [lane, {
      schemaVersion: 1,
      kind: 'live-first-instance-compute-timestamp-group-registration',
      laneId: lane,
      timestampContextId: contextBase + index,
      registrationSerial: contextBase + index,
      backendIdentity: 51,
      backendWrapperIdentity: 52,
      computeGroupIdentity: 1_001 + planIndex * 2 + index,
      computeNodeIds: [...validation.lifecycle.computeNodeIds[lane]],
    }]),
  );
}

function fixtureProductionBundleOutput(lane, parityLane, validation, targetTextureUuid, serialBase) {
  const lifecycle = validation.lifecycle;
  const commitment = {
    schemaVersion: 1,
    kind: 'live-first-instance-timed-bundle-commitment',
    laneId: lane,
    bundleUuid: lifecycle.bundleUuids[lane],
    bundleVersion: lifecycle.bundleVersions[lane],
    meshUuid: lifecycle.meshUuids[lane],
    geometryUuid: lifecycle.geometryUuids[lane],
    materialUuid: lifecycle.materialUuids[lane],
    materialVersion: lifecycle.materialVersions[lane],
    bundleRecordCallbackCount: 1,
    renderResourceIdentities: structuredClone(
      validation.shaderEvidence.observation.resourceIdentitiesAtStart.render[lane],
    ),
    inspectionState: structuredClone(
      validation.shaderEvidence.observation.inspectionStateBefore,
    ),
  };
  const executionBefore = {
    laneSelectionSerial: validation.lifecycle.laneSelectionSerial,
    strategyComputeCallSerial: validation.lifecycle.computeCallSerial,
    strategyPrepareSerial: validation.lifecycle.prepareSerial,
    rendererComputeCallSerial: serialBase + 10,
    rendererRenderCallSerial: serialBase + 20,
  };
  const executionBetween = {
    ...executionBefore,
    rendererRenderCallSerial: executionBefore.rendererRenderCallSerial + 1,
  };
  const executionAfter = {
    ...executionBefore,
    rendererRenderCallSerial: executionBefore.rendererRenderCallSerial + 2,
  };
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-production-bundle-output',
    pass: true,
    laneId: lane,
    target: {
      textureUuid: targetTextureUuid,
      width: 1280,
      height: 720,
      samples: 0,
      depthBuffer: true,
    },
    captures: 2,
    color: structuredClone(parityLane.color),
    directDiagnosticColor: structuredClone(parityLane.color),
    directDiagnosticExact: true,
    resourcesStable: true,
    bundleRecordedExactlyOnce: true,
    executionExact: true,
    executionBefore,
    executionBetween,
    executionAfter,
    commitmentBefore: structuredClone(commitment),
    commitmentBetween: structuredClone(commitment),
    commitmentAfter: structuredClone(commitment),
    stability: {
      pass: true,
      firstCapture: structuredClone(parityLane.color),
      secondCapture: structuredClone(parityLane.color),
    },
  };
}

function rehashFixtureShaderEvidence(shaderEvidence) {
  const semanticEvidence = structuredClone(shaderEvidence);
  delete semanticEvidence.observation;
  shaderEvidence.observation.semanticSha256 = sha256(JSON.stringify(semanticEvidence));
  const observationCore = structuredClone(shaderEvidence.observation);
  delete observationCore.observationSha256;
  shaderEvidence.observation.observationSha256 = sha256(JSON.stringify(observationCore));
}

function setFixtureValidationCounters(validation, counters) {
  validation.prepareSerialStart = counters.prepareSerial;
  validation.prepareSerialEnd = counters.prepareSerial + 2;
  validation.computeCallSerialStart = counters.computeCallSerial;
  validation.computeCallSerialEnd = counters.computeCallSerial + 2;
  for (const [index, lane] of ['portable', 'feature'].entries()) {
    validation.lanes[lane].prepareSerial = counters.prepareSerial + index + 1;
    validation.lanes[lane].computeCallSerial = counters.computeCallSerial + index + 1;
    validation.lanes[lane].laneSelectionSerial = counters.laneSelectionSerial + index + 1;
  }
  validation.lifecycle.prepareSerial = counters.prepareSerial + 2;
  validation.lifecycle.computeCallSerial = counters.computeCallSerial + 2;
  validation.lifecycle.laneSelectionSerial = counters.laneSelectionSerial + 2;
}

function upgradeFixtureValidationCapture(
  capture,
  parityRequest,
  validationRequest,
  parityCounters,
  validationCounters,
  planIndex,
  targetTextureUuid,
) {
  const validation = capture.validation.payload;
  addFixtureComputeTimestampRegistrations(validation, planIndex);
  addCommandAttributeVersions(validation.commandBufferCommitments);
  addCommandAttributeVersions(validation.lifecycle.commandBufferCommitments);
  for (const lane of ['portable', 'feature']) {
    validation.lanes[lane].commandBuffer.attributeVersion = 0;
    validation.hooks[lane].addressChallenge.commandBuffer.attributeVersion = 0;
    validation.addressChallenges.lanes[lane].commandBuffer.attributeVersion = 0;
  }
  validation.lifecycle.indirectAttributeVersions = {
    portable: validation.commandBufferCommitments.portable.attributeVersion,
    feature: validation.commandBufferCommitments.feature.attributeVersion,
  };
  const parityValidation = structuredClone(validation);
  setFixtureValidationCounters(parityValidation, parityCounters);
  addFixtureShaderObservation(
    parityValidation.shaderEvidence,
    parityValidation.lifecycle,
    parityRequest,
    parityCounters,
  );
  setFixtureValidationCounters(validation, validationCounters);
  addFixtureShaderObservation(
    validation.shaderEvidence,
    validation.lifecycle,
    validationRequest,
    validationCounters,
  );
  capture.validation.payloadSha256 = sha256(JSON.stringify(validation));
  capture.validation.semanticSha256 = liveFirstInstanceValidationSemanticSha256(validation);
  capture.renderParity.snapshotValidation = parityValidation;
  capture.renderParity.crossLaneProductionExact = true;
  for (const [index, lane] of ['portable', 'feature'].entries()) {
    capture.renderParity.lanes[lane].productionBundleOutput = fixtureProductionBundleOutput(
      lane,
      capture.renderParity.lanes[lane],
      parityValidation,
      targetTextureUuid,
      planIndex * 100 + parityRequest.captureOrdinal * 10 + index,
    );
  }
  const paritySha256 = liveFirstInstanceRenderParityIdentity(capture.renderParity);
  capture.renderParitySemanticSha256 = paritySha256;
  capture.renderParityOutputSha256 = paritySha256;
}

async function upgradeStrictFixtureSchema(privateDirectory) {
  const artifactsPath = path.join(privateDirectory, 'validation-artifacts.json');
  const summariesPath = path.join(privateDirectory, 'trial-summaries.json');
  const framesPath = path.join(privateDirectory, 'frames.csv');
  const artifacts = JSON.parse(await readFile(artifactsPath));
  const summaries = JSON.parse(await readFile(summariesPath));
  const summaryByTrial = new Map(summaries.map((summary) => [summary.trialId, summary]));
  const validationByTrial = new Map();

  for (const artifact of artifacts) {
    const summary = summaryByTrial.get(artifact.trialId);
    const phaseRoles = [
      ['preflight', 'render-parity'],
      ['preflight', 'main-validation'],
      ['timing-start', 'render-parity'],
      ['timing-start', 'main-validation'],
      ['postflight', 'render-parity'],
      ['postflight', 'main-validation'],
    ];
    artifact.shaderObservationChallenges = phaseRoles.map(([phase, role], index) => ({
      schemaVersion: 1,
      kind: 'live-first-instance-shader-observation-challenge',
      origin: 'node-runner',
      runId: artifact.trialId.slice(0, -4),
      trialId: artifact.trialId,
      planIndex: artifact.planIndex,
      repetitionIndex: artifact.repetitionIndex,
      phase,
      role,
      captureOrdinal: index + 1,
      challengeNonce: sha256(`${artifact.trialId}|shader-observation|${index + 1}`),
    }));
    const counterStarts = [
      [7, 3, 0],
      [10, 5, 2],
      [12, 7, 4],
      [15, 9, 6],
      [817, 811, 8],
      [820, 813, 10],
    ].map(([laneSelectionSerial, computeCallSerial, prepareSerial]) => ({
      laneSelectionSerial,
      computeCallSerial,
      prepareSerial,
    }));
    const captures = [artifact.pre, artifact.timingStart, artifact.post];
    captures.forEach((capture, index) => upgradeFixtureValidationCapture(
      capture,
      artifact.shaderObservationChallenges[index * 2],
      artifact.shaderObservationChallenges[index * 2 + 1],
      counterStarts[index * 2],
      counterStarts[index * 2 + 1],
      artifact.planIndex,
      summary.completionInvariant.renderTargetTextureUuidAtTimingStart,
    ));
    const semanticSha256 = artifact.timingStart.validation.semanticSha256;
    assert.equal(artifact.pre.validation.semanticSha256, semanticSha256);
    assert.equal(artifact.post.validation.semanticSha256, semanticSha256);
    artifact.liveFirstInstanceTrialEvidence.semanticSha256 = semanticSha256;
    const history = firstInstanceLiveCrossoverHistoryCounts(480, 0);
    artifact.liveFirstInstanceTrialEvidence.historyBalance = {
      schemaVersion: 1,
      kind: 'live-first-instance-crossover-history-balance',
      frameCount: 480,
      expectedTransitionCountPerCell: 120,
      transitionCounts: { ...history.transitionCounts },
      expectedHistoryTripleCountPerCell: 60,
      historyTripleCounts: { ...history.historyTripleCounts },
    };
    artifact.plannedScheduleSha256 = liveFirstInstanceCrossoverScheduleSha256(
      artifact.superblockOrientationOffset,
    );
    summary.plannedScheduleSha256 = artifact.plannedScheduleSha256;

    const invariant = summary.completionInvariant;
    invariant.commandBuffersExact = true;
    addCommandAttributeVersions(invariant.commandBufferCommitments);
    const staticLifecycle = staticFixtureLifecycle(artifact.timingStart.validation.payload.lifecycle);
    invariant.staticLifecycleAtTimingStart = structuredClone(staticLifecycle);
    invariant.staticLifecycleAtTimingEnd = structuredClone(staticLifecycle);
    invariant.lifecycleCommitmentAtTimingStart = fnv1a64Text(JSON.stringify(staticLifecycle));
    invariant.lifecycleCommitmentAtTimingEnd = invariant.lifecycleCommitmentAtTimingStart;
    invariant.computeProgramEntriesExact = true;
    invariant.computeProgramEntriesAtTimingStart = 4;
    invariant.computeProgramEntriesAtTimingEnd = 4;
    invariant.rendererMemoryExact = true;
    invariant.rendererMemoryAtTimingStart = structuredClone(FIXTURE_RENDERER_MEMORY);
    invariant.rendererMemoryAtTimingEnd = structuredClone(FIXTURE_RENDERER_MEMORY);
    invariant.viewportStateExact = true;
    const viewport = fixtureViewportState(invariant.renderTargetTextureUuidAtTimingStart);
    invariant.viewportStateAtTimingStart = structuredClone(viewport);
    invariant.viewportStateAtTimingEnd = structuredClone(viewport);
    summary.validation.liveFirstInstanceSemanticSha256 = semanticSha256;

    delete artifact.sha256;
    artifact.sha256 = sha256(JSON.stringify(artifact));
    summary.validation.artifactSha256 = artifact.sha256;
    validationByTrial.set(artifact.trialId, artifact.timingStart.validation.payload);
  }

  const csvText = await readFile(framesPath, 'utf8');
  const lines = csvText.trimEnd().split(/\r?\n/);
  const originalHeaders = parseCsvHeader(lines[0]);
  const headers = [...new Set([
    ...originalHeaders,
    'computeProgramEntriesAtTimingStart',
    'computeFrameCallIndex',
    'computeGroupIdentity',
    'computeTimestampBackendIdentity',
    'computeTimestampBackendWrapperIdentity',
    'computeTimestampContextId',
    'computeTimestampRegistrationSerial',
    'gpuComputeTimestampDurationValid',
    'gpuComputeTimestampRecords',
    'gpuComputeTimestampUids',
    'gpuRenderTimestampDurationValid',
    'gpuRenderTimestampRecords',
    'gpuRenderTimestampUids',
    'previousLaneId',
    'previousPreviousLaneId',
    'renderFrameCallIndex',
    'rendererMemoryAtTimingStart',
    'strictTimestampUidAttribution',
    'submittedComputeLaneId',
    'submittedComputeNodeIds',
    'viewportStateAtTimingStart',
  ])].sort();
  const rowRecords = lines.slice(1).map((line) => {
    const values = parseCsvHeader(line);
    return Object.fromEntries(originalHeaders.map(
      (header, index) => [header, values[index]],
    ));
  });
  const timingFields = [
    'cpuCommonUpdateMs',
    'cpuComputeSubmitMs',
    'cpuRenderSubmitMs',
    'cpuSubmitTotalMs',
    'cpuFrameBodyMs',
    'gpuComputeMs',
    'gpuRenderMs',
    'gpuPassTotalMs',
  ];
  for (let start = 0; start < rowRecords.length; start += 8) {
    const block = rowRecords.slice(start, start + 8);
    const donors = {
      portable: block.filter((row) => row.laneId === 'portable').map((row) => (
        Object.fromEntries(timingFields.map((field) => [field, row[field]]))
      )),
      feature: block.filter((row) => row.laneId === 'feature').map((row) => (
        Object.fromEntries(timingFields.map((field) => [field, row[field]]))
      )),
    };
    for (const row of block) {
      const scheduled = firstInstanceLiveCrossoverFrame(
        Number(row.phaseFrameIndex),
        Number(row.superblockOrientationOffset),
      );
      const donor = donors[scheduled.laneId].shift();
      for (const field of timingFields) row[field] = donor[field];
      row.crossoverBlockIndex = scheduled.crossoverBlockIndex;
      row.withinBlockPosition = scheduled.withinBlockPosition;
      row.crossoverPattern = scheduled.pattern;
      row.crossoverPatternIndex = scheduled.patternIndex;
      row.previousPreviousLaneId = scheduled.previousPreviousLaneId;
      row.previousLaneId = scheduled.previousLaneId;
      row.laneId = scheduled.laneId;
      row.plannedScheduleSha256 = liveFirstInstanceCrossoverScheduleSha256(
        Number(row.superblockOrientationOffset),
      );
    }
  }
  const timestampRecord = (type, row, registration) => {
    const frameId = Number(row.gpuFrameId);
    const callIndex = 1;
    const contextId = type === 'compute' ? registration.timestampContextId : 601;
    const durationMs = Number(type === 'compute' ? row.gpuComputeMs : row.gpuRenderMs);
    const prefix = type === 'compute' ? 'c' : 'r';
    return {
      uid: `${prefix}:${callIndex}:${contextId}:f${frameId}`,
      type,
      callIndex,
      contextId,
      frameId,
      durationMs,
    };
  };
  const quantumNs = (records) => {
    const gcd = (left, right) => {
      let a = Math.abs(left);
      let b = Math.abs(right);
      while (b !== 0) [a, b] = [b, a % b];
      return a;
    };
    return records.map((record) => Math.round(record.durationMs * 1e6)).reduce(gcd);
  };
  const phaseEvidence = (phase, events) => ({
    schemaVersion: 1,
    kind: 'three-r185-timestamp-phase-result',
    phase,
    includedTypes: ['render', 'compute'],
    strictUidGrammar: true,
    pools: Object.fromEntries(['render', 'compute'].map((type) => {
      const records = events.flatMap((event) => (
        type === 'compute'
          ? event.gpuComputeTimestampRecords
          : event.gpuRenderTimestampRecords
      ));
      return [type, {
        type,
        included: true,
        frames: events.map((event) => event.gpuFrameId),
        uidRecords: records,
        resolution: {
          quantumNs: quantumNs(records),
          classification: 'fine',
          recordCount: records.length,
          positiveDurationCount: records.length,
          nonpositiveDurationCount: 0,
        },
      }];
    })),
  });
  for (const [trialId, summary] of summaryByTrial) {
    const trialRows = rowRecords.filter((row) => row.trialId === trialId);
    const validation = validationByTrial.get(trialId);
    const laneOrdinals = { portable: 0, feature: 0 };
    for (const row of trialRows) {
      const laneOrdinal = laneOrdinals[row.laneId];
      laneOrdinals[row.laneId] += 1;
      const adjustment = laneOrdinal % 2 === 0 ? 0 : 0.001;
      row.gpuComputeMs = Number(row.gpuComputeMs) + adjustment;
      row.gpuRenderMs = Number(row.gpuRenderMs) + adjustment;
      row.gpuPassTotalMs = row.gpuComputeMs + row.gpuRenderMs;
      const registration = validation.lifecycle.computeTimestampRegistrations[row.laneId];
      const computeRecord = timestampRecord('compute', row, registration);
      const renderRecord = timestampRecord('render', row, registration);
      row.strictTimestampUidAttribution = true;
      row.submittedComputeLaneId = row.laneId;
      row.computeTimestampContextId = registration.timestampContextId;
      row.computeGroupIdentity = registration.computeGroupIdentity;
      row.computeTimestampRegistrationSerial = registration.registrationSerial;
      row.computeTimestampBackendIdentity = registration.backendIdentity;
      row.computeTimestampBackendWrapperIdentity = registration.backendWrapperIdentity;
      row.submittedComputeNodeIds = JSON.stringify(registration.computeNodeIds);
      row.computeFrameCallIndex = 1;
      row.renderFrameCallIndex = 1;
      row.gpuComputeTimestampUids = JSON.stringify([computeRecord.uid]);
      row.gpuComputeTimestampRecords = JSON.stringify([computeRecord]);
      row.gpuComputeTimestampDurationValid = true;
      row.gpuRenderTimestampUids = JSON.stringify([renderRecord.uid]);
      row.gpuRenderTimestampRecords = JSON.stringify([renderRecord]);
      row.gpuRenderTimestampDurationValid = true;
    }
    const invariant = summary.completionInvariant;
    const orientation = Number(trialRows[0].superblockOrientationOffset);
    const firstMeasuredGpuFrameId = Number(trialRows[0].gpuFrameId);
    const warmupEvents = Array.from({ length: 320 }, (_, frameIndex) => {
      const scheduled = firstInstanceLiveCrossoverFrame(frameIndex, orientation);
      const registration = validation.lifecycle.computeTimestampRegistrations[scheduled.laneId];
      const row = {
        gpuFrameId: firstMeasuredGpuFrameId - 320 + frameIndex,
        gpuComputeMs: frameIndex % 2 === 0 ? 0.03 : 0.031,
        gpuRenderMs: frameIndex % 2 === 0 ? 0.2 : 0.201,
      };
      const computeRecord = timestampRecord('compute', row, registration);
      const renderRecord = timestampRecord('render', row, registration);
      const ordinal = frameIndex + 1;
      return {
        schemaVersion: 1,
        kind: 'live-first-instance-warmup-frame-event',
        phase: 'warmup',
        frameIndex,
        warmupFrameIndex: frameIndex,
        phaseFrameIndex: frameIndex,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        previousPreviousLaneId: frameIndex < 2
          ? null
          : firstInstanceLiveCrossoverFrame(frameIndex - 2, orientation).laneId,
        previousLaneId: frameIndex === 0
          ? null
          : firstInstanceLiveCrossoverFrame(frameIndex - 1, orientation).laneId,
        laneId: scheduled.laneId,
        commandBufferId: validation.commandBufferCommitments[scheduled.laneId].attributeId,
        submittedComputeLaneId: scheduled.laneId,
        computeTimestampContextId: registration.timestampContextId,
        computeGroupIdentity: registration.computeGroupIdentity,
        computeTimestampRegistrationSerial: registration.registrationSerial,
        computeTimestampBackendIdentity: registration.backendIdentity,
        computeTimestampBackendWrapperIdentity: registration.backendWrapperIdentity,
        submittedComputeNodeIds: [...registration.computeNodeIds],
        gpuFrameId: row.gpuFrameId,
        selectorWriteSerial: invariant.selectorWriteSerialAtTimingStart + ordinal,
        strategySelectionSerial: invariant.strategySelectionSerialAtTimingStart + ordinal,
        strategyComputeCallSerial: invariant.strategyComputeCallSerialAtTimingStart + ordinal,
        computeCallSerial: invariant.computeCallSerialAtTimingStart + ordinal,
        computeFrameCallIndex: 1,
        renderCallSerial: invariant.renderCallSerialAtTimingStart + ordinal,
        renderFrameCallIndex: 1,
        gpuComputeMs: row.gpuComputeMs,
        gpuComputeTimestampUids: [computeRecord.uid],
        gpuComputeTimestampRecords: [computeRecord],
        gpuRenderMs: row.gpuRenderMs,
        gpuRenderTimestampUids: [renderRecord.uid],
        gpuRenderTimestampRecords: [renderRecord],
        gpuPassTotalMs: row.gpuComputeMs + row.gpuRenderMs,
      };
    });
    const lastWarmup = warmupEvents.at(-1);
    const penultimateWarmup = warmupEvents.at(-2);
    invariant.warmupScheduleAudit = {
      schemaVersion: 1,
      kind: 'live-first-instance-compact-warmup-schedule-audit',
      pass: true,
      expectedFrameCount: 320,
      actualStartupHistory: { previousPreviousLaneId: null, previousLaneId: null },
      eventsExact: true,
      timestampPhaseExact: true,
      postWarmupStateExact: true,
      postWarmupState: {
        schemaVersion: 1,
        kind: 'live-first-instance-post-warmup-state',
        previousPreviousLaneId: penultimateWarmup.laneId,
        previousLaneId: lastWarmup.laneId,
        lastWarmupGpuFrameId: lastWarmup.gpuFrameId,
        selectorWriteSerial: lastWarmup.selectorWriteSerial,
        strategySelectionSerial: lastWarmup.strategySelectionSerial,
        strategyComputeCallSerial: lastWarmup.strategyComputeCallSerial,
        computeCallSerial: lastWarmup.computeCallSerial,
        renderCallSerial: lastWarmup.renderCallSerial,
      },
      events: warmupEvents,
    };
    const measuredEvents = trialRows.map((row) => ({
      gpuFrameId: Number(row.gpuFrameId),
      gpuComputeTimestampRecords: JSON.parse(row.gpuComputeTimestampRecords),
      gpuRenderTimestampRecords: JSON.parse(row.gpuRenderTimestampRecords),
    }));
    const warmupPhase = phaseEvidence('warmup', warmupEvents);
    const measurementPhase = phaseEvidence('measurement', measuredEvents);
    Object.assign(summary.timestamps, {
      warmupRowCount: 320,
      missingWarmupComputeFrames: 0,
      missingWarmupRenderFrames: 0,
      invalidWarmupComputeTimestampUidCountFrames: 0,
      invalidWarmupRenderTimestampUidCountFrames: 0,
      invalidWarmupComputeTimestampDurationFrames: 0,
      invalidWarmupRenderTimestampDurationFrames: 0,
      invalidComputeTimestampDurationFrames: 0,
      invalidRenderTimestampDurationFrames: 0,
      renderTimestampPoolQualityValid: true,
      computeTimestampPoolQualityValid: true,
      warmupRenderTimestampPoolQualityValid: true,
      warmupComputeTimestampPoolQualityValid: true,
      warmupTimestampFrameCountValid: true,
      measurementTimestampFrameCountValid: true,
      timestampResolutions: {
        render: structuredClone(measurementPhase.pools.render.resolution),
        compute: structuredClone(measurementPhase.pools.compute.resolution),
      },
      timestampPhases: {
        schemaVersion: 1,
        kind: 'three-r185-timestamp-phase-results',
        warmup: warmupPhase,
        measurement: measurementPhase,
      },
    });
    summary.timestamps.quantumNs = Math.max(
      measurementPhase.pools.render.resolution.quantumNs,
      measurementPhase.pools.compute.resolution.quantumNs,
    );
    summary.timestamps.classification = 'fine';
    const percentile = (values, fraction) => {
      const sorted = [...values].sort((left, right) => left - right);
      return sorted[Math.min(
        sorted.length - 1,
        Math.max(0, Math.ceil(sorted.length * fraction) - 1),
      )];
    };
    for (const [prefix, field] of [
      ['gpuCompute', 'gpuComputeMs'],
      ['gpuRender', 'gpuRenderMs'],
      ['gpuPassTotal', 'gpuPassTotalMs'],
    ]) {
      const values = trialRows.map((row) => Number(row[field]));
      summary.timing[`${prefix}P50Ms`] = percentile(values, 0.5);
      summary.timing[`${prefix}P95Ms`] = percentile(values, 0.95);
    }
  }
  const changedLines = rowRecords.map((row) => {
    const summary = summaryByTrial.get(row.trialId);
    const validation = validationByTrial.get(row.trialId);
    row.commandBufferId = validation.commandBufferCommitments[row.laneId].attributeId;
    row.commandBufferCommitmentsAtTimingStart = JSON.stringify(
      validation.commandBufferCommitments,
    );
    row.lifecycleCommitmentAtTimingStart =
      summary.completionInvariant.lifecycleCommitmentAtTimingStart;
    row.computeProgramEntriesAtTimingStart =
      summary.completionInvariant.computeProgramEntriesAtTimingStart;
    row.rendererMemoryAtTimingStart = JSON.stringify(
      summary.completionInvariant.rendererMemoryAtTimingStart,
    );
    row.viewportStateAtTimingStart = JSON.stringify(
      summary.completionInvariant.viewportStateAtTimingStart,
    );
    return headers.map((header) => csvCell(row[header])).join(',');
  });
  await writeFile(framesPath, `${headers.join(',')}\n${changedLines.join('\n')}\n`);
  await writeFile(artifactsPath, jsonBytes(artifacts));
  await writeFile(summariesPath, jsonBytes(summaries));
  return artifacts.map(({ sha256: digest }) => digest);
}

function csvLine(record) {
  return TELEMETRY_CSV_FIELDS.map((field) => {
    const text = record[field] === null || record[field] === undefined
      ? ''
      : String(record[field]);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }).join(',');
}

function processRecord(gpuUuid, pid, processName, usedMemoryMiB = null) {
  return { gpuUuid, pid, processName, usedMemoryMiB };
}

function identityRecord({ gpuUuid, pid, processName }) {
  return { gpuUuid, pid, processName };
}

function telemetryFixture(runId) {
  const gpuUuid = 'GPU-private-machine-uuid';
  const processes = [
    processRecord(gpuUuid, 73, 'private-app.exe'),
    processRecord(gpuUuid, 41, 'chrome.exe'),
  ];
  const coverageAudit = createNvidiaTelemetryCoverageAudit([
    { runElapsedMs: 0, gpuIndex: 0, gpuName: 'Example GPU', gpuUuid },
    { runElapsedMs: 250, gpuIndex: 0, gpuName: 'Example GPU', gpuUuid },
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 250,
    requestedIntervalMs: 250,
  });
  const snapshot = (label, capturedAtIso, runElapsedMs, snapshotProcesses) => ({
    label,
    status: 'available',
    capturedAtIso,
    runElapsedMs,
    reason: null,
    rawNonemptyLineCount: snapshotProcesses.length,
    parsedRecordCount: snapshotProcesses.length,
    malformedLineCount: 0,
    stdoutByteCount: Buffer.byteLength(JSON.stringify(snapshotProcesses)),
    stdoutTruncated: false,
    stderrByteCount: 0,
    processes: snapshotProcesses,
  });
  return {
    provider: 'nvidia-smi',
    status: 'available',
    reason: null,
    command: 'nvidia-smi',
    sampling: {
      processModel: 'one-long-lived-process',
      requestedIntervalMs: 250,
      queryFields: ['index', 'name', 'uuid'],
      outputFile: 'gpu-telemetry.csv',
      collectorStartedRunElapsedMs: 0,
      collectorStopRequestedRunElapsedMs: 250,
      malformedLineCount: 0,
      stderrByteCount: 0,
      exit: { code: null, signal: 'SIGTERM' },
    },
    summary: {
      sampleCount: 2,
      gpuCount: 1,
      firstObservedAtIso: '2026-09-01T00:00:00.000Z',
      lastObservedAtIso: '2026-09-01T00:00:00.250Z',
      gpus: [{
        gpuIndex: 0,
        gpuName: 'Example GPU',
        gpuUuid,
        sampleCount: 2,
        marker: 'unchanged-summary-value',
      }],
    },
    coverageAudit,
    computeProcesses: {
      pre: snapshot(
        'pre-run',
        '2026-09-01T00:00:00.000Z',
        0,
        structuredClone(processes),
      ),
      post: snapshot(
        'post-run',
        '2026-09-01T00:01:00.000Z',
        60_000,
        [...structuredClone(processes)].reverse(),
      ),
    },
    acceptanceBoundary: {
      affectsTechnicalRunAcceptance: false,
      candidateEnvironmentReviewRequired: true,
      automaticPstateRejectionThreshold: null,
    },
    testRunId: runId,
  };
}

function metadataFixture(runId, telemetry) {
  const identities = telemetry.computeProcesses.pre.processes
    .map(identityRecord)
    .sort((left, right) => (
      left.gpuUuid.localeCompare(right.gpuUuid)
        || left.pid - right.pid
        || left.processName.localeCompare(right.processName)
    ));
  return {
    schemaVersion: 2,
    runId,
    status: 'complete',
    environment: {
      note: 'Private runner note from C:\\Users\\researcher\\workspace',
      gpuTelemetry: structuredClone(telemetry),
      retainedResearchField: 'unchanged',
    },
    evidenceStatus: 'candidate',
    protocol: { matrixKind: 'first-instance-live' },
    liveFirstInstanceEnvironmentAudit: {
      schemaVersion: 1,
      kind: 'first-instance-live-crossover-environment-audit',
      telemetryStatus: 'available',
      telemetryMalformedLineCount: 0,
      telemetryStderrByteCount: 0,
      telemetrySampleCount: 2,
      computeProcessIdentityComparison: {
        schemaVersion: 1,
        kind: 'nvidia-compute-process-identity-set-comparison',
        identityFields: ['gpuUuid', 'pid', 'processName'],
        ignoredFields: ['usedMemoryMiB'],
        pass: true,
        pre: structuredClone(identities),
        post: structuredClone(identities),
        reasons: [],
      },
    },
    resultMarker: 'unchanged-result-value',
  };
}

function telemetryCsvFixture(runId) {
  const base = {
    observedAtIso: '2026-09-01T00:00:00.000Z',
    runElapsedMs: 0,
    runId,
    trialId: '',
    planIndex: '',
    repetitionIndex: '',
    modeId: '',
    visibilityFraction: '',
    layout: '',
    phase: 'startup',
    gpuIndex: 0,
    gpuName: 'Example GPU',
    gpuUuid: 'GPU-private-machine-uuid',
    pstate: 'P1',
    graphicsClockMHz: 2500,
    memoryClockMHz: 10000,
    gpuUtilizationPercent: 80,
    memoryUtilizationPercent: 4,
    memoryUsedMiB: 1000,
    memoryTotalMiB: 16000,
    temperatureC: 50,
    powerDrawW: 100,
  };
  return Buffer.from([
    TELEMETRY_CSV_FIELDS.join(','),
    csvLine(base),
    csvLine({
      ...base,
      observedAtIso: '2026-09-01T00:00:00.250Z',
      runElapsedMs: 250,
      phase: 'measurement',
    }),
    '',
  ].join('\n'));
}

async function createPrivateFixture(root, mutate = () => {}) {
  const privateDirectory = path.join(root, 'private-run');
  await writeFile(path.join(root, '.placeholder'), 'fixture root');
  await mkdir(privateDirectory);
  const runId = 'first-instance-live-test-run';
  const telemetry = telemetryFixture(runId);
  const metadata = metadataFixture(runId, telemetry);
  const values = {
    'frames.csv': Buffer.from('runId,marker\nfirst-instance-live-test-run,unchanged\n'),
    'metadata.json': jsonBytes(metadata),
    'trial-summaries.json': jsonBytes([{ marker: 'unchanged-trial' }]),
    'validation-artifacts.json': jsonBytes([{ marker: 'unchanged-validation' }]),
    'workload-manifests.json': jsonBytes({ marker: 'unchanged-workload' }),
    'gpu-telemetry-summary.json': jsonBytes(telemetry),
    'forced-feature-off-evidence.json': jsonBytes({ accepted: true, marker: 'unchanged-gate' }),
    'gpu-telemetry.csv': telemetryCsvFixture(runId),
  };
  mutate({ values, metadata, telemetry });
  values['metadata.json'] = jsonBytes(metadata);
  values['gpu-telemetry-summary.json'] = jsonBytes(telemetry);
  for (const [name, contents] of Object.entries(values)) {
    await writeFile(path.join(privateDirectory, name), contents);
  }
  const manifest = {
    schemaVersion: 2,
    runId,
    hashAlgorithm: 'sha256',
    requiredFiles: [...REQUIRED_ARTIFACTS],
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: true,
      evidenceAvailable: true,
      absenceReason: null,
    }],
    files: Object.entries(values).map(([name, contents]) => ({
      name,
      role: `test role for ${name}`,
      required: REQUIRED_ARTIFACTS.includes(name),
      present: true,
      bytes: contents.length,
      sha256: sha256(contents),
      absenceReason: null,
    })),
  };
  await writeFile(path.join(privateDirectory, 'artifact-manifest.json'), jsonBytes(manifest));
  return { privateDirectory, runId, values, manifest };
}

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'live-evidence-sanitizer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function createCandidateRepository(root) {
  const repositoryRoot = path.join(root, 'candidate-repository');
  await mkdir(path.join(repositoryRoot, 'scripts'), { recursive: true });
  await mkdir(path.join(repositoryRoot, 'src', 'strategies'), { recursive: true });
  await mkdir(path.join(repositoryRoot, 'node_modules', 'runtime-fixture'), {
    recursive: true,
  });
  for (const repositoryPath of [
    'index.html',
    'package-lock.json',
    ...LIVE_SANITIZER_IMPLEMENTATION_PATHS,
  ]) {
    await copyFile(
      path.join(PROJECT_ROOT, repositoryPath),
      path.join(repositoryRoot, repositoryPath),
    );
  }
  for (const repositoryPath of CANDIDATE_VITE_REQUIRED_MODULE_PATHS) {
    await copyFile(
      path.join(PROJECT_ROOT, ...repositoryPath.split('/')),
      path.join(repositoryRoot, ...repositoryPath.split('/')),
    );
  }
  await writeFile(
    path.join(repositoryRoot, 'node_modules', 'runtime-fixture', 'index.js'),
    "export const fixture = 'runtime-dependency';\n",
  );
  await execFileAsync('git', ['init', '--quiet'], { cwd: repositoryRoot });
  await execFileAsync('git', ['add', '--force', '.'], { cwd: repositoryRoot });
  await execFileAsync('git', [
    '-c', 'user.name=Strict fixture',
    '-c', 'user.email=strict-fixture@example.invalid',
    'commit', '--quiet', '-m', 'strict sanitizer fixture',
  ], { cwd: repositoryRoot });
  return {
    repositoryRoot,
    provenance: await collectSourceProvenance(repositoryRoot),
  };
}

async function candidateViteRuntimeAuditFixture(repositoryRoot) {
  const entryHtmlContents = await readFile(path.join(repositoryRoot, 'index.html'));
  const entryHtmlSha256 = sha256(entryHtmlContents);
  const modulePaths = [
    'node_modules/runtime-fixture/index.js',
    ...CANDIDATE_VITE_REQUIRED_MODULE_PATHS,
  ].sort();
  const modules = await Promise.all(modulePaths.map(async (sourceRelativePath) => {
    const contents = await readFile(
      path.join(repositoryRoot, ...sourceRelativePath.split('/')),
    );
    const digest = sha256(contents);
    return {
      sourceRelativePath,
      sourceByteCount: contents.length,
      sourceSha256: digest,
      successfulResponseCount: 1,
      transformedVariants: [{
        byteCount: contents.length,
        sha256: digest,
        responseCount: 1,
      }],
    };
  }));
  return {
    schemaVersion: 3,
    kind: 'candidate-vite-runtime-module-audit',
    policyId: CANDIDATE_VITE_RUNTIME_POLICY_ID,
    configuration: {
      configFile: false,
      appType: 'custom',
      optimizeDepsNoDiscovery: true,
      optimizeDepsInclude: [],
      entryHtmlPolicy: 'exact-tracked-index-html-without-vite-client',
      cacheDirectoryPolicy: 'unique-fresh-os-temporary-directory-outside-project',
      cachePreexisted: false,
    },
    cache: { existedAtFinalization: false, entryCount: 0 },
    entryHtml: {
      sourceRelativePath: 'index.html',
      sourceByteCount: entryHtmlContents.length,
      sourceSha256: entryHtmlSha256,
      successfulResponseCount: 2,
      responseHeaders: {
        crossOriginOpenerPolicy: 'same-origin',
        crossOriginEmbedderPolicy: 'require-corp',
        contentType: 'text/html; charset=utf-8',
        cacheControl: 'no-store',
      },
      responseVariants: [{
        byteCount: entryHtmlContents.length,
        sha256: entryHtmlSha256,
        responseCount: 2,
      }],
    },
    prohibitedOptimizedArtifactCount: 0,
    requiredModulePaths: [...CANDIDATE_VITE_REQUIRED_MODULE_PATHS],
    moduleCount: modules.length,
    dependencyModuleCount: 1,
    modulesSha256: candidateViteRuntimeModulesSha256(modules),
    modules,
  };
}

async function inflateStrictLiveFixture(root, candidate) {
  const { provenance, repositoryRoot } = candidate;
  const privateDirectory = path.join(root, 'strict-private-run');
  await mkdir(privateDirectory);
  const packed = brotliDecompressSync(Buffer.from(
    (await readFile(STRICT_FIXTURE_PATH, 'utf8')).replaceAll(/\s/g, ''),
    'base64',
  ));
  let offset = 0;
  const names = [];
  while (offset < packed.length) {
    if (offset + 6 > packed.length) throw new Error('strict fixture has a truncated header');
    const nameLength = packed.readUInt16BE(offset);
    const contentsLength = packed.readUInt32BE(offset + 2);
    offset += 6;
    const end = offset + nameLength + contentsLength;
    if (end > packed.length) throw new Error('strict fixture has a truncated record');
    const name = packed.subarray(offset, offset + nameLength).toString('utf8');
    offset += nameLength;
    if (name.includes('/') || name.includes('\\') || names.includes(name)) {
      throw new Error(`strict fixture has an unsafe or duplicate name: ${name}`);
    }
    await writeFile(
      path.join(privateDirectory, name),
      packed.subarray(offset, offset + contentsLength),
      { flag: 'wx' },
    );
    offset += contentsLength;
    names.push(name);
  }
  assert.deepEqual(names.sort(), [
    ...REQUIRED_ARTIFACTS,
    'gpu-telemetry.csv',
  ].sort());
  const validationArtifactSha256 = await upgradeStrictFixtureSchema(privateDirectory);

  const telemetryPath = path.join(privateDirectory, 'gpu-telemetry-summary.json');
  const telemetry = JSON.parse(await readFile(telemetryPath));
  telemetry.computeProcesses.post.processes =
    telemetry.computeProcesses.post.processes.slice(1);
  for (const snapshot of Object.values(telemetry.computeProcesses)) {
    snapshot.rawNonemptyLineCount = snapshot.processes.length;
    snapshot.parsedRecordCount = snapshot.processes.length;
    snapshot.malformedLineCount = 0;
    snapshot.stdoutByteCount = Buffer.byteLength(JSON.stringify(snapshot.processes));
    snapshot.stdoutTruncated = false;
    snapshot.stderrByteCount = 0;
  }
  const telemetryCsv = await readFile(
    path.join(privateDirectory, 'gpu-telemetry.csv'),
    'utf8',
  );
  const telemetryLines = telemetryCsv.trimEnd().split(/\r?\n/);
  const telemetryHeaders = parseCsvHeader(telemetryLines[0]);
  const parsedTelemetry = telemetryLines.slice(1).map((line) => {
    const values = parseCsvHeader(line);
    const record = Object.fromEntries(telemetryHeaders.map(
      (header, index) => [header, values[index]],
    ));
    const nullableNumber = (value) => (value === '' ? null : Number(value));
    return { line, row: {
      observedAtIso: record.observedAtIso,
      runElapsedMs: Number(record.runElapsedMs),
      runId: record.runId,
      trialId: record.trialId === '' ? null : record.trialId,
      planIndex: nullableNumber(record.planIndex),
      repetitionIndex: nullableNumber(record.repetitionIndex),
      modeId: record.modeId === '' ? null : record.modeId,
      visibilityFraction: nullableNumber(record.visibilityFraction),
      layout: record.layout === '' ? null : record.layout,
      phase: record.phase,
      gpuIndex: nullableNumber(record.gpuIndex),
      gpuName: record.gpuName === '' ? null : record.gpuName,
      gpuUuid: record.gpuUuid === '' ? null : record.gpuUuid,
      pstate: record.pstate === '' ? null : record.pstate,
      ...Object.fromEntries([
        'graphicsClockMHz',
        'memoryClockMHz',
        'gpuUtilizationPercent',
        'memoryUtilizationPercent',
        'memoryUsedMiB',
        'memoryTotalMiB',
        'temperatureC',
        'powerDrawW',
      ].map((field) => [field, nullableNumber(record[field])])),
    } };
  });
  const retainedTelemetry = [];
  let previousElapsedMs = -Infinity;
  for (const record of parsedTelemetry) {
    if (record.row.runElapsedMs - previousElapsedMs <= 125) continue;
    retainedTelemetry.push(record);
    previousElapsedMs = record.row.runElapsedMs;
  }
  const telemetryRows = retainedTelemetry.map(({ row }) => row);
  await writeFile(
    path.join(privateDirectory, 'gpu-telemetry.csv'),
    `${telemetryLines[0]}\n${retainedTelemetry.map(({ line }) => line).join('\n')}\n`,
  );
  telemetry.summary = summarizeTelemetryRows(telemetryRows);
  const firstElapsedMs = Math.min(...telemetryRows.map(({ runElapsedMs }) => runElapsedMs));
  const lastElapsedMs = Math.max(...telemetryRows.map(({ runElapsedMs }) => runElapsedMs));
  telemetry.sampling.collectorStartedRunElapsedMs = Math.max(0, firstElapsedMs - 250);
  telemetry.sampling.collectorStopRequestedRunElapsedMs = lastElapsedMs + 250;
  telemetry.coverageAudit = createNvidiaTelemetryCoverageAudit(telemetryRows, {
    collectorStartedRunElapsedMs: telemetry.sampling.collectorStartedRunElapsedMs,
    collectorStopRequestedRunElapsedMs:
      telemetry.sampling.collectorStopRequestedRunElapsedMs,
    requestedIntervalMs: telemetry.sampling.requestedIntervalMs,
  });
  assert.equal(
    telemetry.coverageAudit.pass,
    true,
    JSON.stringify(telemetry.coverageAudit),
  );
  await writeFile(telemetryPath, jsonBytes(telemetry));

  const metadataPath = path.join(privateDirectory, 'metadata.json');
  const metadata = JSON.parse(await readFile(metadataPath));
  metadata.evidenceStatus = 'candidate';
  metadata.sourceProvenance = {
    start: structuredClone(provenance),
    end: structuredClone(provenance),
    stable: true,
  };
  const executionDependencyClosure = {
    schemaVersion: 1,
    kind: 'installed-execution-dependency-closure',
    root: 'node_modules',
    format: 'node-modules-sorted-path-size-content-sha256-v1',
    hashAlgorithm: 'sha256',
    exclusions: ['.bin/**', '.vite*/**'],
    fileCount: 1,
    totalBytes: 1,
    sha256: 'c'.repeat(64),
  };
  metadata.candidateSeriesReservation = {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-series-reservation',
    seriesId: 'strict-live-sanitizer-fixture',
    reservationEventSha256: 'd'.repeat(64),
    attemptOrdinal: 1,
    matrixOrdinal: 1,
    sourceCommit: provenance.commit,
    sourceTree: provenance.tree,
    executionDependencyClosureSha256: executionDependencyClosure.sha256,
  };
  metadata.executionDependencyClosure = {
    start: structuredClone(executionDependencyClosure),
    end: structuredClone(executionDependencyClosure),
    stable: true,
  };
  metadata.candidateViteRuntimeAudit = await candidateViteRuntimeAuditFixture(repositoryRoot);
  metadata.validationArtifactSha256 = validationArtifactSha256;
  const upgradedArtifacts = JSON.parse(await readFile(
    path.join(privateDirectory, 'validation-artifacts.json'),
    'utf8',
  ));
  metadata.workload.renderParitySha256ByCell = Object.fromEntries(
    upgradedArtifacts.map((artifact) => [
      `${artifact.layout}|${artifact.visibilityFraction}`,
      artifact.pre.renderParityOutputSha256,
    ]),
  );
  const protocolHistory = firstInstanceLiveCrossoverHistoryCounts(480, 0);
  metadata.protocol.ordering =
    'twelve-repetition-two-visibility-cyclic-eight-frame-live-crossover-with-exact-transition-history-balance-and-pairwise-balanced-lane-physical-order-visibility-order-and-orientation';
  metadata.protocol.firstInstanceLiveCrossover.patterns = ['PPPFPFFF', 'FFFPFPPP'];
  metadata.protocol.firstInstanceLiveCrossover.scheduleDesign =
    'cyclic-binary-de-bruijn-order-three-with-complementary-orientation';
  metadata.protocol.firstInstanceLiveCrossover.expectedMeasuredTransitionCounts = {
    ...protocolHistory.transitionCounts,
  };
  metadata.protocol.firstInstanceLiveCrossover.expectedMeasuredHistoryTripleCounts = {
    ...protocolHistory.historyTripleCounts,
  };
  metadata.protocol.firstInstanceLiveCrossover.scheduleSha256ByOrientation = {
    0: liveFirstInstanceCrossoverScheduleSha256(0),
    1: liveFirstInstanceCrossoverScheduleSha256(1),
  };
  const { artifactVerification: _artifactVerification, ...analysis } = summarizeCsv(
    await readFile(path.join(privateDirectory, 'frames.csv'), 'utf8'),
  );
  metadata.liveFirstInstanceAnalysisAudit = {
    schemaVersion: analysis.schemaVersion,
    kind: analysis.kind,
    nTrials: analysis.nTrials,
    nRows: analysis.nRows,
    preregisteredDecision: analysis.preregisteredDecision,
    preregisteredNumericalDecision: analysis.preregisteredNumericalDecision,
    sha256: sha256(JSON.stringify(analysis)),
  };
  metadata.environment.gpuTelemetry = structuredClone(telemetry);
  const comparison = compareComputeProcessIdentitySets(
    telemetry.computeProcesses.pre,
    telemetry.computeProcesses.post,
  );
  assert.equal(comparison.pass, false);
  metadata.liveFirstInstanceEnvironmentAudit = createLiveFirstInstanceEnvironmentAudit({
    evidenceStatus: metadata.evidenceStatus,
    telemetryReport: telemetry,
    adapterInfo: metadata.environment.benchmarkPage.adapterInfo,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision:
      metadata.liveFirstInstanceAnalysisAudit.preregisteredNumericalDecision,
  });
  await writeFile(metadataPath, jsonBytes(metadata));

  const roles = {
    'frames.csv': 'frame-level timing rows',
    'metadata.json': 'run protocol, environment, provenance, and completion state',
    'trial-summaries.json': 'per-trial acceptance and timing summaries',
    'validation-artifacts.json': 'crash-safe pre/post correctness and workload evidence',
    'workload-manifests.json': 'deduplicated geometry and scenario fingerprint manifests',
    'gpu-telemetry-summary.json': 'telemetry availability and process-snapshot summary',
    'forced-feature-off-evidence.json':
      'isolated portable deployment-selection, correctness, address, and output gate',
    'gpu-telemetry.csv': 'optional device telemetry samples',
  };
  const manifest = {
    schemaVersion: 2,
    runId: metadata.runId,
    hashAlgorithm: 'sha256',
    requiredFiles: [...REQUIRED_ARTIFACTS],
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: true,
      evidenceAvailable: true,
      absenceReason: null,
    }],
    files: [],
  };
  for (const name of [...REQUIRED_ARTIFACTS, 'gpu-telemetry.csv']) {
    const contents = await readFile(path.join(privateDirectory, name));
    manifest.files.push({
      name,
      role: roles[name],
      required: REQUIRED_ARTIFACTS.includes(name),
      present: true,
      bytes: contents.length,
      sha256: sha256(contents),
      absenceReason: null,
    });
  }
  await writeFile(
    path.join(privateDirectory, 'artifact-manifest.json'),
    jsonBytes(manifest),
  );
  return { privateDirectory, metadata, telemetry };
}

async function directoryDigests(directory) {
  return Object.fromEntries(await Promise.all(
    (await readdir(directory)).sort().map(async (name) => [
      name,
      sha256(await readFile(path.join(directory, name))),
    ]),
  ));
}

test('strict live candidate sanitizes to an independently accepted, commit-bound public bundle', async (t) => {
  const root = await temporaryRoot(t);
  const candidate = await createCandidateRepository(root);
  const fixture = await inflateStrictLiveFixture(root, candidate);
  const privateBefore = await directoryDigests(fixture.privateDirectory);
  const privateVerification = await verifyRunDirectory(fixture.privateDirectory, {
    repositoryRoot: candidate.repositoryRoot,
  });
  assert.equal(privateVerification.artifactVerification.evidenceStatus, 'candidate');
  assert.equal(
    privateVerification.liveFirstInstanceEvidenceDecision.candidateEnvironmentGate.status,
    'failed-non-replaceable-process-set-mismatch',
  );
  assert.equal(
    privateVerification.liveFirstInstanceEvidenceDecision.candidateEnvironmentGate.retryable,
    false,
  );

  const artifacts = JSON.parse(await readFile(
    path.join(fixture.privateDirectory, 'validation-artifacts.json'),
    'utf8',
  ));
  const workloadCatalog = JSON.parse(await readFile(
    path.join(fixture.privateDirectory, 'workload-manifests.json'),
    'utf8',
  ));
  const artifact = artifacts[0];
  const planned = fixture.metadata.plan.find((entry) => entry.trialId === artifact.trialId);
  const geometryManifest = workloadCatalog.geometryFixturesBySha256[
    artifact.pre.workload.geometryFixtureSha256
  ];
  const scenarioManifest = workloadCatalog.scenariosBySha256[
    artifact.pre.workload.scenarioSha256
  ];
  const validationOptions = {
    spec: planned,
    environment: fixture.metadata.environment.benchmarkPage,
    geometryManifest,
    scenarioManifest,
  };

  assert.deepEqual(validateLiveFirstInstanceCrossoverRenderParity(
    artifact.pre.renderParity,
    {
      spec: planned,
      validation: artifact.pre.validation.payload,
      scenarioManifest,
    },
  ), []);
  const skippedProductionRender = structuredClone(artifact.pre.renderParity);
  skippedProductionRender.lanes.portable.productionBundleOutput
    .executionBetween.rendererRenderCallSerial -= 1;
  assert.ok(validateLiveFirstInstanceCrossoverRenderParity(
    skippedProductionRender,
    {
      spec: planned,
      validation: artifact.pre.validation.payload,
      scenarioManifest,
    },
  ).some((reason) => reason.includes('first production render')));

  const computeNodeAttack = structuredClone(artifact.pre.validation.payload);
  for (const boundary of ['resourceIdentitiesAtStart', 'resourceIdentitiesAtEnd']) {
    computeNodeAttack.shaderEvidence.observation[boundary]
      .compute.portable.reset.computeNodeId = 'forged-compute-node';
  }
  rehashFixtureShaderEvidence(computeNodeAttack.shaderEvidence);
  assert.ok((await validateLiveFirstInstanceCrossoverValidation(
    computeNodeAttack,
    validationOptions,
  )).some((reason) => reason.includes('compute-node lifecycle identity')));

  const sharedResourceAttack = structuredClone(artifact.pre.validation.payload);
  for (const lane of ['portable', 'feature']) {
    sharedResourceAttack.shaderEvidence.render[lane]
      .storageBindings.matrix.resourceId = 900_001;
  }
  rehashFixtureShaderEvidence(sharedResourceAttack.shaderEvidence);
  assert.ok((await validateLiveFirstInstanceCrossoverValidation(
    sharedResourceAttack,
    validationOptions,
  )).some((reason) => reason.includes('shared-resource lifecycle identity')));

  const indirectResourceAttack = structuredClone(artifact.pre.validation.payload);
  for (const phase of ['reset', 'cull']) {
    const phaseEvidence = indirectResourceAttack.shaderEvidence.compute.phases[phase];
    const comparison = phaseEvidence.bindings.resourceComparisons.find(
      (record) => record.semantic === 'indirectCommands',
    );
    comparison.portableResourceId = 900_002;
    comparison.featureResourceId = 900_003;
    for (const [lane, resourceId] of [
      ['portable', comparison.portableResourceId],
      ['feature', comparison.featureResourceId],
    ]) {
      phaseEvidence.lanes[lane].bindings.find(
        (binding) => binding.semantic === 'indirectCommands',
      ).resourceId = resourceId;
    }
  }
  rehashFixtureShaderEvidence(indirectResourceAttack.shaderEvidence);
  assert.ok((await validateLiveFirstInstanceCrossoverValidation(
    indirectResourceAttack,
    validationOptions,
  )).some((reason) => reason.includes('command-buffer lifecycle identity')));

  const publicOutput = path.join(root, 'private-user-path-sentinel-public-output');
  const result = await sanitizeLiveEvidence(fixture.privateDirectory, publicOutput, {
    repositoryRoot: candidate.repositoryRoot,
  });
  const cliStdout = serializeLiveEvidenceSanitizerResult(result);
  assert.equal(Object.hasOwn(result, 'publicOutputDirectory'), false);
  assert.equal(cliStdout.includes(publicOutput), false);
  assert.equal(cliStdout.includes(root), false);
  assert.doesNotMatch(cliStdout, /publicOutputDirectory|[A-Za-z]:\\/);
  const publicVerification = await verifyRunDirectory(publicOutput, {
    repositoryRoot: candidate.repositoryRoot,
  });

  assert.equal(result.privateBundleLabel, 'private-original');
  assert.equal(result.publicBundleLabel, 'public-derived');
  assert.equal(result.copiedArtifactCount, 8);
  assert.notEqual(
    result.privateArtifactManifestSha256,
    result.publicArtifactManifestSha256,
  );
  assert.deepEqual(await directoryDigests(fixture.privateDirectory), privateBefore);
  assert.equal(publicVerification.bundleIntegrity.bundleLabel, 'public-derived');
  assert.equal(
    publicVerification.bundleIntegrity.sanitizerImplementationCandidateCommitMatch,
    true,
  );
  assert.equal(publicVerification.bundleIntegrity.privateSourceRelationshipVerified, false);
  assert.equal(publicVerification.bundleIntegrity.authenticityVerified, false);

  const publicManifestPath = path.join(publicOutput, 'artifact-manifest.json');
  const publicManifestBytes = await readFile(publicManifestPath);
  const publicManifest = JSON.parse(await readFile(
    publicManifestPath,
    'utf8',
  ));
  assert.equal(publicManifest.bundleProvenance.bundleLabel, 'public-derived');
  assert.equal(publicManifest.bundleProvenance.sourceBundleLabel, 'private-original');
  assert.equal(
    publicManifest.bundleProvenance.privateArtifactManifest.sha256,
    privateBefore['artifact-manifest.json'],
  );
  assert.equal(
    publicManifest.bundleProvenance.sanitizer.candidateCommit,
    candidate.provenance.commit,
  );
  assert.deepEqual(
    publicManifest.bundleProvenance.sanitizer.files.map(({ repositoryPath }) => repositoryPath),
    LIVE_SANITIZER_IMPLEMENTATION_PATHS,
  );
  assert.deepEqual(
    publicManifest.bundleProvenance.redactionPolicy.jsonPathAllowlist,
    PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST,
  );
  assert.deepEqual(
    publicManifest.bundleProvenance.redactionPolicy.csvColumnAllowlist,
    PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST,
  );
  for (const entry of publicManifest.files) {
    const contents = await readFile(path.join(publicOutput, entry.name));
    assert.equal(entry.bytes, contents.length);
    assert.equal(entry.sha256, sha256(contents));
  }

  const privateMetadata = JSON.parse(await readFile(
    path.join(fixture.privateDirectory, 'metadata.json'),
  ));
  const publicMetadata = JSON.parse(await readFile(path.join(publicOutput, 'metadata.json')));
  const publicTelemetry = JSON.parse(await readFile(
    path.join(publicOutput, 'gpu-telemetry-summary.json'),
  ));
  assert.deepEqual(
    publicMetadata.candidateViteRuntimeAudit,
    privateMetadata.candidateViteRuntimeAudit,
  );
  assert.equal(
    Object.hasOwn(
      publicMetadata.candidateViteRuntimeAudit.configuration,
      'cacheDirectory',
    ),
    false,
  );
  assert.equal(
    JSON.stringify(publicMetadata.candidateViteRuntimeAudit).includes(root),
    false,
  );
  assert.equal(publicMetadata.environment.note, '[redacted: unrelated private environment note]');
  assert.deepEqual(publicMetadata.environment.gpuTelemetry, publicTelemetry);
  assert.equal(publicTelemetry.summary.gpus[0].gpuUuid, 'GPU-PUBLIC-DEVICE-0');
  assert.equal(
    publicTelemetry.coverageAudit.gpuIdentities[0].gpuUuid,
    'GPU-PUBLIC-DEVICE-0',
  );
  assert.equal(publicTelemetry.coverageAudit.pass, true);
  const publicPre = publicTelemetry.computeProcesses.pre.processes;
  const publicPost = publicTelemetry.computeProcesses.post.processes;
  assert.equal(publicPre.length, privateMetadata.environment.gpuTelemetry.computeProcesses.pre.processes.length);
  assert.equal(publicPost.length, privateMetadata.environment.gpuTelemetry.computeProcesses.post.processes.length);
  assert.notEqual(publicPre.length, publicPost.length);
  for (const process of [...publicPre, ...publicPost]) {
    assert.equal(process.gpuUuid, 'GPU-PUBLIC-DEVICE-0');
    assert.match(process.processName, /^resident-gpu-process-[1-9][0-9]*\.redacted$/);
    assert.ok(Number.isSafeInteger(process.pid) && process.pid > 100_000);
  }
  assert.equal(publicMetadata.status, 'complete');
  assert.equal(publicMetadata.error, null);
  assert.deepEqual(
    publicMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
    privateMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
  );
  assert.equal(
    publicMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation.pass,
    true,
  );
  assert.equal(
    JSON.stringify(
      publicMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
    ).includes('GPU-STRICT-LIVE-TEST-DEVICE'),
    false,
  );
  assert.equal(
    publicMetadata.liveFirstInstanceEnvironmentAudit.candidateEnvironmentGate.status,
    'failed-non-replaceable-process-set-mismatch',
  );
  assert.deepEqual(
    publicVerification.liveFirstInstanceEvidenceDecision,
    {
      adapterTelemetryAssociation:
        publicMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
      candidateEnvironmentGate:
        publicMetadata.liveFirstInstanceEnvironmentAudit.candidateEnvironmentGate,
      overallEvidenceDecision:
        publicMetadata.liveFirstInstanceEnvironmentAudit.overallEvidenceDecision,
    },
  );

  const telemetryLines = (await readFile(path.join(publicOutput, 'gpu-telemetry.csv'), 'utf8'))
    .trimEnd().split('\n');
  const headers = parseCsvHeader(telemetryLines[0]);
  const uuidIndex = headers.indexOf('gpuUuid');
  const gpuNameIndex = headers.indexOf('gpuName');
  for (const line of telemetryLines.slice(1)) {
    const record = parseCsvHeader(line);
    assert.equal(record[uuidIndex], 'GPU-PUBLIC-DEVICE-0');
    assert.equal(
      record[gpuNameIndex],
      privateMetadata.environment.gpuTelemetry.summary.gpus[0].gpuName,
    );
  }

  const framesPath = path.join(publicOutput, 'frames.csv');
  const framesBytes = await readFile(framesPath);
  await writeFile(framesPath, Buffer.concat([framesBytes, Buffer.from('tampered\n')]));
  await assert.rejects(
    verifyRunDirectory(publicOutput, { repositoryRoot: candidate.repositoryRoot }),
    /frames\.csv byte size|frames\.csv SHA-256/,
  );
  await writeFile(framesPath, framesBytes);

  const malformedFrameLines = framesBytes.toString('utf8').trimEnd().split(/\r?\n/);
  const malformedFrameHeaders = parseCsvHeader(malformedFrameLines[0]);
  const malformedFirstFrame = parseCsvHeader(malformedFrameLines[1]);
  malformedFirstFrame[
    malformedFrameHeaders.indexOf('rendererMemoryAtTimingStart')
  ] = '{invalid-json';
  malformedFrameLines[1] = malformedFirstFrame.map(csvCell).join(',');
  const malformedFramesBytes = Buffer.from(`${malformedFrameLines.join('\n')}\n`);
  await writeFile(framesPath, malformedFramesBytes);
  const malformedFramesManifest = structuredClone(publicManifest);
  const malformedFramesEntry = malformedFramesManifest.files.find(
    ({ name }) => name === 'frames.csv',
  );
  malformedFramesEntry.bytes = malformedFramesBytes.length;
  malformedFramesEntry.sha256 = sha256(malformedFramesBytes);
  await writeFile(publicManifestPath, jsonBytes(malformedFramesManifest));
  await assert.rejects(
    verifyRunDirectory(publicOutput, { repositoryRoot: candidate.repositoryRoot }),
    /invalid JSON in rendererMemoryAtTimingStart/,
  );
  await writeFile(framesPath, framesBytes);
  await writeFile(publicManifestPath, publicManifestBytes);

  const publicSummariesPath = path.join(publicOutput, 'trial-summaries.json');
  const publicSummariesBytes = await readFile(publicSummariesPath);
  const changedSummaries = JSON.parse(publicSummariesBytes);
  changedSummaries[0].timestamps.undeclaredTimestampEvidence = true;
  const changedSummariesBytes = jsonBytes(changedSummaries);
  await writeFile(publicSummariesPath, changedSummariesBytes);
  const changedSummariesManifest = structuredClone(publicManifest);
  const changedSummariesEntry = changedSummariesManifest.files.find(
    ({ name }) => name === 'trial-summaries.json',
  );
  changedSummariesEntry.bytes = changedSummariesBytes.length;
  changedSummariesEntry.sha256 = sha256(changedSummariesBytes);
  await writeFile(publicManifestPath, jsonBytes(changedSummariesManifest));
  await assert.rejects(
    verifyRunDirectory(publicOutput, { repositoryRoot: candidate.repositoryRoot }),
    /timestamps has an unexpected schema/,
  );
  await writeFile(publicSummariesPath, publicSummariesBytes);
  await writeFile(publicManifestPath, publicManifestBytes);

  const changedManifest = structuredClone(publicManifest);
  changedManifest.bundleProvenance.sanitizer.files[0].sha256 = '0'.repeat(64);
  await writeFile(publicManifestPath, jsonBytes(changedManifest));
  await assert.rejects(
    verifyRunDirectory(publicOutput, { repositoryRoot: candidate.repositoryRoot }),
    /implementation hashes do not match the recorded candidate commit/,
  );
  await writeFile(publicManifestPath, publicManifestBytes);

  const undeclaredPath = path.join(publicOutput, 'undeclared.txt');
  await writeFile(undeclaredPath, 'undeclared');
  await assert.rejects(
    verifyRunDirectory(publicOutput, { repositoryRoot: candidate.repositoryRoot }),
    /undeclared or non-artifact entries/,
  );
  await rm(undeclaredPath);

  const committedPolicyPath = path.join(
    candidate.repositoryRoot,
    LIVE_SANITIZER_IMPLEMENTATION_PATHS[0],
  );
  const committedPolicyBytes = await readFile(committedPolicyPath);
  await writeFile(
    committedPolicyPath,
    Buffer.concat([committedPolicyBytes, Buffer.from('\n// dirty test\n')]),
  );
  await assert.rejects(
    sanitizeLiveEvidence(
      fixture.privateDirectory,
      path.join(root, 'dirty-source-output'),
      { repositoryRoot: candidate.repositoryRoot },
    ),
    /sanitizer source is not the clean recorded candidate source tree/,
  );
  await writeFile(committedPolicyPath, committedPolicyBytes);

  await writeFile(path.join(candidate.repositoryRoot, 'later-checkout.txt'), 'later checkout\n');
  await execFileAsync('git', ['add', 'later-checkout.txt'], { cwd: candidate.repositoryRoot });
  await execFileAsync('git', [
    '-c', 'user.name=Strict fixture',
    '-c', 'user.email=strict-fixture@example.invalid',
    'commit', '--quiet', '-m', 'later checkout',
  ], { cwd: candidate.repositoryRoot });
  assert.equal(
    (await verifyRunDirectory(publicOutput, {
      repositoryRoot: candidate.repositoryRoot,
    })).bundleIntegrity.sanitizerImplementationCandidateCommitMatch,
    true,
  );
  await assert.rejects(
    sanitizeLiveEvidence(
      fixture.privateDirectory,
      path.join(root, 'other-commit-output'),
      { repositoryRoot: candidate.repositoryRoot },
    ),
    /sanitizer source is not the clean recorded candidate source tree/,
  );
});

function parseCsvHeader(line) {
  const fields = [];
  let value = '';
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === ',' && !quoted) {
      fields.push(value);
      value = '';
    } else {
      value += character;
    }
  }
  fields.push(value);
  return fields;
}

test('live evidence sanitizer rejects a private artifact that no longer matches its manifest', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root);
  await writeFile(path.join(fixture.privateDirectory, 'frames.csv'), 'changed\n');
  const output = path.join(root, 'public');
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, output),
    /frames\.csv does not match the private artifact manifest/,
  );
  await assert.rejects(readFile(path.join(output, 'artifact-manifest.json')), /ENOENT/);
});

test('live evidence sanitizer rejects an undeclared private directory entry', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root);
  await writeFile(path.join(fixture.privateDirectory, 'undeclared.txt'), 'undeclared');
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, path.join(root, 'public')),
    /private run directory contains undeclared or non-artifact entries/,
  );
});

test('live evidence sanitizer rejects an unexpected sensitive JSON path', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root, ({ values }) => {
    values['forced-feature-off-evidence.json'] = jsonBytes({
      accepted: true,
      hostname: 'private-workstation',
    });
  });
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, path.join(root, 'public')),
    /sensitive key "hostname" at unlisted JSON path \$\.hostname/,
  );
});

test('live evidence sanitizer rejects a machine-local path outside the environment-note allowlist', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root, ({ values }) => {
    values['validation-artifacts.json'] = jsonBytes([{
      error: 'unexpected fixture at C:\\Users\\private\\fixture.bin',
    }]);
  });
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, path.join(root, 'public')),
    /machine-local path at unlisted JSON path \$\[\*\]\.error/,
  );
});

test('live evidence sanitizer rejects an unexpected sensitive telemetry column', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root, ({ values }) => {
    const lines = values['gpu-telemetry.csv'].toString('utf8').trimEnd().split('\n');
    values['gpu-telemetry.csv'] = Buffer.from([
      `${lines[0]},pid`,
      ...lines.slice(1).map((line) => `${line},1234`),
      '',
    ].join('\n'));
  });
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, path.join(root, 'public')),
    /sensitive column "pid" outside the CSV allowlist/,
  );
});

test('live evidence sanitizer refuses an output inside the immutable private bundle', async (t) => {
  const root = await temporaryRoot(t);
  const fixture = await createPrivateFixture(root);
  await assert.rejects(
    sanitizeLiveEvidence(fixture.privateDirectory, path.join(fixture.privateDirectory, 'public')),
    /must not be the private directory or one of its descendants/,
  );
});

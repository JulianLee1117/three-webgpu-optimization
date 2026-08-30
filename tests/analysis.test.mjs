import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, unlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import {
  parseCsv,
  summarizeCsv,
  summarizeInput,
  verifyRunDirectory,
} from '../analysis/summarize.mjs';

const execFileAsync = promisify(execFile);
const ANALYZER_PATH = path.resolve('analysis/summarize.mjs');
const GEOMETRY_SHA256 = '1'.repeat(64);
const SCENARIO_SHA256 = '2'.repeat(64);
const SEMANTIC_SHA256 = '3'.repeat(64);

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

async function createVerifiedRunFixture(t, mutate = () => undefined) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'three-webgpu-analysis-'));
  t.after(async () => {
    await rm(directory, { recursive: true, force: true });
  });
  const runId = 'focused-o4096-b4-test';
  const trialId = `${runId}-t01`;
  const payload = { pass: true, kind: 'fixed-slice-exact-membership' };
  const capture = () => ({
    capturedAt: '2026-08-29T20:00:01.000Z',
    workload: {
      scenarioSeed: 123,
      geometryFixtureSha256: GEOMETRY_SHA256,
      scenarioSha256: SCENARIO_SHA256,
    },
    validation: {
      payloadSha256: sha256(JSON.stringify(payload)),
      semanticSha256: SEMANTIC_SHA256,
      payload,
    },
    accepted: true,
    rejectionReasons: [],
  });
  const validationArtifact = {
    schemaVersion: 2,
    trialId,
    planIndex: 0,
    repetitionIndex: 0,
    modeId: 'fixed-slice',
    visibilityFraction: 0.2,
    objectCount: 4096,
    bucketCount: 4,
    selectedConfig: { modeId: 'fixed-slice' },
    status: 'accepted',
    rejectionReasons: [],
    pre: capture(),
    timingStart: capture(),
    post: capture(),
  };
  validationArtifact.sha256 = sha256(JSON.stringify(validationArtifact));
  const planEntry = {
    runId,
    trialId,
    planIndex: 0,
    repetitionIndex: 0,
    modeId: 'fixed-slice',
    visibilityFraction: 0.2,
    objectCount: 4096,
    bucketCount: 4,
  };
  const telemetrySummary = { status: 'unavailable', reason: 'test fixture' };
  const metadata = {
    schemaVersion: 2,
    runId,
    status: 'complete',
    startedAt: '2026-08-29T20:00:00.000Z',
    completedAt: '2026-08-29T20:00:02.000Z',
    elapsedMs: 2000,
    environment: { gpuTelemetry: telemetrySummary },
    evidenceStatus: 'development',
    sourceProvenance: {
      start: { status: 'unavailable' },
      end: { status: 'unavailable' },
      stable: null,
    },
    workload: {
      scenarioGenerator: 'createFixedSubsetScenario',
      scenarioSeed: 123,
      manifestArtifact: 'workload-manifests.json',
      geometryFixtureSha256: GEOMETRY_SHA256,
      scenarioSha256ByVisibility: { '0.2': SCENARIO_SHA256 },
    },
    protocol: {
      matrix: 'focused-o4096-b4',
      objectCount: 4096,
      bucketCount: 4,
      modes: ['fixed-slice'],
      visibilityLevels: [0.2],
      repetitions: 1,
      warmupFrames: 1,
      measuredFrames: 1,
    },
    plan: [planEntry],
    expectedTrialCount: 1,
    completedTrialCount: 1,
    acceptedTrialCount: 1,
    frameRowCount: 1,
    validationArtifactCount: 1,
    validationArtifactSha256: [validationArtifact.sha256],
    pageErrors: [],
    error: null,
  };
  const trialSummaries = [{
    ...planEntry,
    startedAt: '2026-08-29T20:00:00.000Z',
    completedAt: '2026-08-29T20:00:02.000Z',
    elapsedMs: 2000,
    validation: { pass: true, artifactSha256: validationArtifact.sha256 },
    timestamps: {
      accepted: true,
      available: true,
      rowCount: 1,
      missingRenderFrames: 0,
      missingComputeFrames: 0,
    },
    accepted: true,
    rejectionReasons: [],
  }];
  const workloadManifests = {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    geometryFixturesBySha256: {
      [GEOMETRY_SHA256]: { sha256: GEOMETRY_SHA256 },
    },
    scenariosBySha256: {
      [SCENARIO_SHA256]: { sha256: SCENARIO_SHA256 },
    },
    invalidObservations: [],
  };
  const framesCsv = [
    'runId,trialId,planIndex,repetitionIndex,frameIndex,modeId,targetVisibilityFraction,objectCount,bucketCount,validationPass,timestampAvailable,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuCommonUpdateMs,cpuComputeSubmitMs,cpuRenderSubmitMs,cpuFrameBodyMs,cpuSubmitTotalMs',
    `${runId},${trialId},0,0,0,fixed-slice,0.2,4096,4,true,true,true,4,1,3,0.3,0.5,1,2.1,1.5`,
  ].join('\n');
  const fixture = {
    directory,
    metadata,
    trialSummaries,
    validationArtifacts: [validationArtifact],
    workloadManifests,
    telemetrySummary,
    framesCsv,
  };
  mutate(fixture);

  const artifacts = new Map([
    ['frames.csv', Buffer.from(fixture.framesCsv)],
    ['metadata.json', jsonBytes(fixture.metadata)],
    ['trial-summaries.json', jsonBytes(fixture.trialSummaries)],
    ['validation-artifacts.json', jsonBytes(fixture.validationArtifacts)],
    ['workload-manifests.json', jsonBytes(fixture.workloadManifests)],
    ['gpu-telemetry-summary.json', jsonBytes(fixture.telemetrySummary)],
  ]);
  await Promise.all([...artifacts].map(([name, contents]) => (
    writeFile(path.join(directory, name), contents)
  )));
  const requiredFiles = [...artifacts.keys()];
  const manifest = {
    schemaVersion: 2,
    runId: fixture.metadata.runId,
    hashAlgorithm: 'sha256',
    requiredFiles,
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: false,
      evidenceAvailable: false,
      absenceReason: 'test fixture has no device telemetry',
    }],
    files: [
      ...[...artifacts].map(([name, contents]) => ({
        name,
        role: `test ${name}`,
        required: true,
        present: true,
        bytes: contents.length,
        sha256: sha256(contents),
        absenceReason: null,
      })),
      {
        name: 'gpu-telemetry.csv',
        role: 'optional device telemetry samples',
        required: false,
        present: false,
        bytes: null,
        sha256: null,
        absenceReason: 'test fixture has no device telemetry',
      },
    ],
  };
  await writeFile(path.join(directory, 'artifact-manifest.json'), jsonBytes(manifest));
  return fixture;
}

test('CSV parsing preserves quoted commas and escaped quotes', () => {
  const parsed = parseCsv('name,note\r\n"fixed,slice","two ""dispatches"""\r\n');
  assert.deepEqual(parsed.records, [{ name: 'fixed,slice', note: 'two "dispatches"' }]);
});

test('analysis pairs modes by repetitionIndex within a visibility cell', () => {
  const csv = [
    'modeId,targetVisibilityFraction,repetitionIndex,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuCommonUpdateMs,cpuComputeSubmitMs,cpuRenderSubmitMs,cpuFrameBodyMs,cpuSubmitTotalMs',
    'draw-all,0.2,0,false,10,,10,0.1,,1,1.4,1',
    'three-blocks-current,0.2,0,true,8,2,6,0.8,0.8,1.2,3.2,2',
    'three-blocks-historical,0.2,0,true,7,1,6,0.6,0.6,1.2,2.8,1.8',
    'fixed-slice,0.2,0,true,4,1,3,0.3,0.5,1,2.1,1.5',
    'draw-all,0.2,1,false,12,,12,0.2,,1.1,1.6,1.1',
    'three-blocks-current,0.2,1,true,10,2,8,1,0.9,1.2,3.6,2.1',
    'three-blocks-historical,0.2,1,true,9,1,8,0.8,0.7,1.2,3.2,1.9',
    'fixed-slice,0.2,1,true,6,1,5,0.5,0.6,1,2.3,1.6',
  ].join('\n');

  const summary = summarizeCsv(csv);
  assert.equal(summary.repetitionColumn, 'repetitionIndex');
  const fixed = summary.groups.find((group) => group.modeId === 'fixed-slice');
  const drawAll = summary.groups.find((group) => group.modeId === 'draw-all');
  assert.equal(fixed.nTrials, 2);
  assert.equal(drawAll.medianAcrossTrials.gpuPassTotalMs, 11);
  assert.equal(fixed.medianAcrossTrials.cpuCommonUpdateMs, 0.4);
  assert.equal(fixed.medianAcrossTrials.cpuFrameBodyMs, 2.2);
  assert.equal(fixed.medianAcrossTrials.cpuSubmitTotalMs, 1.55);
  assert.equal(fixed.medianAcrossTrials.cpuRenderSubmitMs, 1);
  assert.equal(fixed.medianAcrossTrials.accountedCpuSubmitPlusGpuPassMs, 6.55);
  const versusDrawAll = summary.comparisons.fixedSliceVsDrawAll[0];
  const versusThreeBlocks = summary.comparisons.fixedSliceVsThreeBlocksCurrent[0];
  const versusHistorical = summary.comparisons.fixedSliceVsThreeBlocksHistorical[0];
  assert.equal(versusDrawAll.nPairs, 2);
  assert.equal(versusThreeBlocks.nPairs, 2);
  assert.equal(versusHistorical.nPairs, 2);
  assert.equal(versusDrawAll.medianPairedDelta.gpuPassTotalMs.absoluteMs, -6);
  assert.equal(versusThreeBlocks.medianPairedDelta.gpuPassTotalMs.absoluteMs, -4);
  assert.equal(versusHistorical.medianPairedDelta.gpuPassTotalMs.absoluteMs, -3);
  assert.ok(Math.abs(versusDrawAll.medianPairedDelta.cpuCommonUpdateMs.absoluteMs - 0.25) < 1e-12);
  assert.ok(Math.abs(versusThreeBlocks.medianPairedDelta.cpuFrameBodyMs.absoluteMs + 1.2) < 1e-12);
  assert.ok(Math.abs(versusHistorical.medianPairedDelta.cpuSubmitTotalMs.absoluteMs + 0.3) < 1e-12);
});

test('analysis rejects missing GPU compute data for compute modes', () => {
  const csv = [
    'modeId,targetVisibilityFraction,repetitionIndex,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuCommonUpdateMs,cpuComputeSubmitMs,cpuRenderSubmitMs,cpuFrameBodyMs,cpuSubmitTotalMs',
    'fixed-slice,0.2,0,true,4,,3,0.3,0.5,1,2.1,1.5',
  ].join('\n');
  assert.throws(() => summarizeCsv(csv), /compute mode but has no gpuComputeMs/);
});

test('analysis requires full-frame and common-update CPU timing without redefining submit timing', () => {
  const csv = [
    'modeId,targetVisibilityFraction,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuSubmitTotalMs',
    'draw-all,0.2,false,10,,10,1',
  ].join('\n');
  assert.throws(
    () => summarizeCsv(csv),
    /CSV is missing required columns: cpuCommonUpdateMs, cpuComputeSubmitMs, cpuRenderSubmitMs, cpuFrameBodyMs/,
  );
});

test('raw CSV analysis is explicitly unverified', async (t) => {
  const fixture = await createVerifiedRunFixture(t);
  const rawSummary = await summarizeInput(path.join(fixture.directory, 'frames.csv'));
  assert.deepEqual(rawSummary.artifactVerification, {
    status: 'unverified',
    scope: 'artifact-integrity-and-schema-only',
    authenticityVerified: false,
    inputKind: 'raw-frames-csv',
    evidenceStatus: null,
    reason: 'Standalone CSV input is not bound to a consistent run artifact manifest.',
  });
  assert.equal(rawSummary.nFrames, 1);
});

test('run-directory analysis verifies artifacts, acceptance, and counts before summarizing', async (t) => {
  const fixture = await createVerifiedRunFixture(t);
  const summary = await summarizeInput(fixture.directory);
  assert.deepEqual(summary.artifactVerification, {
    status: 'consistent',
    scope: 'artifact-integrity-and-schema-only',
    authenticityVerified: false,
    inputKind: 'run-directory',
    runId: fixture.metadata.runId,
    evidenceStatus: 'development',
    manifestSchemaVersion: 2,
    verifiedArtifactCount: 6,
    requiredArtifactCount: 6,
    completedTrialCount: 1,
    acceptedTrialCount: 1,
    sourceProvenanceStable: null,
  });
  assert.equal(summary.nFrames, 1);
  assert.equal(summary.groups[0].modeId, 'fixed-slice');
});

test('run-directory analysis rejects a byte-valid failed run', async (t) => {
  const fixture = await createVerifiedRunFixture(t, ({ metadata }) => {
    metadata.status = 'failed';
    metadata.error = { name: 'Error', message: 'fixture failure', stack: null };
  });
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /metadata\.json status must be "complete"/,
  );
});

test('run-directory analysis rejects tampering after manifest creation', async (t) => {
  const fixture = await createVerifiedRunFixture(t);
  await writeFile(path.join(fixture.directory, 'frames.csv'), `${fixture.framesCsv}\n`);
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /frames\.csv byte size .* manifest records/,
  );
});

test('run-directory analysis rejects missing required artifacts', async (t) => {
  const fixture = await createVerifiedRunFixture(t);
  await unlink(path.join(fixture.directory, 'validation-artifacts.json'));
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /artifact validation-artifacts\.json is missing/,
  );
});

test('run-directory analysis rejects inconsistent accepted-trial counts', async (t) => {
  const fixture = await createVerifiedRunFixture(t, ({ metadata }) => {
    metadata.acceptedTrialCount = 0;
  });
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /does not report every expected trial as completed and accepted/,
  );
});

test('run-directory analysis rejects invalid validation artifact counts', async (t) => {
  const fixture = await createVerifiedRunFixture(t, ({ metadata }) => {
    metadata.validationArtifactCount = 2;
  });
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /validation-artifacts\.json count is inconsistent/,
  );
});

test('candidate run-directory analysis requires stable clean source provenance', async (t) => {
  const fixture = await createVerifiedRunFixture(t, ({ metadata }) => {
    metadata.evidenceStatus = 'candidate';
    metadata.sourceProvenance.stable = false;
  });
  await assert.rejects(
    verifyRunDirectory(fixture.directory),
    /candidate evidence does not declare stable source provenance/,
  );
});

test('candidate run-directory analysis accepts matching clean source provenance', async (t) => {
  const fixture = await createVerifiedRunFixture(t, ({ metadata }) => {
    const provenance = {
      status: 'available',
      commit: 'a'.repeat(40),
      tree: 'b'.repeat(40),
      ref: 'experiment/verified-fixture',
      dirty: false,
      stagedChanges: 0,
      unstagedChanges: 0,
      untrackedFiles: 0,
      porcelainEntryCount: 0,
      porcelainByteCount: 0,
      porcelainSha256: '4'.repeat(64),
      trackedFileCount: 42,
      trackedFilesSha256: '5'.repeat(64),
      packageLockTracked: true,
      packageLockSha256: '6'.repeat(64),
    };
    metadata.evidenceStatus = 'candidate';
    metadata.sourceProvenance = {
      start: { ...provenance, capturedAt: '2026-08-29T20:00:00.000Z' },
      end: { ...provenance, capturedAt: '2026-08-29T20:00:02.000Z' },
      stable: true,
    };
  });
  const summary = await summarizeInput(fixture.directory);
  assert.equal(summary.artifactVerification.status, 'consistent');
  assert.equal(summary.artifactVerification.evidenceStatus, 'candidate');
  assert.equal(summary.artifactVerification.sourceProvenanceStable, true);
});

test('analysis CLI accepts both run directories and standalone CSV files', async (t) => {
  const fixture = await createVerifiedRunFixture(t);
  const [{ stdout: directoryStdout }, { stdout: csvStdout }] = await Promise.all([
    execFileAsync(process.execPath, [ANALYZER_PATH, fixture.directory], { encoding: 'utf8' }),
    execFileAsync(
      process.execPath,
      [ANALYZER_PATH, path.join(fixture.directory, 'frames.csv')],
      { encoding: 'utf8' },
    ),
  ]);
  assert.equal(JSON.parse(directoryStdout).artifactVerification.status, 'consistent');
  assert.equal(JSON.parse(csvStdout).artifactVerification.status, 'unverified');
});

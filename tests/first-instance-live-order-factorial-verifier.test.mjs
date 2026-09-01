import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { brotliCompressSync } from 'node:zlib';

import {
  classifyFirstInstanceLiveOrderManifestEnvelope,
  deriveFirstInstanceLiveOrderEnvironmentIdentity,
  readFirstInstanceLiveOrderFactorialDeclaredArtifacts,
  summarizeFirstInstanceLiveOrderTimingStartCacheStates,
  verifyRetainedFirstInstanceLiveOrderFactorialSummary,
  verifyRetainedFirstInstanceLiveOrderSmokeSummary,
} from '../analysis/verify-first-instance-live-order-factorial.mjs';
import {
  summarizeFirstInstanceLiveOrderFactorial,
} from '../analysis/first-instance-live-order-factorial-summary.mjs';
import {
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';

const RUN_ID = 'compact-order-factorial-verifier-fixture';

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function writeJsonArtifact(directory, relativePath, value) {
  const bytes = jsonBytes(value);
  const filename = path.join(directory, ...relativePath.split('/'));
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, bytes);
  return {
    path: relativePath,
    encoding: 'json-utf8',
    byteLength: bytes.length,
    sha256: sha256(bytes),
  };
}

async function writeBrotliArtifact(directory, relativePath, value) {
  const json = jsonBytes(value);
  const compressed = brotliCompressSync(json);
  const filename = path.join(directory, ...relativePath.split('/'));
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, compressed);
  return {
    path: relativePath,
    encoding: 'brotli-json-utf8',
    jsonByteLength: json.length,
    jsonSha256: sha256(json),
    brotliByteLength: compressed.length,
    brotliSha256: sha256(compressed),
  };
}

async function createCompactContainer(t, { smokeMode = false } = {}) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'order-factorial-verifier-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const plan = await writeJsonArtifact(directory, 'plan.json', { artifact: 'plan' });
  const forcedFeatureOffGate = await writeBrotliArtifact(
    directory,
    'forced-feature-off-gate.json.br',
    { artifact: 'gate' },
  );
  const executionIdentityStart = await writeJsonArtifact(
    directory,
    'execution-identity-start.json',
    { artifact: 'identity-start' },
  );
  const executionIdentityEnd = await writeJsonArtifact(
    directory,
    'execution-identity-end.json',
    { artifact: 'identity-end' },
  );
  const sessionCount = smokeMode ? 1 : 2;
  const trialCount = smokeMode ? 1 : 64;
  const sessions = await Promise.all(Array.from(
    { length: sessionCount },
    (_, sessionIndex) => writeJsonArtifact(
    directory,
    `session-${sessionIndex + 1}.json`,
    { artifact: 'session', sessionIndex },
    ),
  ));
  const trials = [];
  for (let index = 0; index < trialCount; index += 1) {
    const artifact = await writeBrotliArtifact(
      directory,
      `trials/trial-${String(index).padStart(2, '0')}.json.br`,
      { artifact: 'trial', index },
    );
    trials.push({
      factorialPlanIndex: index,
      factorialTrialId: `${RUN_ID}-trial-${index}`,
      sessionIndex: Math.floor(index / 32),
      sessionTrialIndex: index % 32,
      factorialCellId: `cell-${index % 16}`,
      compatibilityPlanIndex: index % 4,
      compatibilityRepetitionIndex: index % 4,
      strictSemanticSha256: 'a'.repeat(64),
      timingStartCacheState: {
        totalPipelineCacheEntries: 10,
        computePipelineCacheEntries: 2,
        computeProgramEntries: 1,
        rendererMemory: { programs: 10 },
      },
      artifact,
    });
  }
  const diagnosticSummary = await writeJsonArtifact(
    directory,
    'diagnostic-summary.json',
    { artifact: 'summary' },
  );
  const viteRuntimeAudit = await writeJsonArtifact(
    directory,
    'vite-runtime-audit.json',
    { artifact: 'vite-audit' },
  );
  const manifest = {
    artifacts: {
      plan,
      forcedFeatureOffGate,
      executionIdentityStart,
      executionIdentityEnd,
      sessions,
      trials,
      diagnosticSummary,
      viteRuntimeAudit,
    },
  };
  await writeFile(path.join(directory, 'manifest.json'), jsonBytes(manifest));
  return { directory, manifest };
}

function smokeManifestEnvelope(artifacts) {
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-smoke-manifest',
    scope: 'one-trial plumbing smoke; explicitly excluded from factorial diagnostic analysis',
    runId: `${RUN_ID}-smoke`,
    completedAt: new Date(0).toISOString(),
    browser: {
      executable: 'browser.exe',
      arguments: [
        '--enable-unsafe-webgpu',
        '--enable-webgpu-developer-features',
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
      ],
    },
    fixedWorkload: {
      objectCount: 65_536,
      bucketCount: 32,
      visibilityFraction: 0.99,
      layout: 'baseline',
      scenarioSeed: 0xb1ad_2026,
      warmupFrames: 320,
      measuredFrames: 480,
      setupPrimeTopology: 'staged-order-factorial-v1',
    },
    executionPolicy: {
      smokeMode: true,
      sessionCount: 1,
      trialsPerSession: 1,
      retryCount: 0,
      replacementAllowed: false,
      outlierRemovalAllowed: false,
      efficacyStoppingAllowed: false,
      artificialTimestampMapDelayMs: 0,
    },
    compatibilityMapping: 'candidate-high-visibility-c-orientation-v1',
    environmentIdentity: { performanceNowQuantumMs: 0.005 },
    workloadIdentity: {},
    executionIdentity: {},
    processLifecycle: {},
    artifacts,
    completion: {
      frozenTrialCount: 64,
      executedTrialCount: 1,
      persistedTrialCount: 1,
      freshShaderChallengeCount: 6,
      entryDocumentCount: 2,
      allBrowsersClosed: true,
      sessionOverlapDetected: false,
    },
  };
}

function createAnalysisRecords() {
  return buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID }).map((trial) => ({
    ...trial,
    trialSummary: {
      trialId: trial.trialId,
      visibilityFraction: trial.visibilityFraction,
      superblockOrientationOffset: trial.superblockOrientationOffset,
      nRows: 480,
      nBlocks: 60,
      estimates: {
        gpuPassTotal: {
          deltaMs: trial.factorialCellIndex - 8,
          deltaPercent: trial.factorialCellIndex - 8,
        },
        gpuRender: {
          deltaMs: trial.factorialCellIndex - 8,
          deltaPercent: trial.factorialCellIndex - 8,
        },
        gpuCompute: {
          deltaMs: trial.factorialCellIndex - 8,
          deltaPercent: trial.factorialCellIndex - 8,
        },
      },
    },
  }));
}

test('declared artifact reader pins exact traversal, hashes, Brotli decode, and inventory', async (t) => {
  const { directory, manifest } = await createCompactContainer(t);
  const values = await readFirstInstanceLiveOrderFactorialDeclaredArtifacts(
    directory,
    manifest,
  );
  assert.equal(values.size, 72);
  assert.deepEqual(values.get('plan.json'), { artifact: 'plan' });
  assert.deepEqual(values.get('forced-feature-off-gate.json.br'), { artifact: 'gate' });
  assert.deepEqual(values.get('trials/trial-63.json.br'), { artifact: 'trial', index: 63 });
});

test('declared artifact reader rejects extras, traversal, duplicate declarations, and byte drift', async (t) => {
  await t.test('extra file', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    await writeFile(path.join(directory, 'extra.json'), '{}');
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /file inventory differs/,
    );
  });
  await t.test('unsafe traversal', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    manifest.artifacts.plan.path = '../plan.json';
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /canonical safe relative path/,
    );
  });
  await t.test('duplicate declaration', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    manifest.artifacts.diagnosticSummary = { ...manifest.artifacts.plan };
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /declared twice/,
    );
  });
  await t.test('JSON bytes', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    await writeFile(path.join(directory, 'plan.json'), '{"changed":true}\n');
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /byteLength|byte limit|sha256/,
    );
  });
  await t.test('compressed bytes', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    const filename = path.join(directory, 'trials', 'trial-00.json.br');
    const bytes = await readFile(filename);
    bytes[0] ^= 0xff;
    await writeFile(filename, bytes);
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /brotliSha256/,
    );
  });
  await t.test('decoded JSON commitment', async (t) => {
    const { directory, manifest } = await createCompactContainer(t);
    manifest.artifacts.trials[0].artifact.jsonSha256 = 'b'.repeat(64);
    await assert.rejects(
      readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
      /jsonSha256/,
    );
  });
});

test('declared artifact reader rejects symlink substitution when supported', async (t) => {
  const { directory, manifest } = await createCompactContainer(t);
  const target = path.join(directory, 'plan-target.json');
  const declared = path.join(directory, 'plan.json');
  const bytes = await readFile(declared);
  await rm(declared);
  await writeFile(target, bytes);
  try {
    await symlink(target, declared, 'file');
  } catch (error) {
    if (['EPERM', 'EACCES', 'ENOTSUP'].includes(error?.code)) {
      t.skip(`symlinks unavailable: ${error.code}`);
      return;
    }
    throw error;
  }
  await assert.rejects(
    readFirstInstanceLiveOrderFactorialDeclaredArtifacts(directory, manifest),
    /symlink|file inventory differs/,
  );
});

test('retained diagnostic summary is independently recomputed from all 64 records', () => {
  const records = createAnalysisRecords();
  const retained = summarizeFirstInstanceLiveOrderFactorial(records, RUN_ID);
  assert.deepEqual(
    verifyRetainedFirstInstanceLiveOrderFactorialSummary(records, retained, RUN_ID),
    retained,
  );
  const changed = structuredClone(retained);
  changed.nRecords = 63;
  assert.throws(
    () => verifyRetainedFirstInstanceLiveOrderFactorialSummary(records, changed, RUN_ID),
    /retained diagnostic summary differs/,
  );
});

test('smoke envelope and one-trial declared container are valid but explicitly non-factorial', async (t) => {
  const { directory, manifest } = await createCompactContainer(t, { smokeMode: true });
  const smokeManifest = smokeManifestEnvelope(manifest.artifacts);
  assert.deepEqual(classifyFirstInstanceLiveOrderManifestEnvelope(smokeManifest), {
    smokeMode: true,
    sessionCount: 1,
    executedTrialCount: 1,
    trialsPerSession: 1,
    entryDocumentCount: 2,
  });
  await writeFile(path.join(directory, 'manifest.json'), jsonBytes(smokeManifest));
  const values = await readFirstInstanceLiveOrderFactorialDeclaredArtifacts(
    directory,
    smokeManifest,
  );
  assert.equal(values.size, 8);
});

test('manifest envelope rejects an absent, nonpositive, or coarse CPU timer quantum', () => {
  const base = smokeManifestEnvelope({});
  for (const invalidQuantum of [undefined, null, 0, -0.001, 0.010_001]) {
    const manifest = structuredClone(base);
    if (invalidQuantum === undefined) {
      delete manifest.environmentIdentity.performanceNowQuantumMs;
    } else {
      manifest.environmentIdentity.performanceNowQuantumMs = invalidQuantum;
    }
    assert.throws(
      () => classifyFirstInstanceLiveOrderManifestEnvelope(manifest),
      /invalid CPU timer quantum/,
    );
  }
});

test('environment identity binds the measured CPU timer quantum', () => {
  const base = deriveFirstInstanceLiveOrderEnvironmentIdentity({
    performanceNowQuantumMs: 0.005,
  });
  const changed = deriveFirstInstanceLiveOrderEnvironmentIdentity({
    performanceNowQuantumMs: 0.006,
  });
  assert.equal(base.performanceNowQuantumMs, 0.005);
  assert.notDeepEqual(base, changed);
});

test('retained smoke summary binds the executed cell and never invokes factorial analysis', () => {
  const analysisRecord = createAnalysisRecords().find((record) => (
    record.sessionIndex === 0
      && JSON.stringify(record.firstComputeUseOrder)
        !== JSON.stringify(record.renderPipelinePrimeOrder)
  ));
  const retained = {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-smoke-summary',
    decision: 'smoke-only-excluded-from-factorial-diagnostic-analysis',
    analysisInvoked: false,
    runId: RUN_ID,
    factorialPlanIndex: analysisRecord.planIndex,
    factorialCellId: analysisRecord.factorialCellId,
    trialSummary: analysisRecord.trialSummary,
  };
  assert.deepEqual(
    verifyRetainedFirstInstanceLiveOrderSmokeSummary(
      analysisRecord,
      retained,
      RUN_ID,
    ),
    retained,
  );
  const changed = structuredClone(retained);
  changed.analysisInvoked = true;
  assert.throws(
    () => verifyRetainedFirstInstanceLiveOrderSmokeSummary(
      analysisRecord,
      changed,
      RUN_ID,
    ),
    /retained smoke summary differs/,
  );
});

test('timing-start cache audit retains ordered per-session states and unique counts', () => {
  const executionPlan = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  const manifestTrials = executionPlan.map((trial) => ({
    factorialPlanIndex: trial.planIndex,
    factorialTrialId: trial.trialId,
    sessionIndex: trial.sessionIndex,
    sessionTrialIndex: trial.sessionTrialIndex,
    factorialCellId: trial.factorialCellId,
    timingStartCacheState: {
      totalPipelineCacheEntries: trial.sessionIndex === 0 ? 10 : 11,
      computePipelineCacheEntries: 2,
      computeProgramEntries: 1,
      rendererMemory: { programs: 10 },
    },
  }));
  manifestTrials[48].timingStartCacheState = {
    ...manifestTrials[48].timingStartCacheState,
    computeProgramEntries: 2,
  };
  const sessions = summarizeFirstInstanceLiveOrderTimingStartCacheStates(
    manifestTrials,
    executionPlan,
  );
  assert.equal(sessions.length, 2);
  assert.equal(sessions[0].trialCount, 32);
  assert.equal(sessions[0].uniqueStateCount, 1);
  assert.deepEqual(
    sessions[0].orderedStates.map(({ factorialPlanIndex }) => factorialPlanIndex),
    Array.from({ length: 32 }, (_, index) => index),
  );
  assert.equal(sessions[1].trialCount, 32);
  assert.equal(sessions[1].uniqueStateCount, 2);
  assert.deepEqual(
    sessions[1].uniqueStates.map(({ occurrenceCount }) => occurrenceCount),
    [31, 1],
  );
  assert.equal(Object.hasOwn(sessions[1], 'pass'), false);
});

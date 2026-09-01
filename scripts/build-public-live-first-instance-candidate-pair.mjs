import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  summarizeCsv,
  verifyRunDirectory,
} from '../analysis/summarize.mjs';
import {
  verifyCandidateSeries,
} from '../analysis/verify-live-first-instance-candidate-pair.mjs';
import {
  inspectPublicLiveCandidateRun,
  publicLivePairJsonBytes,
  publicLivePairSha256Bytes,
  verifyPublicLiveCandidatePairBundle,
} from '../analysis/verify-public-live-first-instance-candidate-pair.mjs';
import {
  CANDIDATE_LEDGER_FILENAME,
  canonicalJson,
  parseCandidateLedgerText,
} from './live-first-instance-candidate-ledger.mjs';
import {
  CANDIDATE_REGISTRY_FILENAME,
  candidateRegistryAnchor,
  parseCandidateRegistryText,
  verifyCandidateRegistryAnchorTag,
} from './live-first-instance-candidate-registry.mjs';
import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
  PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE,
  PUBLIC_LIVE_PAIR_LEDGER_NAME,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_POLICY_ID,
  PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
  PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE,
  PUBLIC_LIVE_PAIR_REGISTRY_NAME,
  PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  projectPrivatePairVerification,
  validatePublicLivePairBundleManifest,
  validatePublicLivePairReceipt,
} from './public-live-first-instance-pair-policy.mjs';
import { sanitizeLiveEvidence } from './sanitize-live-evidence.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function fail(message) {
  throw new Error(`Public live candidate-pair build failed: ${message}`);
}

function sameCanonical(left, right) {
  return canonicalJson(left) === canonicalJson(right);
}

function pathIsWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === ''
    || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

async function assertNewPath(filename, label) {
  try {
    await lstat(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') return;
    throw error;
  }
  fail(`${label} already exists.`);
}

async function readRegularFile(filename, label) {
  let stats;
  try {
    stats = await lstat(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') fail(`${label} is missing.`);
    throw error;
  }
  if (!stats.isFile() || stats.isSymbolicLink()) {
    fail(`${label} must be a non-symbolic-link regular file.`);
  }
  return readFile(filename);
}

async function snapshotRegularDirectory(directory, label) {
  const entries = await readdir(directory, { withFileTypes: true });
  const snapshot = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const filename = path.join(directory, entry.name);
    const stats = await lstat(filename);
    if (!entry.isFile() || entry.isSymbolicLink() || !stats.isFile()) {
      fail(`${label} contains a non-regular entry.`);
    }
    const bytes = await readFile(filename);
    snapshot.push({
      name: entry.name,
      bytes: bytes.length,
      sha256: publicLivePairSha256Bytes(bytes),
    });
  }
  return snapshot;
}

async function assertDirectorySnapshot(directory, expected, label) {
  const current = await snapshotRegularDirectory(directory, label);
  if (!sameCanonical(current, expected)) fail(`${label} changed during pair derivation.`);
}

async function assertByteIdenticalDirectories(expectedDirectory, actualDirectory, label) {
  const expected = await snapshotRegularDirectory(expectedDirectory, `${label} expected output`);
  const actual = await snapshotRegularDirectory(actualDirectory, `${label} supplied output`);
  if (!sameCanonical(actual, expected)) {
    fail(`${label} is not the deterministic sanitizer output of its selected private run.`);
  }
  for (const { name } of expected) {
    const expectedBytes = await readFile(path.join(expectedDirectory, name));
    const actualBytes = await readFile(path.join(actualDirectory, name));
    if (!actualBytes.equals(expectedBytes)) {
      fail(`${label} differs byte-for-byte from its deterministic sanitizer output.`);
    }
  }
}

function summaryFromVerification(verified) {
  return {
    ...summarizeCsv(verified.csvText),
    bundleIntegrity: verified.bundleIntegrity,
    liveFirstInstanceEvidenceDecision: verified.liveFirstInstanceEvidenceDecision,
    artifactVerification: verified.artifactVerification,
  };
}

function publicRunBinding(publicRun) {
  return {
    matrixOrdinal: publicRun.reservation.matrixOrdinal,
    attemptOrdinal: publicRun.reservation.attemptOrdinal,
    portableLabel: `matrix-${publicRun.reservation.matrixOrdinal}-public-run`,
    runId: publicRun.metadata.runId,
    reservationEventSha256: publicRun.reservation.reservationEventSha256,
    privateArtifactManifest: {
      name: 'artifact-manifest.json',
      sha256: publicRun.privateArtifactManifestSha256,
    },
    publicArtifactManifest: {
      name: 'artifact-manifest.json',
      bytes: publicRun.manifestBytes.length,
      sha256: publicRun.manifestSha256,
    },
    publicBundlePolicyId: publicRun.publicBundlePolicyId,
    candidateCommit: publicRun.candidateCommit,
    candidateTree: publicRun.candidateTree,
  };
}

function attemptForPublicRun(state, publicRun) {
  const reservation = publicRun.reservation;
  const attempt = state.attempts.find(
    ({ reservation: event }) => event.eventSha256 === reservation.reservationEventSha256,
  );
  if (attempt === undefined
    || attempt.finalization.classification !== 'valid-candidate'
    || attempt.reservation.attemptOrdinal !== reservation.attemptOrdinal
    || attempt.reservation.matrixOrdinal !== reservation.matrixOrdinal
    || attempt.reservation.seriesId !== reservation.seriesId) {
    fail(`public matrix ${reservation.matrixOrdinal} does not match a valid ledger reservation.`);
  }
  return attempt;
}

export async function buildPublicLiveCandidatePairBundle(
  privateSeriesDirectory,
  publicRunDirectories,
  publicOutputDirectory,
  {
    repositoryRoot = PROJECT_ROOT,
    privatePairVerifier = verifyCandidateSeries,
    inspectPublicRun = inspectPublicLiveCandidateRun,
    publicBundleVerifier = verifyPublicLiveCandidatePairBundle,
    anchorTagVerifier = verifyCandidateRegistryAnchorTag,
    publicRunDeriver = sanitizeLiveEvidence,
  } = {},
) {
  if (!Array.isArray(publicRunDirectories) || publicRunDirectories.length !== 2) {
    fail('exactly two sanitized public run directories are required.');
  }
  const seriesDirectory = path.resolve(privateSeriesDirectory);
  const outputDirectory = path.resolve(publicOutputDirectory);
  const publicDirectories = publicRunDirectories.map((directory) => path.resolve(directory));
  const seriesStats = await lstat(seriesDirectory);
  if (!seriesStats.isDirectory() || seriesStats.isSymbolicLink()) {
    fail('private series input is not a non-symbolic-link directory.');
  }
  for (const [index, directory] of publicDirectories.entries()) {
    const stats = await lstat(directory);
    if (!stats.isDirectory() || stats.isSymbolicLink()) {
      fail(`public run input ${index + 1} is not a non-symbolic-link directory.`);
    }
  }
  for (const inputDirectory of [seriesDirectory, ...publicDirectories]) {
    if (pathIsWithin(inputDirectory, outputDirectory)) {
      fail('public pair output must not be an input directory or its descendant.');
    }
  }
  await assertNewPath(outputDirectory, 'public pair output directory');
  const publicInputSnapshots = new Map();
  for (const [index, directory] of publicDirectories.entries()) {
    publicInputSnapshots.set(
      directory,
      await snapshotRegularDirectory(directory, `public run input ${index + 1}`),
    );
  }

  const ledgerPath = path.join(seriesDirectory, CANDIDATE_LEDGER_FILENAME);
  const ledgerBytes = await readRegularFile(ledgerPath, CANDIDATE_LEDGER_FILENAME);
  const ledgerText = ledgerBytes.toString('utf8');
  const ledgerState = parseCandidateLedgerText(ledgerText);
  if (ledgerState.pendingReservation !== null) fail('private series has a pending reservation.');
  const selectedPrivateRunStartSnapshots = new Map();
  for (const { finalization } of ledgerState.attempts) {
    if (finalization.classification !== 'valid-candidate') continue;
    const runDirectory = path.resolve(seriesDirectory, finalization.runDirectory);
    if (!pathIsWithin(seriesDirectory, runDirectory)) {
      fail('a selected private run directory escapes the candidate series.');
    }
    selectedPrivateRunStartSnapshots.set(
      runDirectory,
      await snapshotRegularDirectory(
        runDirectory,
        `private matrix ${finalization.matrixOrdinal} run`,
      ),
    );
  }
  const candidateSeriesRoot = path.dirname(seriesDirectory);
  const registryPath = path.join(candidateSeriesRoot, CANDIDATE_REGISTRY_FILENAME);
  const registryBytes = await readRegularFile(registryPath, CANDIDATE_REGISTRY_FILENAME);
  const registryState = parseCandidateRegistryText(registryBytes.toString('utf8'));

  const privateRunVerificationCache = new Map();
  const strictPrivateRunVerifier = async (runDirectory) => {
    const key = path.resolve(runDirectory);
    if (!privateRunVerificationCache.has(key)) {
      privateRunVerificationCache.set(
        key,
        verifyRunDirectory(key, { repositoryRoot }),
      );
    }
    return privateRunVerificationCache.get(key);
  };
  const strictPrivateSummarizer = async (runDirectory) => (
    summaryFromVerification(await strictPrivateRunVerifier(runDirectory))
  );
  const privatePair = await privatePairVerifier(seriesDirectory, {
    runVerifier: strictPrivateRunVerifier,
    summarizeRun: strictPrivateSummarizer,
    candidateSeriesRoot,
    repositoryRoot,
  });
  const privatePairProjection = projectPrivatePairVerification(privatePair);
  if (privatePairProjection.pairEligibility.pass !== true
    || privatePairProjection.validCandidateCount !== 2
    || privatePairProjection.completedMatrixCount !== 2) {
    fail('private series is not an eligible closed two-matrix candidate pair.');
  }
  if (privatePairProjection.seriesId !== ledgerState.seriesId
    || privatePairProjection.ledger.eventCount !== ledgerState.events.length
    || privatePairProjection.ledger.finalEventSha256
      !== ledgerState.events.at(-1).eventSha256) {
    fail('private pair verifier result differs from the private ledger bytes.');
  }
  const selectedRegistryRecord = registryState.byStudyKey.get(
    privatePairProjection.registry.studyKey,
  );
  if (selectedRegistryRecord === undefined
    || selectedRegistryRecord.materialization === null
    || selectedRegistryRecord.claim.seriesDirectory !== path.basename(seriesDirectory)
    || !sameCanonical(selectedRegistryRecord.claim.sourceIdentity, ledgerState.sourceIdentity)
    || selectedRegistryRecord.materialization.seriesId !== ledgerState.seriesId
    || selectedRegistryRecord.materialization.seriesOpeningEventSha256
      !== ledgerState.events[0].eventSha256) {
    fail('private pair verifier registry does not materialize the supplied series.');
  }
  const registryAnchor = candidateRegistryAnchor(selectedRegistryRecord);
  const registryBinding = {
    filename: PUBLIC_LIVE_PAIR_REGISTRY_NAME,
    bytes: registryBytes.length,
    sha256: publicLivePairSha256Bytes(registryBytes),
    eventCount: registryState.events.length,
    finalEventSha256: registryState.finalEventSha256,
    claimEventSha256: selectedRegistryRecord.claim.eventSha256,
    materializedEventSha256: selectedRegistryRecord.materialization.eventSha256,
    seriesOpeningEventSha256:
      selectedRegistryRecord.materialization.seriesOpeningEventSha256,
    studyKey: selectedRegistryRecord.claim.studyKey,
    seriesDirectory: selectedRegistryRecord.claim.seriesDirectory,
    anchorTagName: registryAnchor.tagName,
    anchorMessageSha256: registryAnchor.messageSha256,
  };
  if (!sameCanonical(privatePairProjection.registry, {
    experimentId: selectedRegistryRecord.claim.experimentId,
    studyKey: registryBinding.studyKey,
    filename: registryBinding.filename,
    eventCount: registryBinding.eventCount,
    finalEventSha256: registryBinding.finalEventSha256,
    claimEventSha256: registryBinding.claimEventSha256,
    materializedEventSha256: registryBinding.materializedEventSha256,
    seriesOpeningEventSha256: registryBinding.seriesOpeningEventSha256,
    anchorTagName: registryBinding.anchorTagName,
    anchorMessageSha256: registryBinding.anchorMessageSha256,
    anchorVerified: true,
  })) {
    fail('private pair verifier registry receipt differs from the disclosed registry bytes.');
  }

  const publicRuns = [];
  for (const directory of publicDirectories) {
    publicRuns.push(await inspectPublicRun(directory, { repositoryRoot }));
  }
  publicRuns.sort(
    (left, right) => left.reservation.matrixOrdinal - right.reservation.matrixOrdinal,
  );
  if (publicRuns[0].reservation.matrixOrdinal !== 1
    || publicRuns[1].reservation.matrixOrdinal !== 2) {
    fail('sanitized public runs must represent matrices 1 and 2.');
  }
  if (!sameCanonical(
    privatePairProjection.matrices,
    publicRuns.map(({ matrixDecision }) => matrixDecision),
  )) {
    fail('sanitized public run decisions differ from the private pair verifier result.');
  }
  const privateManifestSnapshots = [];
  for (const publicRun of publicRuns) {
    if (!publicDirectories.includes(path.resolve(publicRun.resolvedDirectory))) {
      fail(`public matrix ${publicRun.reservation.matrixOrdinal} inspection changed its input directory.`);
    }
    const attempt = attemptForPublicRun(ledgerState, publicRun);
    const privateRunDirectory = path.join(seriesDirectory, attempt.finalization.runDirectory);
    const privateDirectorySnapshot = selectedPrivateRunStartSnapshots.get(
      path.resolve(privateRunDirectory),
    );
    if (privateDirectorySnapshot === undefined) {
      fail(`private matrix ${publicRun.reservation.matrixOrdinal} was not snapshotted before verification.`);
    }
    const privateManifestPath = path.join(privateRunDirectory, 'artifact-manifest.json');
    const privateManifestBytes = await readRegularFile(
      privateManifestPath,
      `private matrix ${publicRun.reservation.matrixOrdinal} artifact-manifest.json`,
    );
    if (publicLivePairSha256Bytes(privateManifestBytes)
      !== publicRun.privateArtifactManifestSha256) {
      fail(
        `public matrix ${publicRun.reservation.matrixOrdinal} private-manifest commitment `
          + 'differs from the finalized private run.',
      );
    }
    privateManifestSnapshots.push({
      matrixOrdinal: publicRun.reservation.matrixOrdinal,
      directory: privateRunDirectory,
      directorySnapshot: privateDirectorySnapshot,
      filename: privateManifestPath,
      bytes: privateManifestBytes,
    });
  }

  const outputParent = path.dirname(outputDirectory);
  await mkdir(outputParent, { recursive: true });
  const derivationDirectory = await mkdtemp(path.join(
    outputParent,
    `.${path.basename(outputDirectory)}.sanitizer-check-`,
  ));
  try {
    for (const [index, publicRun] of publicRuns.entries()) {
      const privateSnapshot = privateManifestSnapshots[index];
      const derivedDirectory = path.join(
        derivationDirectory,
        `matrix-${publicRun.reservation.matrixOrdinal}`,
      );
      await publicRunDeriver(privateSnapshot.directory, derivedDirectory, {
        repositoryRoot,
      });
      await assertByteIdenticalDirectories(
        derivedDirectory,
        publicRun.resolvedDirectory,
        `public matrix ${publicRun.reservation.matrixOrdinal}`,
      );
    }
  } finally {
    await rm(derivationDirectory, { recursive: true, force: true });
  }

  const ledgerBinding = {
    filename: PUBLIC_LIVE_PAIR_LEDGER_NAME,
    bytes: ledgerBytes.length,
    sha256: publicLivePairSha256Bytes(ledgerBytes),
    eventCount: ledgerState.events.length,
    finalEventSha256: ledgerState.events.at(-1).eventSha256,
  };
  const receipt = validatePublicLivePairReceipt({
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-receipt',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    bundleLabel: PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
    seriesId: ledgerState.seriesId,
    integrityScope: PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE,
    privateAttemptScope: PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE,
    authenticityVerified: false,
    privateAttemptArtifactBytesDisclosed: false,
    ledgerBinding,
    privatePairVerification: privatePairProjection,
    publicRuns: publicRuns.map(publicRunBinding),
    seriesRootRegistryBinding: registryBinding,
  });
  const receiptBytes = publicLivePairJsonBytes(receipt);
  const bundleContents = new Map([
    [PUBLIC_LIVE_PAIR_REGISTRY_NAME, registryBytes],
    [PUBLIC_LIVE_PAIR_LEDGER_NAME, ledgerBytes],
    [PUBLIC_LIVE_PAIR_RECEIPT_NAME, receiptBytes],
  ]);
  const manifest = validatePublicLivePairBundleManifest({
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-bundle-manifest',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    bundleLabel: PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
    hashAlgorithm: 'sha256',
    files: PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(({ name, role }) => {
      const bytes = bundleContents.get(name);
      return {
        name,
        role,
        bytes: bytes.length,
        sha256: publicLivePairSha256Bytes(bytes),
      };
    }),
  });
  const manifestBytes = publicLivePairJsonBytes(manifest);

  await assertNewPath(outputDirectory, 'public pair output directory');
  const stagingDirectory = await mkdtemp(path.join(
    outputParent,
    `.${path.basename(outputDirectory)}.staging-`,
  ));
  let committed = false;
  try {
    for (const [name, bytes] of bundleContents) {
      await writeFile(path.join(stagingDirectory, name), bytes, { flag: 'wx' });
    }
    await writeFile(
      path.join(stagingDirectory, PUBLIC_LIVE_PAIR_MANIFEST_NAME),
      manifestBytes,
      { flag: 'wx' },
    );
    await publicBundleVerifier(stagingDirectory, publicDirectories, {
      repositoryRoot,
      inspectPublicRun,
      anchorTagVerifier,
    });
    const privatePairEnd = projectPrivatePairVerification(await privatePairVerifier(
      seriesDirectory,
      {
        runVerifier: strictPrivateRunVerifier,
        summarizeRun: strictPrivateSummarizer,
        candidateSeriesRoot,
        repositoryRoot,
      },
    ));
    if (!sameCanonical(privatePairEnd, privatePairProjection)
      || !(await readRegularFile(ledgerPath, CANDIDATE_LEDGER_FILENAME)).equals(ledgerBytes)
      || !(await readRegularFile(registryPath, CANDIDATE_REGISTRY_FILENAME)).equals(registryBytes)) {
      fail('private registry, ledger, or pair verification changed during receipt derivation.');
    }
    for (const snapshot of privateManifestSnapshots) {
      const current = await readRegularFile(
        snapshot.filename,
        `private matrix ${snapshot.matrixOrdinal} artifact-manifest.json`,
      );
      if (!current.equals(snapshot.bytes)) {
        fail(`private matrix ${snapshot.matrixOrdinal} manifest changed during receipt derivation.`);
      }
      await assertDirectorySnapshot(
        snapshot.directory,
        snapshot.directorySnapshot,
        `private matrix ${snapshot.matrixOrdinal} run`,
      );
    }
    for (const [index, directory] of publicDirectories.entries()) {
      await assertDirectorySnapshot(
        directory,
        publicInputSnapshots.get(directory),
        `public run input ${index + 1}`,
      );
    }
    await assertNewPath(outputDirectory, 'public pair output directory');
    await rename(stagingDirectory, outputDirectory);
    committed = true;
  } finally {
    if (!committed) await rm(stagingDirectory, { recursive: true, force: true });
  }

  return Object.freeze({
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-build-result',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    bundleLabel: PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
    seriesId: ledgerState.seriesId,
    deterministicSanitizerRederivationVerified: true,
    decision: structuredClone(privatePairProjection.decision),
    bundleManifest: {
      name: PUBLIC_LIVE_PAIR_MANIFEST_NAME,
      bytes: manifestBytes.length,
      sha256: publicLivePairSha256Bytes(manifestBytes),
    },
    ledger: ledgerBinding,
    registry: registryBinding,
    receipt: {
      name: PUBLIC_LIVE_PAIR_RECEIPT_NAME,
      bytes: receiptBytes.length,
      sha256: publicLivePairSha256Bytes(receiptBytes),
    },
    publicRuns: receipt.publicRuns.map(({ matrixOrdinal, portableLabel, runId, publicArtifactManifest }) => ({
      matrixOrdinal,
      portableLabel,
      runId,
      publicArtifactManifestSha256: publicArtifactManifest.sha256,
    })),
  });
}

export function serializePublicLiveCandidatePairBuildResult(result) {
  return `${JSON.stringify(result, null, 2)}\n`;
}

async function main() {
  const [privateSeries, firstPublicRun, secondPublicRun, publicOutput, ...extra] =
    process.argv.slice(2);
  if (!privateSeries || !firstPublicRun || !secondPublicRun || !publicOutput
    || extra.length > 0) {
    process.stderr.write(
      'Usage: node scripts/build-public-live-first-instance-candidate-pair.mjs '
        + '<private-series> <public-run-1> <public-run-2> <new-public-pair-output>\n',
    );
    process.exitCode = 2;
    return;
  }
  const result = await buildPublicLiveCandidatePairBundle(
    privateSeries,
    [firstPublicRun, secondPublicRun],
    publicOutput,
  );
  process.stdout.write(serializePublicLiveCandidatePairBuildResult(result));
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}

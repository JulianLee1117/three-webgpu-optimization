export const PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION = 1;
export const PUBLIC_LIVE_PAIR_POLICY_ID =
  'first-instance-live-public-candidate-pair-v1';
export const PUBLIC_LIVE_PAIR_LEDGER_NAME = 'candidate-attempts.jsonl';
export const PUBLIC_LIVE_PAIR_REGISTRY_NAME = 'candidate-series-registry.jsonl';
export const PUBLIC_LIVE_PAIR_RECEIPT_NAME = 'public-candidate-pair-receipt.json';
export const PUBLIC_LIVE_PAIR_MANIFEST_NAME = 'public-candidate-pair-manifest.json';
export const PUBLIC_LIVE_PAIR_BUNDLE_LABEL = 'public-candidate-pair-receipt-bundle';

export const PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE =
  'The receipt binds the disclosed candidate ledger, the private pair-verifier decision, and the two independently verified public-derived run manifests.';
export const PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE =
  'Private attempt-content digests remain commitments because the private attempt bytes are not included in this public receipt bundle.';

export const PUBLIC_LIVE_PAIR_BUNDLE_FILES = Object.freeze([
  Object.freeze({
    name: PUBLIC_LIVE_PAIR_REGISTRY_NAME,
    role: 'disclosed hash-chained candidate root registry',
  }),
  Object.freeze({
    name: PUBLIC_LIVE_PAIR_LEDGER_NAME,
    role: 'disclosed hash-chained candidate selection ledger',
  }),
  Object.freeze({
    name: PUBLIC_LIVE_PAIR_RECEIPT_NAME,
    role: 'public candidate-pair verification receipt',
  }),
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GIT_OBJECT_PATTERN = /^[0-9a-f]{40,64}$/;
const SAFE_RELATIVE_DIRECTORY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/;
const LOCAL_PATH_PATTERNS = Object.freeze([
  /(?:^|["'\s])(?:[a-zA-Z]:[\\/])[^\r\n]*/,
  /(?:^|["'\s])(?:\/home\/|\/Users\/)[^/\s]+(?:\/[^\s]*)?/,
  /file:\/\/\/(?:[a-zA-Z]:\/|home\/|Users\/)/i,
  /^\\\\[^\\/]+[\\/][^\\/]+/,
]);
const FORBIDDEN_PRIVATE_KEYS = new Set([
  'gpuUuid',
  'pid',
  'processName',
  'hostname',
  'hostName',
  'username',
  'userName',
  'homeDir',
  'homedir',
  'cwd',
  'userDataDir',
  'profilePath',
  'absolutePath',
  'localPath',
]);

function reject(message) {
  throw new Error(`Public live candidate-pair policy rejected: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expected, label) {
  if (!isRecord(value)) reject(`${label} must be an object.`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    reject(`${label} has an unexpected schema.`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    reject(`${label} must be a non-empty string.`);
  }
  return value;
}

function safeInteger(value, label, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    reject(`${label} must be an integer no smaller than ${minimum}.`);
  }
  return value;
}

function sha256(value, label) {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    reject(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function gitObject(value, label) {
  if (typeof value !== 'string' || !GIT_OBJECT_PATTERN.test(value)) {
    reject(`${label} must be a Git object ID.`);
  }
  return value;
}

function boolean(value, label) {
  if (typeof value !== 'boolean') reject(`${label} must be boolean.`);
  return value;
}

function nullableSha256(value, label) {
  if (value !== null) sha256(value, label);
  return value;
}

function safeRelativeDirectory(value, label) {
  nonemptyString(value, label);
  if (!SAFE_RELATIVE_DIRECTORY_PATTERN.test(value)
    || value.startsWith('/')
    || value.includes('\\')
    || value.split('/').some((part) => part === '' || part === '.' || part === '..')) {
    reject(`${label} must be a portable relative directory.`);
  }
  return value;
}

function validateSourceIdentity(value, label) {
  const identity = exactKeys(value, [
    'commit',
    'tree',
    'trackedFilesSha256',
    'packageLockSha256',
    'executionDependencyClosureSha256',
    'executionDependencyFileCount',
    'executionDependencyTotalBytes',
  ], label);
  gitObject(identity.commit, `${label}.commit`);
  gitObject(identity.tree, `${label}.tree`);
  sha256(identity.trackedFilesSha256, `${label}.trackedFilesSha256`);
  sha256(identity.packageLockSha256, `${label}.packageLockSha256`);
  sha256(
    identity.executionDependencyClosureSha256,
    `${label}.executionDependencyClosureSha256`,
  );
  safeInteger(identity.executionDependencyFileCount, `${label}.executionDependencyFileCount`);
  safeInteger(identity.executionDependencyTotalBytes, `${label}.executionDependencyTotalBytes`);
  return identity;
}

function validateClassificationEvidence(value, label) {
  const evidence = exactKeys(value, [
    'artifactManifestStatus',
    'artifactInfrastructureFailureObserved',
    'attemptLayoutExact',
    'browserOrDeviceLossObserved',
    'candidateReservationBindingStatus',
    'childLifecycleClosed',
    'runVerificationStatus',
    'runDirectoryCount',
    'sourceIdentityStable',
    'telemetryCollectorFailureObserved',
  ], label);
  nonemptyString(evidence.artifactManifestStatus, `${label}.artifactManifestStatus`);
  nonemptyString(
    evidence.candidateReservationBindingStatus,
    `${label}.candidateReservationBindingStatus`,
  );
  nonemptyString(evidence.runVerificationStatus, `${label}.runVerificationStatus`);
  safeInteger(evidence.runDirectoryCount, `${label}.runDirectoryCount`);
  for (const key of [
    'artifactInfrastructureFailureObserved',
    'attemptLayoutExact',
    'browserOrDeviceLossObserved',
    'childLifecycleClosed',
    'sourceIdentityStable',
    'telemetryCollectorFailureObserved',
  ]) {
    boolean(evidence[key], `${label}.${key}`);
  }
  return evidence;
}

function validatePairEligibility(value, label) {
  const eligibility = exactKeys(value, [
    'canonicalRegistryAndAnchorBound',
    'exactlyTwoValidCompletedMatrices',
    'secondMatrixPresentRegardlessOfFirst',
    'sameFrozenIdentity',
    'everyCandidateBoundToSeriesSource',
    'noLaterSubstitutes',
    'sourceSeriesFreeOfNonretryableFailure',
    'pass',
  ], label);
  for (const [key, entry] of Object.entries(eligibility)) {
    boolean(entry, `${label}.${key}`);
  }
  return eligibility;
}

function validatePrivatePairProjection(value) {
  const pair = exactKeys(value, [
    'schemaVersion',
    'kind',
    'seriesId',
    'ledger',
    'registry',
    'attemptCount',
    'attempts',
    'retryCount',
    'completedMatrixCount',
    'validCandidateCount',
    'nonretryableFailureCount',
    'pairIdentitySha256',
    'pairEligibility',
    'matrices',
    'decision',
  ], 'receipt.privatePairVerification');
  if (pair.schemaVersion !== 1
    || pair.kind !== 'first-instance-live-candidate-pair-verification') {
    reject('receipt.privatePairVerification has an unsupported identity.');
  }
  nonemptyString(pair.seriesId, 'receipt.privatePairVerification.seriesId');
  const ledger = exactKeys(pair.ledger, [
    'filename', 'eventCount', 'finalEventSha256', 'sourceIdentity',
  ], 'receipt.privatePairVerification.ledger');
  if (ledger.filename !== PUBLIC_LIVE_PAIR_LEDGER_NAME) {
    reject('receipt.privatePairVerification.ledger filename is unsupported.');
  }
  safeInteger(ledger.eventCount, 'receipt.privatePairVerification.ledger.eventCount', 1);
  sha256(
    ledger.finalEventSha256,
    'receipt.privatePairVerification.ledger.finalEventSha256',
  );
  validateSourceIdentity(
    ledger.sourceIdentity,
    'receipt.privatePairVerification.ledger.sourceIdentity',
  );
  const registry = exactKeys(pair.registry, [
    'experimentId',
    'studyKey',
    'filename',
    'eventCount',
    'finalEventSha256',
    'claimEventSha256',
    'materializedEventSha256',
    'seriesOpeningEventSha256',
    'anchorTagName',
    'anchorMessageSha256',
    'anchorVerified',
  ], 'receipt.privatePairVerification.registry');
  nonemptyString(registry.experimentId, 'receipt.privatePairVerification.registry.experimentId');
  sha256(registry.studyKey, 'receipt.privatePairVerification.registry.studyKey');
  if (registry.filename !== PUBLIC_LIVE_PAIR_REGISTRY_NAME) {
    reject('receipt.privatePairVerification.registry filename is unsupported.');
  }
  safeInteger(registry.eventCount, 'receipt.privatePairVerification.registry.eventCount', 1);
  for (const field of [
    'finalEventSha256',
    'claimEventSha256',
    'materializedEventSha256',
    'seriesOpeningEventSha256',
    'anchorMessageSha256',
  ]) {
    sha256(registry[field], `receipt.privatePairVerification.registry.${field}`);
  }
  nonemptyString(registry.anchorTagName, 'receipt.privatePairVerification.registry.anchorTagName');
  if (registry.anchorVerified !== true) {
    reject('receipt.privatePairVerification registry anchor is not verified.');
  }
  for (const field of [
    'attemptCount',
    'retryCount',
    'completedMatrixCount',
    'validCandidateCount',
    'nonretryableFailureCount',
  ]) {
    safeInteger(pair[field], `receipt.privatePairVerification.${field}`);
  }
  nullableSha256(
    pair.pairIdentitySha256,
    'receipt.privatePairVerification.pairIdentitySha256',
  );
  if (!Array.isArray(pair.attempts)) {
    reject('receipt.privatePairVerification.attempts must be an array.');
  }
  pair.attempts.forEach((attempt, index) => {
    const label = `receipt.privatePairVerification.attempts[${index}]`;
    exactKeys(attempt, [
      'attemptOrdinal',
      'matrixOrdinal',
      'attemptDirectory',
      'classification',
      'reasonCode',
      'classificationEvidence',
    ], label);
    safeInteger(attempt.attemptOrdinal, `${label}.attemptOrdinal`, 1);
    safeInteger(attempt.matrixOrdinal, `${label}.matrixOrdinal`, 1);
    safeRelativeDirectory(attempt.attemptDirectory, `${label}.attemptDirectory`);
    nonemptyString(attempt.classification, `${label}.classification`);
    nonemptyString(attempt.reasonCode, `${label}.reasonCode`);
    validateClassificationEvidence(attempt.classificationEvidence, `${label}.classificationEvidence`);
  });
  validatePairEligibility(
    pair.pairEligibility,
    'receipt.privatePairVerification.pairEligibility',
  );
  if (!Array.isArray(pair.matrices)) {
    reject('receipt.privatePairVerification.matrices must be an array.');
  }
  pair.matrices.forEach((matrix, index) => {
    const label = `receipt.privatePairVerification.matrices[${index}]`;
    exactKeys(matrix, [
      'matrixOrdinal',
      'attemptOrdinal',
      'runId',
      'numericalPass',
      'environmentPass',
      'evidencePass',
      'pass',
      'numericalDecision',
      'environmentDecision',
      'overallEvidenceDecision',
    ], label);
    safeInteger(matrix.matrixOrdinal, `${label}.matrixOrdinal`, 1);
    safeInteger(matrix.attemptOrdinal, `${label}.attemptOrdinal`, 1);
    nonemptyString(matrix.runId, `${label}.runId`);
    for (const key of ['numericalPass', 'environmentPass', 'evidencePass', 'pass']) {
      boolean(matrix[key], `${label}.${key}`);
    }
    if (!isRecord(matrix.numericalDecision)
      || !isRecord(matrix.environmentDecision)
      || !isRecord(matrix.overallEvidenceDecision)) {
      reject(`${label} decision records must be objects.`);
    }
  });
  const decision = exactKeys(
    pair.decision,
    ['status', 'pass', 'rule'],
    'receipt.privatePairVerification.decision',
  );
  nonemptyString(decision.status, 'receipt.privatePairVerification.decision.status');
  boolean(decision.pass, 'receipt.privatePairVerification.decision.pass');
  nonemptyString(decision.rule, 'receipt.privatePairVerification.decision.rule');
  return pair;
}

function auditPublicValue(value, path = '$') {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => auditPublicValue(entry, `${path}[${index}]`));
    return;
  }
  if (isRecord(value)) {
    for (const [key, child] of Object.entries(value)) {
      if (FORBIDDEN_PRIVATE_KEYS.has(key)) {
        reject(`${path}.${key} is a forbidden private-identifier field.`);
      }
      auditPublicValue(child, `${path}.${key}`);
    }
    return;
  }
  if (typeof value === 'string'
    && LOCAL_PATH_PATTERNS.some((pattern) => pattern.test(value))) {
    reject(`${path} contains a machine-local path.`);
  }
}

export function projectPrivatePairVerification(pair) {
  const projected = {
    schemaVersion: pair.schemaVersion,
    kind: pair.kind,
    seriesId: pair.seriesId,
    ledger: {
      filename: pair.ledger?.filename,
      eventCount: pair.ledger?.eventCount,
      finalEventSha256: pair.ledger?.finalEventSha256,
      sourceIdentity: structuredClone(pair.ledger?.sourceIdentity),
    },
    registry: {
      experimentId: pair.registry?.experimentId,
      studyKey: pair.registry?.studyKey,
      filename: pair.registry?.filename,
      eventCount: pair.registry?.eventCount,
      finalEventSha256: pair.registry?.finalEventSha256,
      claimEventSha256: pair.registry?.claimEventSha256,
      materializedEventSha256: pair.registry?.materializedEventSha256,
      seriesOpeningEventSha256: pair.registry?.seriesOpeningEventSha256,
      anchorTagName: pair.registry?.anchorTagName,
      anchorMessageSha256: pair.registry?.anchorMessageSha256,
      anchorVerified: pair.registry?.anchorVerified,
    },
    attemptCount: pair.attemptCount,
    attempts: Array.isArray(pair.attempts) ? pair.attempts.map((attempt) => ({
      attemptOrdinal: attempt.attemptOrdinal,
      matrixOrdinal: attempt.matrixOrdinal,
      attemptDirectory: attempt.attemptDirectory,
      classification: attempt.classification,
      reasonCode: attempt.reasonCode,
      classificationEvidence: structuredClone(attempt.classificationEvidence),
    })) : pair.attempts,
    retryCount: pair.retryCount,
    completedMatrixCount: pair.completedMatrixCount,
    validCandidateCount: pair.validCandidateCount,
    nonretryableFailureCount: pair.nonretryableFailureCount,
    pairIdentitySha256: pair.pairIdentitySha256,
    pairEligibility: structuredClone(pair.pairEligibility),
    matrices: Array.isArray(pair.matrices) ? pair.matrices.map((matrix) => ({
      matrixOrdinal: matrix.matrixOrdinal,
      attemptOrdinal: matrix.attemptOrdinal,
      runId: matrix.runId,
      numericalPass: matrix.numericalPass,
      environmentPass: matrix.environmentPass,
      evidencePass: matrix.evidencePass,
      pass: matrix.pass,
      numericalDecision: structuredClone(matrix.numericalDecision),
      environmentDecision: structuredClone(matrix.environmentDecision),
      overallEvidenceDecision: structuredClone(matrix.overallEvidenceDecision),
    })) : pair.matrices,
    decision: {
      status: pair.decision?.status,
      pass: pair.decision?.pass,
      rule: pair.decision?.rule,
    },
  };
  validatePrivatePairProjection(projected);
  auditPublicValue(projected);
  return projected;
}

export function validatePublicLivePairReceipt(value) {
  const receipt = exactKeys(value, [
    'schemaVersion',
    'kind',
    'policyId',
    'bundleLabel',
    'seriesId',
    'integrityScope',
    'privateAttemptScope',
    'authenticityVerified',
    'privateAttemptArtifactBytesDisclosed',
    'ledgerBinding',
    'privatePairVerification',
    'publicRuns',
    'seriesRootRegistryBinding',
  ], 'receipt');
  if (receipt.schemaVersion !== PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION
    || receipt.kind !== 'first-instance-live-public-candidate-pair-receipt'
    || receipt.policyId !== PUBLIC_LIVE_PAIR_POLICY_ID
    || receipt.bundleLabel !== PUBLIC_LIVE_PAIR_BUNDLE_LABEL) {
    reject('receipt identity differs from the frozen public-pair policy.');
  }
  nonemptyString(receipt.seriesId, 'receipt.seriesId');
  if (receipt.integrityScope !== PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE
    || receipt.privateAttemptScope !== PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE) {
    reject('receipt disclosure or integrity scope differs from the frozen policy.');
  }
  if (receipt.authenticityVerified !== false
    || receipt.privateAttemptArtifactBytesDisclosed !== false) {
    reject('receipt overstates authenticity or private-byte disclosure.');
  }
  const registryBinding = exactKeys(receipt.seriesRootRegistryBinding, [
    'filename',
    'bytes',
    'sha256',
    'eventCount',
    'finalEventSha256',
    'claimEventSha256',
    'materializedEventSha256',
    'seriesOpeningEventSha256',
    'studyKey',
    'seriesDirectory',
    'anchorTagName',
    'anchorMessageSha256',
  ], 'receipt.seriesRootRegistryBinding');
  if (registryBinding.filename !== PUBLIC_LIVE_PAIR_REGISTRY_NAME) {
    reject('receipt.seriesRootRegistryBinding filename is unsupported.');
  }
  safeInteger(registryBinding.bytes, 'receipt.seriesRootRegistryBinding.bytes', 1);
  safeInteger(registryBinding.eventCount, 'receipt.seriesRootRegistryBinding.eventCount', 1);
  for (const field of [
    'sha256',
    'finalEventSha256',
    'claimEventSha256',
    'materializedEventSha256',
    'seriesOpeningEventSha256',
    'studyKey',
    'anchorMessageSha256',
  ]) {
    sha256(registryBinding[field], `receipt.seriesRootRegistryBinding.${field}`);
  }
  safeRelativeDirectory(
    registryBinding.seriesDirectory,
    'receipt.seriesRootRegistryBinding.seriesDirectory',
  );
  nonemptyString(
    registryBinding.anchorTagName,
    'receipt.seriesRootRegistryBinding.anchorTagName',
  );
  const ledger = exactKeys(receipt.ledgerBinding, [
    'filename', 'bytes', 'sha256', 'eventCount', 'finalEventSha256',
  ], 'receipt.ledgerBinding');
  if (ledger.filename !== PUBLIC_LIVE_PAIR_LEDGER_NAME) {
    reject('receipt.ledgerBinding filename is unsupported.');
  }
  safeInteger(ledger.bytes, 'receipt.ledgerBinding.bytes', 1);
  sha256(ledger.sha256, 'receipt.ledgerBinding.sha256');
  safeInteger(ledger.eventCount, 'receipt.ledgerBinding.eventCount', 1);
  sha256(ledger.finalEventSha256, 'receipt.ledgerBinding.finalEventSha256');
  validatePrivatePairProjection(receipt.privatePairVerification);
  if (!Array.isArray(receipt.publicRuns) || receipt.publicRuns.length !== 2) {
    reject('receipt.publicRuns must contain exactly two entries.');
  }
  receipt.publicRuns.forEach((run, index) => {
    const label = `receipt.publicRuns[${index}]`;
    exactKeys(run, [
      'matrixOrdinal',
      'attemptOrdinal',
      'portableLabel',
      'runId',
      'reservationEventSha256',
      'privateArtifactManifest',
      'publicArtifactManifest',
      'publicBundlePolicyId',
      'candidateCommit',
      'candidateTree',
    ], label);
    safeInteger(run.matrixOrdinal, `${label}.matrixOrdinal`, 1);
    safeInteger(run.attemptOrdinal, `${label}.attemptOrdinal`, 1);
    if (run.portableLabel !== `matrix-${run.matrixOrdinal}-public-run`) {
      reject(`${label}.portableLabel is not deterministic.`);
    }
    nonemptyString(run.runId, `${label}.runId`);
    sha256(run.reservationEventSha256, `${label}.reservationEventSha256`);
    const privateManifest = exactKeys(
      run.privateArtifactManifest,
      ['name', 'sha256'],
      `${label}.privateArtifactManifest`,
    );
    if (privateManifest.name !== 'artifact-manifest.json') {
      reject(`${label}.privateArtifactManifest name is unsupported.`);
    }
    sha256(privateManifest.sha256, `${label}.privateArtifactManifest.sha256`);
    const publicManifest = exactKeys(
      run.publicArtifactManifest,
      ['name', 'bytes', 'sha256'],
      `${label}.publicArtifactManifest`,
    );
    if (publicManifest.name !== 'artifact-manifest.json') {
      reject(`${label}.publicArtifactManifest name is unsupported.`);
    }
    safeInteger(publicManifest.bytes, `${label}.publicArtifactManifest.bytes`, 1);
    sha256(publicManifest.sha256, `${label}.publicArtifactManifest.sha256`);
    nonemptyString(run.publicBundlePolicyId, `${label}.publicBundlePolicyId`);
    gitObject(run.candidateCommit, `${label}.candidateCommit`);
    gitObject(run.candidateTree, `${label}.candidateTree`);
  });
  const ordinals = receipt.publicRuns.map(({ matrixOrdinal }) => matrixOrdinal);
  if (JSON.stringify(ordinals) !== JSON.stringify([1, 2])) {
    reject('receipt.publicRuns must be sorted as matrices 1 and 2.');
  }
  auditPublicValue(receipt);
  return receipt;
}

export function validatePublicLivePairBundleManifest(value) {
  const manifest = exactKeys(value, [
    'schemaVersion',
    'kind',
    'policyId',
    'bundleLabel',
    'hashAlgorithm',
    'files',
  ], 'public pair bundle manifest');
  if (manifest.schemaVersion !== PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION
    || manifest.kind !== 'first-instance-live-public-candidate-pair-bundle-manifest'
    || manifest.policyId !== PUBLIC_LIVE_PAIR_POLICY_ID
    || manifest.bundleLabel !== PUBLIC_LIVE_PAIR_BUNDLE_LABEL
    || manifest.hashAlgorithm !== 'sha256') {
    reject('public pair bundle manifest identity differs from the frozen policy.');
  }
  if (!Array.isArray(manifest.files)
    || manifest.files.length !== PUBLIC_LIVE_PAIR_BUNDLE_FILES.length) {
    reject('public pair bundle manifest has the wrong file count.');
  }
  manifest.files.forEach((entry, index) => {
    const label = `public pair bundle manifest files[${index}]`;
    exactKeys(entry, ['name', 'role', 'bytes', 'sha256'], label);
    const expected = PUBLIC_LIVE_PAIR_BUNDLE_FILES[index];
    if (entry.name !== expected.name || entry.role !== expected.role) {
      reject(`${label} differs from the frozen file declaration.`);
    }
    safeInteger(entry.bytes, `${label}.bytes`, 1);
    sha256(entry.sha256, `${label}.sha256`);
  });
  auditPublicValue(manifest);
  return manifest;
}

export function auditPublicLivePairValue(value) {
  auditPublicValue(value);
}

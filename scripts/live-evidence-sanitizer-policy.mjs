import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export const LIVE_PUBLIC_EVIDENCE_POLICY_SCHEMA_VERSION = 2;
export const LIVE_PUBLIC_EVIDENCE_POLICY_ID =
  'first-instance-live-public-evidence-v2';
export const LIVE_PRIVATE_BUNDLE_LABEL = 'private-original';
export const LIVE_PUBLIC_BUNDLE_LABEL = 'public-derived';

export const LIVE_PRIVATE_MANIFEST_NAME = 'artifact-manifest.json';
export const LIVE_REQUIRED_ARTIFACTS = Object.freeze([
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
  'forced-feature-off-evidence.json',
]);
export const LIVE_OPTIONAL_ARTIFACTS = Object.freeze(['gpu-telemetry.csv']);

export const GPU_UUID_REPLACEMENT =
  'GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}';
export const PID_REPLACEMENT = '100000 + one-based-process-ordinal';
export const PROCESS_NAME_REPLACEMENT =
  'resident-gpu-process-{one-based-process-ordinal}.redacted';
export const ENVIRONMENT_NOTE_REPLACEMENT =
  '[redacted: unrelated private environment note]';

export const LIVE_PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST = Object.freeze([
  Object.freeze({
    file: 'metadata.json',
    path: '$.environment.note',
    replacement: ENVIRONMENT_NOTE_REPLACEMENT,
  }),
  ...[
    'metadata.json:$.environment.gpuTelemetry',
    'gpu-telemetry-summary.json:$',
  ].flatMap((prefix) => [
    `${prefix}.summary.gpus[*].gpuUuid`,
    `${prefix}.computeProcesses.pre.processes[*].gpuUuid`,
    `${prefix}.computeProcesses.pre.processes[*].pid`,
    `${prefix}.computeProcesses.pre.processes[*].processName`,
    `${prefix}.computeProcesses.post.processes[*].gpuUuid`,
    `${prefix}.computeProcesses.post.processes[*].pid`,
    `${prefix}.computeProcesses.post.processes[*].processName`,
  ]).map((qualifiedPath) => {
    const separator = qualifiedPath.indexOf(':');
    const file = qualifiedPath.slice(0, separator);
    const jsonPath = qualifiedPath.slice(separator + 1);
    const field = jsonPath.split('.').at(-1);
    return Object.freeze({
      file,
      path: jsonPath,
      replacement: field === 'gpuUuid'
        ? GPU_UUID_REPLACEMENT
        : field === 'pid'
          ? PID_REPLACEMENT
          : PROCESS_NAME_REPLACEMENT,
    });
  }),
  ...[
    'metadata.json:$.environment.gpuTelemetry.coverageAudit.gpuIdentities[*].gpuUuid',
    'gpu-telemetry-summary.json:$.coverageAudit.gpuIdentities[*].gpuUuid',
  ].map((qualifiedPath) => {
    const separator = qualifiedPath.indexOf(':');
    return Object.freeze({
      file: qualifiedPath.slice(0, separator),
      path: qualifiedPath.slice(separator + 1),
      replacement: GPU_UUID_REPLACEMENT,
    });
  }),
  ...['pre', 'post'].flatMap((phase) => ['gpuUuid', 'pid', 'processName'].map((field) => (
    Object.freeze({
      file: 'metadata.json',
      path: `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.${phase}[*].${field}`,
      replacement: field === 'gpuUuid'
        ? GPU_UUID_REPLACEMENT
        : field === 'pid'
          ? PID_REPLACEMENT
          : PROCESS_NAME_REPLACEMENT,
    })
  ))),
]);

export const LIVE_PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST = Object.freeze([
  Object.freeze({
    file: 'gpu-telemetry.csv',
    column: 'gpuUuid',
    replacement: GPU_UUID_REPLACEMENT,
  }),
]);

// These are the complete local JavaScript inputs that can change the derived
// bytes. Keeping the list frozen makes the implementation commitment explicit.
export const LIVE_SANITIZER_IMPLEMENTATION_PATHS = Object.freeze([
  'scripts/live-evidence-sanitizer-policy.mjs',
  'scripts/nvidia-telemetry.mjs',
  'scripts/sanitize-live-evidence.mjs',
  'scripts/source-provenance.mjs',
]);

export const LIVE_PUBLIC_INTEGRITY_BYTE_SCOPE =
  'The public manifest binds only the present derived artifact bytes named in its files array.';
export const LIVE_PUBLIC_PRIVATE_BYTE_SCOPE =
  'The private manifest digest is a commitment to the unchanged private-original manifest; the public bundle does not contain or verify the omitted private artifact bytes.';

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expected) {
  return isRecord(value)
    && sameJson(Object.keys(value).sort(), [...expected].sort());
}

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GIT_OBJECT_PATTERN = /^[0-9a-f]{40,64}$/;

async function runGit(projectRoot, args, encoding = 'utf8') {
  const { stdout } = await execFileAsync('git', args, {
    cwd: projectRoot,
    encoding,
    maxBuffer: 256 * 1024 * 1024,
    windowsHide: true,
  });
  if (encoding === 'buffer') {
    return Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout);
  }
  return stdout;
}

/**
 * Resolve the frozen sanitizer closure from immutable Git objects. This is
 * deliberately independent of the caller's checked-out branch and working
 * files, so an older public bundle remains verifiable while its commit is
 * present in the repository object database.
 */
export async function resolveLiveSanitizerImplementationAtCommit(
  projectRoot,
  candidateSource,
) {
  if (!GIT_OBJECT_PATTERN.test(candidateSource?.commit ?? '')
    || !GIT_OBJECT_PATTERN.test(candidateSource?.tree ?? '')) {
    throw new Error('candidate source lacks valid Git commit and tree object IDs');
  }
  const commit = candidateSource.commit;
  const commitObject = (await runGit(
    projectRoot,
    ['rev-parse', '--verify', `${commit}^{commit}`],
  )).trim();
  if (commitObject !== commit) {
    throw new Error('candidate commit does not resolve to the recorded commit object');
  }
  const tree = (await runGit(
    projectRoot,
    ['rev-parse', '--verify', `${commit}^{tree}`],
  )).trim();
  if (tree !== candidateSource.tree) {
    throw new Error('candidate commit tree differs from the recorded candidate tree');
  }

  const files = [];
  for (const repositoryPath of LIVE_SANITIZER_IMPLEMENTATION_PATHS) {
    const contents = await runGit(
      projectRoot,
      ['cat-file', 'blob', `${commit}:${repositoryPath}`],
      'buffer',
    );
    files.push(Object.freeze({
      repositoryPath,
      sha256: createHash('sha256').update(contents).digest('hex'),
    }));
  }
  return Object.freeze({
    candidateCommit: commit,
    candidateTree: tree,
    files: Object.freeze(files),
  });
}

/**
 * Validate only the frozen public provenance schema and return all reasons.
 * Artifact bytes, local implementation hashes, and metadata links are checked
 * by the directory verifier, which has the required filesystem context.
 */
export function validateLivePublicBundleProvenanceShape(value) {
  const reasons = [];
  const require = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  require(exactKeys(value, [
    'schemaVersion',
    'policyId',
    'bundleLabel',
    'sourceBundleLabel',
    'candidateSource',
    'privateArtifactManifest',
    'sanitizer',
    'redactionPolicy',
    'integrityByteScope',
    'privateByteScope',
  ]), 'bundleProvenance has an unexpected schema');
  if (!isRecord(value)) return reasons;
  require(
    value.schemaVersion === LIVE_PUBLIC_EVIDENCE_POLICY_SCHEMA_VERSION,
    'bundleProvenance schemaVersion differs from the frozen policy',
  );
  require(
    value.policyId === LIVE_PUBLIC_EVIDENCE_POLICY_ID,
    'bundleProvenance policyId differs from the frozen policy',
  );
  require(value.bundleLabel === LIVE_PUBLIC_BUNDLE_LABEL,
    'bundleProvenance bundleLabel is not public-derived');
  require(value.sourceBundleLabel === LIVE_PRIVATE_BUNDLE_LABEL,
    'bundleProvenance sourceBundleLabel is not private-original');

  require(exactKeys(value.candidateSource, [
    'commit', 'tree', 'clean', 'stable', 'trackedFilesSha256', 'packageLockSha256',
  ]), 'bundleProvenance candidateSource has an unexpected schema');
  if (isRecord(value.candidateSource)) {
    require(GIT_OBJECT_PATTERN.test(value.candidateSource.commit ?? ''),
      'bundleProvenance candidateSource.commit is invalid');
    require(GIT_OBJECT_PATTERN.test(value.candidateSource.tree ?? ''),
      'bundleProvenance candidateSource.tree is invalid');
    require(value.candidateSource.clean === true,
      'bundleProvenance candidateSource is not clean');
    require(value.candidateSource.stable === true,
      'bundleProvenance candidateSource is not stable');
    require(SHA256_PATTERN.test(value.candidateSource.trackedFilesSha256 ?? ''),
      'bundleProvenance candidateSource.trackedFilesSha256 is invalid');
    require(SHA256_PATTERN.test(value.candidateSource.packageLockSha256 ?? ''),
      'bundleProvenance candidateSource.packageLockSha256 is invalid');
  }

  require(exactKeys(value.privateArtifactManifest, ['name', 'sha256']),
    'bundleProvenance privateArtifactManifest has an unexpected schema');
  if (isRecord(value.privateArtifactManifest)) {
    require(value.privateArtifactManifest.name === LIVE_PRIVATE_MANIFEST_NAME,
      'bundleProvenance private manifest name is invalid');
    require(SHA256_PATTERN.test(value.privateArtifactManifest.sha256 ?? ''),
      'bundleProvenance private manifest SHA-256 is invalid');
  }

  require(exactKeys(value.sanitizer, [
    'implementationKind', 'candidateCommit', 'candidateTree', 'files',
  ]), 'bundleProvenance sanitizer has an unexpected schema');
  if (isRecord(value.sanitizer)) {
    require(value.sanitizer.implementationKind === 'exact-source-file-closure-v1',
      'bundleProvenance sanitizer implementation kind is invalid');
    require(value.sanitizer.candidateCommit === value.candidateSource?.commit,
      'bundleProvenance sanitizer candidate commit differs from candidateSource');
    require(value.sanitizer.candidateTree === value.candidateSource?.tree,
      'bundleProvenance sanitizer candidate tree differs from candidateSource');
    require(Array.isArray(value.sanitizer.files)
      && value.sanitizer.files.length === LIVE_SANITIZER_IMPLEMENTATION_PATHS.length,
    'bundleProvenance sanitizer files do not cover the frozen implementation closure');
    if (Array.isArray(value.sanitizer.files)) {
      value.sanitizer.files.forEach((file, index) => {
        require(exactKeys(file, ['repositoryPath', 'sha256']),
          `bundleProvenance sanitizer.files[${index}] has an unexpected schema`);
        require(file?.repositoryPath === LIVE_SANITIZER_IMPLEMENTATION_PATHS[index],
          `bundleProvenance sanitizer.files[${index}] has the wrong repository path`);
        require(SHA256_PATTERN.test(file?.sha256 ?? ''),
          `bundleProvenance sanitizer.files[${index}] has an invalid SHA-256`);
      });
    }
  }

  require(exactKeys(value.redactionPolicy, [
    'jsonPathAllowlist', 'csvColumnAllowlist', 'unlistedFieldChangesPermitted',
  ]), 'bundleProvenance redactionPolicy has an unexpected schema');
  if (isRecord(value.redactionPolicy)) {
    require(sameJson(
      value.redactionPolicy.jsonPathAllowlist,
      LIVE_PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST,
    ), 'bundleProvenance JSON redaction allowlist differs from the frozen policy');
    require(sameJson(
      value.redactionPolicy.csvColumnAllowlist,
      LIVE_PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST,
    ), 'bundleProvenance CSV redaction allowlist differs from the frozen policy');
    require(value.redactionPolicy.unlistedFieldChangesPermitted === false,
      'bundleProvenance permits unlisted field changes');
  }
  require(value.integrityByteScope === LIVE_PUBLIC_INTEGRITY_BYTE_SCOPE,
    'bundleProvenance public byte scope differs from the frozen policy');
  require(value.privateByteScope === LIVE_PUBLIC_PRIVATE_BYTE_SCOPE,
    'bundleProvenance private byte scope differs from the frozen policy');
  return reasons;
}

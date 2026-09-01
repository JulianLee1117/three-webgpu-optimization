import { createHash } from 'node:crypto';

export const LIVE_FIRST_INSTANCE_ADAPTER_TELEMETRY_ASSOCIATION_KIND =
  'first-instance-live-adapter-telemetry-association';
export const LIVE_FIRST_INSTANCE_ADAPTER_TELEMETRY_ASSOCIATION_POLICY_ID =
  'nvidia-single-telemetry-gpu-exact-ascii-name-v2';

const NORMALIZATION_POLICY =
  'ascii-htab-printable-trim-collapse-space-ascii-lowercase-exact-v1';

function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

/**
 * Normalize a GPU identity field without locale-sensitive or fuzzy matching.
 * Invalid, empty, control-character, or non-ASCII input returns null.
 */
export function normalizeLiveFirstInstanceGpuIdentityText(value) {
  if (typeof value !== 'string' || !/^[\x09\x20-\x7e]+$/.test(value)) return null;
  const trimmed = value.replace(/^[ \t]+|[ \t]+$/g, '');
  if (trimmed.length === 0) return null;
  return trimmed
    .replace(/[ \t]+/g, ' ')
    .replace(/[A-Z]/g, (letter) => letter.toLowerCase());
}

export function evaluateLiveFirstInstanceAdapterTelemetryAssociation({
  adapterInfo,
  telemetryReport,
}) {
  const normalizedAdapterVendor = normalizeLiveFirstInstanceGpuIdentityText(
    adapterInfo?.vendor,
  );
  const normalizedAdapterDescription = normalizeLiveFirstInstanceGpuIdentityText(
    adapterInfo?.description,
  );
  const telemetryGpus = telemetryReport?.summary?.gpus;
  const telemetryGpuCount = Array.isArray(telemetryGpus) ? telemetryGpus.length : null;
  const soleTelemetryGpu = telemetryGpuCount === 1 ? telemetryGpus[0] : null;
  const coverageGpus = telemetryReport?.coverageAudit?.gpuIdentities;
  const coverageGpuCount = Array.isArray(coverageGpus) ? coverageGpus.length : null;
  const soleCoverageGpu = coverageGpuCount === 1 ? coverageGpus[0] : null;
  const telemetryGpuUuidValid = typeof soleTelemetryGpu?.gpuUuid === 'string'
    && soleTelemetryGpu.gpuUuid.trim() !== '';
  const coverageGpuIdentityValid = Number.isSafeInteger(soleCoverageGpu?.gpuIndex)
    && soleCoverageGpu.gpuIndex >= 0
    && typeof soleCoverageGpu?.gpuName === 'string'
    && soleCoverageGpu.gpuName.trim() !== ''
    && typeof soleCoverageGpu?.gpuUuid === 'string'
    && soleCoverageGpu.gpuUuid.trim() !== '';
  const coverageSummaryIdentityExact = telemetryGpuCount === 1
    && coverageGpuCount === 1
    && telemetryGpuUuidValid
    && coverageGpuIdentityValid
    && soleCoverageGpu?.gpuIndex === soleTelemetryGpu?.gpuIndex
    && soleCoverageGpu?.gpuName === soleTelemetryGpu?.gpuName
    && soleCoverageGpu?.gpuUuid === soleTelemetryGpu?.gpuUuid;
  const telemetryGpuIndex = Number.isSafeInteger(soleTelemetryGpu?.gpuIndex)
    && soleTelemetryGpu.gpuIndex >= 0
    ? soleTelemetryGpu.gpuIndex
    : null;
  const normalizedTelemetryGpuName = normalizeLiveFirstInstanceGpuIdentityText(
    soleTelemetryGpu?.gpuName,
  );

  const failures = [];
  if (normalizedAdapterVendor === null) {
    failures.push({
      code: 'adapter-vendor-invalid',
      reason:
        'page adapter vendor must be non-empty text containing only ASCII HTAB or printable characters',
    });
  } else if (normalizedAdapterVendor !== 'nvidia') {
    failures.push({
      code: 'adapter-vendor-not-nvidia',
      reason: 'page adapter vendor does not normalize exactly to "nvidia"',
    });
  }
  if (normalizedAdapterDescription === null) {
    failures.push({
      code: 'adapter-description-invalid',
      reason:
        'page adapter description must be non-empty text containing only ASCII HTAB or printable characters',
    });
  }
  if (telemetryGpuCount !== 1) {
    failures.push({
      code: 'telemetry-gpu-count-not-one',
      reason: 'telemetry summary must contain exactly one GPU',
    });
  } else {
    if (telemetryGpuIndex === null) {
      failures.push({
        code: 'telemetry-gpu-index-invalid',
        reason: 'sole telemetry GPU index is not a non-negative safe integer',
      });
    }
    if (normalizedTelemetryGpuName === null) {
      failures.push({
        code: 'telemetry-gpu-name-invalid',
        reason:
          'sole telemetry GPU name must be non-empty text containing only ASCII HTAB or printable characters',
      });
    } else if (normalizedAdapterDescription !== null
      && normalizedAdapterDescription !== normalizedTelemetryGpuName) {
      failures.push({
        code: 'adapter-telemetry-name-mismatch',
        reason: 'normalized page adapter description differs from the sole telemetry GPU name',
      });
    }
    if (!telemetryGpuUuidValid) {
      failures.push({
        code: 'telemetry-gpu-uuid-invalid',
        reason: 'sole telemetry GPU UUID must be non-empty text',
      });
    }
  }
  if (coverageGpuCount !== 1) {
    failures.push({
      code: 'telemetry-coverage-gpu-count-not-one',
      reason: 'telemetry coverage must contain exactly one GPU identity',
    });
  } else if (!coverageGpuIdentityValid) {
    failures.push({
      code: 'telemetry-coverage-gpu-identity-invalid',
      reason: 'sole telemetry coverage GPU identity tuple is invalid',
    });
  } else if (telemetryGpuCount === 1
    && telemetryGpuUuidValid
    && !coverageSummaryIdentityExact) {
    failures.push({
      code: 'telemetry-summary-coverage-identity-mismatch',
      reason: 'sole telemetry summary and coverage GPU identity tuples differ',
    });
  }

  const adapterIdentitySha256 = sha256Json({
    normalizedVendor: normalizedAdapterVendor,
    normalizedDescription: normalizedAdapterDescription,
  });
  const telemetryIdentitySha256 = sha256Json({
    gpuCount: telemetryGpuCount,
    gpuIndex: telemetryGpuIndex,
    normalizedGpuName: normalizedTelemetryGpuName,
    coverageGpuCount,
    coverageSummaryIdentityExact,
  });
  const associationSha256 = sha256Json({
    policyId: LIVE_FIRST_INSTANCE_ADAPTER_TELEMETRY_ASSOCIATION_POLICY_ID,
    adapterIdentitySha256,
    telemetryIdentitySha256,
  });

  return {
    schemaVersion: 1,
    kind: LIVE_FIRST_INSTANCE_ADAPTER_TELEMETRY_ASSOCIATION_KIND,
    policyId: LIVE_FIRST_INSTANCE_ADAPTER_TELEMETRY_ASSOCIATION_POLICY_ID,
    normalizationPolicy: NORMALIZATION_POLICY,
    pass: failures.length === 0,
    normalizedAdapterVendor,
    normalizedAdapterDescription,
    telemetryGpuCount,
    telemetryGpuIndex,
    normalizedTelemetryGpuName,
    adapterIdentitySha256,
    telemetryIdentitySha256,
    associationSha256,
    failureCodes: failures.map(({ code }) => code),
    reasons: failures.map(({ reason }) => reason),
  };
}

import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from './live-first-instance-adapter-telemetry-association.mjs';

const PROCESS_SET_MISMATCH_REASON =
  'pre-run and post-run compute-process identity sets differ';

const CANDIDATE_GATE_KIND = 'first-instance-live-candidate-environment-gate';
const OVERALL_DECISION_KIND = 'first-instance-live-overall-evidence-decision';

function unique(values) {
  return [...new Set(values)];
}

function processComparisonIsCoherentMismatch(comparison) {
  return comparison?.pass === false
    && Array.isArray(comparison.pre)
    && Array.isArray(comparison.post)
    && JSON.stringify(comparison.pre) !== JSON.stringify(comparison.post)
    && Array.isArray(comparison.reasons)
    && comparison.reasons.length === 1
    && comparison.reasons[0] === PROCESS_SET_MISMATCH_REASON;
}

function telemetryCollectorFailures(report, comparison) {
  const failures = [];
  if (report?.status !== 'available') {
    failures.push({
      code: 'telemetry-status-unavailable',
      reason: `telemetry status is ${JSON.stringify(report?.status)}; expected "available"`,
    });
  }
  if (report?.sampling?.malformedLineCount !== 0) {
    failures.push({
      code: 'telemetry-malformed-records',
      reason: `telemetry malformed-line count is ${
        JSON.stringify(report?.sampling?.malformedLineCount)
      }; expected zero`,
    });
  }
  if (report?.sampling?.stderrByteCount !== 0) {
    failures.push({
      code: 'telemetry-stderr-output',
      reason: `telemetry stderr byte count is ${
        JSON.stringify(report?.sampling?.stderrByteCount)
      }; expected zero`,
    });
  }
  if (!Number.isSafeInteger(report?.summary?.sampleCount)
    || report.summary.sampleCount <= 0) {
    failures.push({
      code: 'telemetry-no-samples',
      reason: 'telemetry summary contains no valid samples',
    });
  }
  if (report?.coverageAudit?.pass !== true) {
    const details = Array.isArray(report?.coverageAudit?.reasons)
      && report.coverageAudit.reasons.length > 0
      ? `: ${report.coverageAudit.reasons.join('; ')}`
      : '';
    failures.push({
      code: 'telemetry-coverage-invalid',
      reason: `telemetry collector coverage audit did not pass${details}`,
    });
  }
  if (comparison?.pass !== true && !processComparisonIsCoherentMismatch(comparison)) {
    failures.push({
      code: 'compute-process-snapshot-invalid',
      reason: Array.isArray(comparison?.reasons) && comparison.reasons.length > 0
        ? `compute-process snapshot evidence is invalid: ${comparison.reasons.join('; ')}`
        : 'compute-process snapshot evidence is unavailable or invalid',
    });
  }
  return failures;
}

export function evaluateLiveFirstInstanceCandidateEnvironmentGate({
  evidenceStatus,
  telemetryReport,
  computeProcessIdentityComparison,
  adapterTelemetryAssociation,
}) {
  if (evidenceStatus !== 'candidate') {
    return {
      schemaVersion: 2,
      kind: CANDIDATE_GATE_KIND,
      applicable: false,
      status: 'not-applicable-development',
      pass: null,
      retryable: false,
      nonReplaceable: false,
      collectorPass: false,
      processIdentityPass: null,
      adapterTelemetryAssociationPass: null,
      failureCodes: [],
      reasons: [],
    };
  }

  const collectorFailures = telemetryCollectorFailures(
    telemetryReport,
    computeProcessIdentityComparison,
  );
  if (collectorFailures.length > 0) {
    return {
      schemaVersion: 2,
      kind: CANDIDATE_GATE_KIND,
      applicable: true,
      status: 'failed-retryable-collector',
      pass: false,
      retryable: true,
      nonReplaceable: false,
      collectorPass: false,
      processIdentityPass: null,
      adapterTelemetryAssociationPass: null,
      failureCodes: unique(collectorFailures.map(({ code }) => code)),
      reasons: unique(collectorFailures.map(({ reason }) => reason)),
    };
  }

  if (computeProcessIdentityComparison.pass !== true) {
    return {
      schemaVersion: 2,
      kind: CANDIDATE_GATE_KIND,
      applicable: true,
      status: 'failed-non-replaceable-process-set-mismatch',
      pass: false,
      retryable: false,
      nonReplaceable: true,
      collectorPass: true,
      processIdentityPass: false,
      adapterTelemetryAssociationPass: adapterTelemetryAssociation?.pass === true,
      failureCodes: ['compute-process-set-mismatch'],
      reasons: [PROCESS_SET_MISMATCH_REASON],
    };
  }

  if (adapterTelemetryAssociation?.pass !== true) {
    const associationFailureCodes = Array.isArray(adapterTelemetryAssociation?.failureCodes)
      && adapterTelemetryAssociation.failureCodes.length > 0
      ? adapterTelemetryAssociation.failureCodes
      : ['adapter-telemetry-association-invalid'];
    const associationReasons = Array.isArray(adapterTelemetryAssociation?.reasons)
      && adapterTelemetryAssociation.reasons.length > 0
      ? adapterTelemetryAssociation.reasons
      : ['page adapter and telemetry GPU association is unavailable or invalid'];
    return {
      schemaVersion: 2,
      kind: CANDIDATE_GATE_KIND,
      applicable: true,
      status: 'failed-non-replaceable-adapter-telemetry-association',
      pass: false,
      retryable: false,
      nonReplaceable: true,
      collectorPass: true,
      processIdentityPass: true,
      adapterTelemetryAssociationPass: false,
      failureCodes: unique(associationFailureCodes),
      reasons: unique(associationReasons),
    };
  }

  return {
    schemaVersion: 2,
    kind: CANDIDATE_GATE_KIND,
    applicable: true,
    status: 'passed',
    pass: true,
    retryable: false,
    nonReplaceable: false,
    collectorPass: true,
    processIdentityPass: true,
    adapterTelemetryAssociationPass: true,
    failureCodes: [],
    reasons: [],
  };
}

export function evaluateLiveFirstInstanceOverallEvidenceDecision({
  evidenceStatus,
  candidateEnvironmentGate,
  preregisteredNumericalDecision,
}) {
  if (evidenceStatus !== 'candidate') {
    return {
      schemaVersion: 1,
      kind: OVERALL_DECISION_KIND,
      applicable: false,
      status: 'not-applicable-development',
      pass: null,
      retryable: false,
      nonReplaceable: false,
      failedGates: [],
      failureCodes: [],
      reasons: [],
    };
  }

  const failedGates = [];
  const failureCodes = [...(candidateEnvironmentGate?.failureCodes ?? [])];
  const reasons = [...(candidateEnvironmentGate?.reasons ?? [])];
  if (candidateEnvironmentGate?.pass !== true) {
    failedGates.push('candidateEnvironmentGate');
  }
  if (preregisteredNumericalDecision?.status !== 'evaluated') {
    failedGates.push('preregisteredNumericalDecision');
    failureCodes.push('preregistered-numerical-decision-unavailable');
    reasons.push('preregistered numerical decision is unavailable');
  } else if (preregisteredNumericalDecision.pass !== true) {
    failedGates.push('preregisteredNumericalDecision');
    failureCodes.push('preregistered-numerical-decision-failed');
    const numericalFailures = Array.isArray(preregisteredNumericalDecision.failedGates)
      ? preregisteredNumericalDecision.failedGates.join(', ')
      : 'unreported gates';
    reasons.push(`preregistered numerical decision failed: ${numericalFailures}`);
  }

  const retryable = candidateEnvironmentGate?.retryable === true;
  const pass = failedGates.length === 0;
  return {
    schemaVersion: 1,
    kind: OVERALL_DECISION_KIND,
    applicable: true,
    status: pass
      ? 'passed'
      : retryable
        ? 'failed-retryable-collector'
        : 'failed-non-replaceable',
    pass,
    retryable,
    nonReplaceable: !pass && !retryable,
    failedGates: unique(failedGates),
    failureCodes: unique(failureCodes),
    reasons: unique(reasons),
  };
}

export function liveFirstInstanceCandidateEnvironmentRequiresRunFailure(gate) {
  return gate?.applicable === true && gate.retryable === true;
}

export function createLiveFirstInstanceEnvironmentAudit({
  evidenceStatus,
  telemetryReport,
  computeProcessIdentityComparison,
  preregisteredNumericalDecision,
  adapterInfo,
}) {
  const adapterTelemetryAssociation =
    evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo,
      telemetryReport,
    });
  const candidateEnvironmentGate = evaluateLiveFirstInstanceCandidateEnvironmentGate({
    evidenceStatus,
    telemetryReport,
    computeProcessIdentityComparison,
    adapterTelemetryAssociation,
  });
  const overallEvidenceDecision = evaluateLiveFirstInstanceOverallEvidenceDecision({
    evidenceStatus,
    candidateEnvironmentGate,
    preregisteredNumericalDecision,
  });
  return {
    schemaVersion: 4,
    kind: 'first-instance-live-crossover-environment-audit',
    telemetryStatus: telemetryReport?.status ?? null,
    telemetryMalformedLineCount: telemetryReport?.sampling?.malformedLineCount ?? null,
    telemetryStderrByteCount: telemetryReport?.sampling?.stderrByteCount ?? null,
    telemetrySampleCount: telemetryReport?.summary?.sampleCount ?? null,
    telemetryCoveragePass: telemetryReport?.coverageAudit?.pass ?? null,
    adapterTelemetryAssociation,
    computeProcessIdentityComparison,
    candidateEnvironmentGate,
    overallEvidenceDecision,
  };
}

export const LIVE_FIRST_INSTANCE_PROCESS_SET_MISMATCH_REASON = PROCESS_SET_MISMATCH_REASON;

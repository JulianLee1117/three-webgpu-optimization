import assert from 'node:assert/strict';
import test from 'node:test';
import {
  validateLiveFirstInstanceEnvironmentAudit,
} from '../analysis/summarize.mjs';
import {
  createLiveFirstInstanceEnvironmentAudit,
  evaluateLiveFirstInstanceCandidateEnvironmentGate,
  evaluateLiveFirstInstanceOverallEvidenceDecision,
  liveFirstInstanceCandidateEnvironmentRequiresRunFailure,
} from '../scripts/live-first-instance-environment-audit.mjs';
import {
  compareComputeProcessIdentitySets,
} from '../scripts/nvidia-telemetry.mjs';

const NUMERICAL_PASS = Object.freeze({
  status: 'evaluated',
  pass: true,
  failedGates: [],
});

const BASE_ADAPTER_INFO = Object.freeze({
  vendor: 'nvidia',
  description: 'NVIDIA GeForce RTX 5070 Ti',
});

function availableTelemetryReport() {
  return {
    status: 'available',
    sampling: {
      malformedLineCount: 0,
      stderrByteCount: 0,
    },
    summary: {
      sampleCount: 12,
      gpus: [{
        gpuIndex: 0,
        gpuName: 'NVIDIA GeForce RTX 5070 Ti',
      }],
    },
    coverageAudit: {
      pass: true,
      reasons: [],
    },
  };
}

function snapshot(processes) {
  return {
    status: 'available',
    rawNonemptyLineCount: processes.length,
    parsedRecordCount: processes.length,
    malformedLineCount: 0,
    stdoutByteCount: processes.length === 0 ? 0 : 1,
    stdoutTruncated: false,
    stderrByteCount: 0,
    processes,
  };
}

function candidateMetadata(audit, adapterInfo = BASE_ADAPTER_INFO) {
  return {
    evidenceStatus: 'candidate',
    environment: {
      benchmarkPage: {
        adapterInfo: structuredClone(adapterInfo),
      },
    },
    liveFirstInstanceAnalysisAudit: {
      preregisteredNumericalDecision: NUMERICAL_PASS,
    },
    liveFirstInstanceEnvironmentAudit: audit,
  };
}

const BASE_PROCESSES = Object.freeze([
  Object.freeze({
    gpuUuid: 'GPU-a',
    pid: 10,
    processName: 'resident.exe',
    usedMemoryMiB: 100,
  }),
]);

test('posthoc reconstruction reports a passed candidate environment and overall decision', () => {
  const telemetryReport = availableTelemetryReport();
  telemetryReport.computeProcesses = {
    pre: snapshot(BASE_PROCESSES),
    post: snapshot(BASE_PROCESSES),
  };
  const comparison = compareComputeProcessIdentitySets(
    telemetryReport.computeProcesses.pre,
    telemetryReport.computeProcesses.post,
  );
  const input = {
    evidenceStatus: 'candidate',
    telemetryReport,
    adapterInfo: BASE_ADAPTER_INFO,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision: NUMERICAL_PASS,
  };
  const audit = createLiveFirstInstanceEnvironmentAudit(input);
  assert.deepEqual(createLiveFirstInstanceEnvironmentAudit(structuredClone(input)), audit);
  assert.deepEqual(validateLiveFirstInstanceEnvironmentAudit(
    telemetryReport,
    candidateMetadata(audit),
  ), {
    adapterTelemetryAssociation: audit.adapterTelemetryAssociation,
    candidateEnvironmentGate: audit.candidateEnvironmentGate,
    overallEvidenceDecision: audit.overallEvidenceDecision,
  });
  assert.equal(audit.schemaVersion, 4);
  assert.equal(audit.telemetryCoveragePass, true);
  assert.equal(audit.adapterTelemetryAssociation.pass, true);
  assert.equal(audit.candidateEnvironmentGate.status, 'passed');
  assert.equal(audit.candidateEnvironmentGate.schemaVersion, 2);
  assert.equal(audit.candidateEnvironmentGate.pass, true);
  assert.equal(audit.candidateEnvironmentGate.adapterTelemetryAssociationPass, true);
  assert.equal(audit.overallEvidenceDecision.status, 'passed');
  assert.equal(audit.overallEvidenceDecision.pass, true);
});

test('posthoc reconstruction keeps a coherent process mismatch as a completed non-replaceable failure', () => {
  const telemetryReport = availableTelemetryReport();
  telemetryReport.computeProcesses = {
    pre: snapshot(BASE_PROCESSES),
    post: snapshot([...BASE_PROCESSES, {
      gpuUuid: 'GPU-a',
      pid: 11,
      processName: 'late-process.exe',
      usedMemoryMiB: 1,
    }]),
  };
  const comparison = compareComputeProcessIdentitySets(
    telemetryReport.computeProcesses.pre,
    telemetryReport.computeProcesses.post,
  );
  const audit = createLiveFirstInstanceEnvironmentAudit({
    evidenceStatus: 'candidate',
    telemetryReport,
    adapterInfo: BASE_ADAPTER_INFO,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision: NUMERICAL_PASS,
  });
  const { candidateEnvironmentGate: gate, overallEvidenceDecision: overall } = audit;
  assert.deepEqual(validateLiveFirstInstanceEnvironmentAudit(
    telemetryReport,
    candidateMetadata(audit),
  ), {
    adapterTelemetryAssociation: audit.adapterTelemetryAssociation,
    candidateEnvironmentGate: gate,
    overallEvidenceDecision: overall,
  });

  assert.deepEqual(gate.failureCodes, ['compute-process-set-mismatch']);
  assert.equal(gate.status, 'failed-non-replaceable-process-set-mismatch');
  assert.equal(gate.pass, false);
  assert.equal(gate.collectorPass, true);
  assert.equal(gate.processIdentityPass, false);
  assert.equal(gate.adapterTelemetryAssociationPass, true);
  assert.equal(gate.retryable, false);
  assert.equal(gate.nonReplaceable, true);
  assert.equal(liveFirstInstanceCandidateEnvironmentRequiresRunFailure(gate), false);

  assert.equal(overall.status, 'failed-non-replaceable');
  assert.equal(overall.pass, false);
  assert.equal(overall.retryable, false);
  assert.equal(overall.nonReplaceable, true);
  assert.deepEqual(overall.failedGates, ['candidateEnvironmentGate']);
});

for (const fixture of [
  {
    name: 'unavailable status',
    code: 'telemetry-status-unavailable',
    mutate(report) { report.status = 'unavailable'; },
  },
  {
    name: 'malformed records',
    code: 'telemetry-malformed-records',
    mutate(report) { report.sampling.malformedLineCount = 1; },
  },
  {
    name: 'stderr output',
    code: 'telemetry-stderr-output',
    mutate(report) { report.sampling.stderrByteCount = 1; },
  },
  {
    name: 'no samples',
    code: 'telemetry-no-samples',
    mutate(report) { report.summary.sampleCount = 0; },
  },
  {
    name: 'invalid coverage',
    code: 'telemetry-coverage-invalid',
    mutate(report) {
      report.coverageAudit = {
        pass: false,
        reasons: ['final GPU sample gap exceeds 2000 ms'],
      };
    },
  },
]) {
  test(`collector ${fixture.name} fails the attempt as retryable`, () => {
    const report = availableTelemetryReport();
    fixture.mutate(report);
    const comparison = compareComputeProcessIdentitySets(
      snapshot(BASE_PROCESSES),
      snapshot(BASE_PROCESSES),
    );
    const gate = evaluateLiveFirstInstanceCandidateEnvironmentGate({
      evidenceStatus: 'candidate',
      telemetryReport: report,
      computeProcessIdentityComparison: comparison,
    });

    assert.equal(gate.status, 'failed-retryable-collector');
    assert.equal(gate.pass, false);
    assert.equal(gate.retryable, true);
    assert.equal(gate.nonReplaceable, false);
    assert.ok(gate.failureCodes.includes(fixture.code));
    assert.equal(liveFirstInstanceCandidateEnvironmentRequiresRunFailure(gate), true);

    const overall = evaluateLiveFirstInstanceOverallEvidenceDecision({
      evidenceStatus: 'candidate',
      candidateEnvironmentGate: gate,
      preregisteredNumericalDecision: NUMERICAL_PASS,
    });
    assert.equal(overall.status, 'failed-retryable-collector');
    assert.equal(overall.pass, false);
    assert.equal(overall.retryable, true);
    assert.equal(overall.nonReplaceable, false);
    assert.deepEqual(overall.failedGates, ['candidateEnvironmentGate']);
  });
}

test('missing process snapshot is a retryable collector failure, not a process-set outcome', () => {
  const comparison = compareComputeProcessIdentitySets(
    null,
    snapshot(BASE_PROCESSES),
  );
  const gate = evaluateLiveFirstInstanceCandidateEnvironmentGate({
    evidenceStatus: 'candidate',
    telemetryReport: availableTelemetryReport(),
    computeProcessIdentityComparison: comparison,
  });
  assert.equal(gate.status, 'failed-retryable-collector');
  assert.equal(gate.collectorPass, false);
  assert.equal(gate.processIdentityPass, null);
  assert.equal(gate.adapterTelemetryAssociationPass, null);
  assert.deepEqual(gate.failureCodes, ['compute-process-snapshot-invalid']);
  assert.equal(liveFirstInstanceCandidateEnvironmentRequiresRunFailure(gate), true);
});

for (const fixture of [
  {
    name: 'zero telemetry GPUs',
    code: 'telemetry-gpu-count-not-one',
    mutate({ report }) { report.summary.gpus = []; },
  },
  {
    name: 'multiple telemetry GPUs',
    code: 'telemetry-gpu-count-not-one',
    mutate({ report }) {
      report.summary.gpus.push({
        gpuIndex: 1,
        gpuName: 'NVIDIA GeForce RTX 5070 Ti',
      });
    },
  },
  {
    name: 'non-NVIDIA page adapter vendor',
    code: 'adapter-vendor-not-nvidia',
    mutate({ adapterInfo }) { adapterInfo.vendor = 'amd'; },
  },
  {
    name: 'near-match telemetry GPU name',
    code: 'adapter-telemetry-name-mismatch',
    mutate({ report }) {
      report.summary.gpus[0].gpuName = 'NVIDIA GeForce RTX 5070';
    },
  },
]) {
  test(`${fixture.name} is a completed non-replaceable association failure`, () => {
    const telemetryReport = availableTelemetryReport();
    telemetryReport.computeProcesses = {
      pre: snapshot(BASE_PROCESSES),
      post: snapshot(BASE_PROCESSES),
    };
    const adapterInfo = structuredClone(BASE_ADAPTER_INFO);
    fixture.mutate({ report: telemetryReport, adapterInfo });
    const comparison = compareComputeProcessIdentitySets(
      telemetryReport.computeProcesses.pre,
      telemetryReport.computeProcesses.post,
    );
    const audit = createLiveFirstInstanceEnvironmentAudit({
      evidenceStatus: 'candidate',
      telemetryReport,
      adapterInfo,
      computeProcessIdentityComparison: comparison,
      preregisteredNumericalDecision: NUMERICAL_PASS,
    });

    assert.equal(audit.adapterTelemetryAssociation.pass, false);
    assert.ok(audit.adapterTelemetryAssociation.failureCodes.includes(fixture.code));
    assert.equal(
      audit.candidateEnvironmentGate.status,
      'failed-non-replaceable-adapter-telemetry-association',
    );
    assert.equal(audit.candidateEnvironmentGate.collectorPass, true);
    assert.equal(audit.candidateEnvironmentGate.processIdentityPass, true);
    assert.equal(audit.candidateEnvironmentGate.adapterTelemetryAssociationPass, false);
    assert.equal(audit.candidateEnvironmentGate.retryable, false);
    assert.equal(audit.candidateEnvironmentGate.nonReplaceable, true);
    assert.equal(liveFirstInstanceCandidateEnvironmentRequiresRunFailure(
      audit.candidateEnvironmentGate,
    ), false);
    assert.equal(audit.overallEvidenceDecision.status, 'failed-non-replaceable');
    assert.deepEqual(validateLiveFirstInstanceEnvironmentAudit(
      telemetryReport,
      candidateMetadata(audit, adapterInfo),
    ), {
      adapterTelemetryAssociation: audit.adapterTelemetryAssociation,
      candidateEnvironmentGate: audit.candidateEnvironmentGate,
      overallEvidenceDecision: audit.overallEvidenceDecision,
    });
  });
}

test('posthoc reconstruction rejects association-field, digest, and source identity tampering', () => {
  const telemetryReport = availableTelemetryReport();
  telemetryReport.computeProcesses = {
    pre: snapshot(BASE_PROCESSES),
    post: snapshot(BASE_PROCESSES),
  };
  const comparison = compareComputeProcessIdentitySets(
    telemetryReport.computeProcesses.pre,
    telemetryReport.computeProcesses.post,
  );
  const audit = createLiveFirstInstanceEnvironmentAudit({
    evidenceStatus: 'candidate',
    telemetryReport,
    adapterInfo: BASE_ADAPTER_INFO,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision: NUMERICAL_PASS,
  });

  for (const mutate of [
    (metadata) => {
      metadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation
        .normalizedTelemetryGpuName = 'nvidia geforce rtx 5090';
    },
    (metadata) => {
      metadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation
        .associationSha256 = '0'.repeat(64);
    },
    (metadata) => {
      metadata.environment.benchmarkPage.adapterInfo.description =
        'NVIDIA GeForce RTX 5090';
    },
  ]) {
    const metadata = candidateMetadata(structuredClone(audit));
    mutate(metadata);
    assert.throws(
      () => validateLiveFirstInstanceEnvironmentAudit(telemetryReport, metadata),
      /does not reconstruct from telemetry evidence/,
    );
  }
});

test('posthoc refuses to count a retryable collector failure as a completed candidate', () => {
  const telemetryReport = availableTelemetryReport();
  telemetryReport.sampling.malformedLineCount = 1;
  telemetryReport.computeProcesses = {
    pre: snapshot(BASE_PROCESSES),
    post: snapshot(BASE_PROCESSES),
  };
  const comparison = compareComputeProcessIdentitySets(
    telemetryReport.computeProcesses.pre,
    telemetryReport.computeProcesses.post,
  );
  const audit = createLiveFirstInstanceEnvironmentAudit({
    evidenceStatus: 'candidate',
    telemetryReport,
    adapterInfo: BASE_ADAPTER_INFO,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision: NUMERICAL_PASS,
  });
  assert.equal(audit.candidateEnvironmentGate.retryable, true);
  assert.throws(
    () => validateLiveFirstInstanceEnvironmentAudit(
      telemetryReport,
      candidateMetadata(audit),
    ),
    /completed candidate contains a retryable telemetry collector failure/,
  );
});

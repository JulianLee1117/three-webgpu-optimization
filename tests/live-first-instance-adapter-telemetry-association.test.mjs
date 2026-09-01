import assert from 'node:assert/strict';
import test from 'node:test';

import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
  normalizeLiveFirstInstanceGpuIdentityText,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';

const ADAPTER = Object.freeze({
  vendor: 'nvidia',
  description: 'NVIDIA GeForce RTX 5070 Ti',
});

function telemetry(gpus, coverageGpus = gpus) {
  return {
    summary: { gpus },
    coverageAudit: { gpuIdentities: coverageGpus },
  };
}

function gpu(overrides = {}) {
  return {
    gpuIndex: 0,
    gpuName: 'NVIDIA GeForce RTX 5070 Ti',
    gpuUuid: 'GPU-a',
    ...overrides,
  };
}

test('ASCII identity normalization is deterministic, locale-free, and deliberately narrow', () => {
  assert.equal(
    normalizeLiveFirstInstanceGpuIdentityText('\t NVIDIA   GeForce\tRTX 5070 Ti  '),
    'nvidia geforce rtx 5070 ti',
  );
  assert.equal(normalizeLiveFirstInstanceGpuIdentityText('NVIDIA\nGPU'), null);
  assert.equal(normalizeLiveFirstInstanceGpuIdentityText('NVIDI\u0410'), null);
  assert.equal(normalizeLiveFirstInstanceGpuIdentityText('\t  '), null);
  assert.equal(normalizeLiveFirstInstanceGpuIdentityText(null), null);
});

test('one NVIDIA page adapter binds exactly to one normalized telemetry GPU identity', () => {
  const association = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: {
      vendor: '\tNVIDIA ',
      description: ' NVIDIA  GeForce\tRTX 5070 Ti ',
    },
    telemetryReport: telemetry([gpu({ gpuName: 'nvidia geforce rtx 5070 ti' })]),
  });
  assert.equal(association.schemaVersion, 1);
  assert.equal(association.pass, true);
  assert.equal(association.normalizedAdapterVendor, 'nvidia');
  assert.equal(association.normalizedAdapterDescription, 'nvidia geforce rtx 5070 ti');
  assert.equal(association.telemetryGpuCount, 1);
  assert.equal(association.telemetryGpuIndex, 0);
  assert.equal(association.normalizedTelemetryGpuName, 'nvidia geforce rtx 5070 ti');
  assert.deepEqual(association.failureCodes, []);
  for (const field of [
    'adapterIdentitySha256',
    'telemetryIdentitySha256',
    'associationSha256',
  ]) {
    assert.match(association[field], /^[0-9a-f]{64}$/);
  }

  const equivalent = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu()]),
  });
  assert.equal(equivalent.associationSha256, association.associationSha256);
  assert.deepEqual({
    adapterIdentitySha256: equivalent.adapterIdentitySha256,
    telemetryIdentitySha256: equivalent.telemetryIdentitySha256,
    associationSha256: equivalent.associationSha256,
  }, {
    adapterIdentitySha256:
      '0372c5201b64aa196c5c21044b2b367fb7e61c4ffcc9ff0cd31cffb44b91dab0',
    telemetryIdentitySha256:
      '83a7b371eeecfb6ae3a9fd3c12c03d82d2220bab2c5ce60717e6a98425856970',
    associationSha256:
      '78559259332c47ac7a741dba7a7574b28970e5dfb322fd4f90f890b8a115d22a',
  });
});

for (const fixture of [
  { name: 'missing telemetry GPU array', gpus: undefined },
  { name: 'zero telemetry GPUs', gpus: [] },
  { name: 'two distinct telemetry GPUs', gpus: [gpu(), gpu({ gpuIndex: 1 })] },
  { name: 'two duplicate telemetry GPUs', gpus: [gpu(), gpu()] },
]) {
  test(`association fails closed for ${fixture.name}`, () => {
    const association = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo: ADAPTER,
      telemetryReport: telemetry(fixture.gpus),
    });
    assert.equal(association.pass, false);
    assert.deepEqual(association.failureCodes, [
      'telemetry-gpu-count-not-one',
      'telemetry-coverage-gpu-count-not-one',
    ]);
    assert.equal(association.telemetryGpuCount, Array.isArray(fixture.gpus)
      ? fixture.gpus.length
      : null);
    assert.equal(association.telemetryGpuIndex, null);
    assert.equal(association.normalizedTelemetryGpuName, null);
  });
}

test('association requires an exact sole coverage-to-summary GPU identity join', () => {
  const missingCoverage = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu()], null),
  });
  assert.equal(missingCoverage.pass, false);
  assert.ok(missingCoverage.failureCodes.includes(
    'telemetry-coverage-gpu-count-not-one',
  ));

  for (const [name, coverageGpu] of [
    ['index', gpu({ gpuIndex: 1 })],
    ['name', gpu({ gpuName: 'NVIDIA GeForce RTX 5070' })],
    ['UUID', gpu({ gpuUuid: 'GPU-b' })],
  ]) {
    const mismatch = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo: ADAPTER,
      telemetryReport: telemetry([gpu()], [coverageGpu]),
    });
    assert.equal(mismatch.pass, false, name);
    assert.ok(
      mismatch.failureCodes.includes('telemetry-summary-coverage-identity-mismatch'),
      name,
    );
  }

  const invalidCoverage = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu()], [gpu({ gpuUuid: '' })]),
  });
  assert.equal(invalidCoverage.pass, false);
  assert.ok(invalidCoverage.failureCodes.includes(
    'telemetry-coverage-gpu-identity-invalid',
  ));
});

for (const fixture of [
  {
    name: 'non-NVIDIA vendor',
    adapterInfo: { ...ADAPTER, vendor: 'amd' },
    expectedCode: 'adapter-vendor-not-nvidia',
  },
  {
    name: 'Unicode vendor lookalike',
    adapterInfo: { ...ADAPTER, vendor: 'NVIDI\u0410' },
    expectedCode: 'adapter-vendor-invalid',
  },
  {
    name: 'missing description',
    adapterInfo: { vendor: 'nvidia' },
    expectedCode: 'adapter-description-invalid',
  },
]) {
  test(`association rejects ${fixture.name}`, () => {
    const association = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo: fixture.adapterInfo,
      telemetryReport: telemetry([gpu()]),
    });
    assert.equal(association.pass, false);
    assert.ok(association.failureCodes.includes(fixture.expectedCode));
  });
}

for (const gpuName of [
  'NVIDIA GeForce RTX 5070',
  'NVIDIA GeForce RTX 5070 Ti SUPER',
  'GeForce RTX 5070 Ti',
  'NVIDIA-GeForce RTX 5070 Ti',
]) {
  test(`association rejects near or partial telemetry name ${JSON.stringify(gpuName)}`, () => {
    const association = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo: ADAPTER,
      telemetryReport: telemetry([gpu({ gpuName })]),
    });
    assert.equal(association.pass, false);
    assert.deepEqual(association.failureCodes, ['adapter-telemetry-name-mismatch']);
  });
}

test('association rejects an invalid sole telemetry GPU identity without fuzzy fallback', () => {
  const invalidIndex = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu({ gpuIndex: -1 })]),
  });
  assert.equal(invalidIndex.pass, false);
  assert.deepEqual(invalidIndex.failureCodes, [
    'telemetry-gpu-index-invalid',
    'telemetry-coverage-gpu-identity-invalid',
  ]);

  const invalidName = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu({ gpuName: 'NVIDIA\nGeForce RTX 5070 Ti' })]),
  });
  assert.equal(invalidName.pass, false);
  assert.deepEqual(invalidName.failureCodes, ['telemetry-gpu-name-invalid']);

  const invalidUuid = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: ADAPTER,
    telemetryReport: telemetry([gpu({ gpuUuid: '' })]),
  });
  assert.equal(invalidUuid.pass, false);
  assert.ok(invalidUuid.failureCodes.includes('telemetry-gpu-uuid-invalid'));
  assert.ok(invalidUuid.failureCodes.includes(
    'telemetry-coverage-gpu-identity-invalid',
  ));
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createServer } from 'vite';
import {
  createCandidateViteRuntimeGuard,
  collectExecutionDependencyClosure,
  executionDependencyClosuresMatch,
  validateCandidateViteRuntimeAudit,
} from '../scripts/execution-dependency-closure.mjs';
import {
  validateLiveCandidateReservationAndDependencies,
  validateLiveCandidateViteRuntime,
} from '../analysis/summarize.mjs';

test('installed dependency closure is deterministic and excludes only tool caches/shims', async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'dependency-closure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, 'node_modules', 'three', 'build'), { recursive: true });
  await mkdir(path.join(root, 'node_modules', '.bin'), { recursive: true });
  await mkdir(path.join(root, 'node_modules', '.vite-cache'), { recursive: true });
  await writeFile(path.join(root, 'node_modules', 'three', 'package.json'), '{"version":"1"}');
  await writeFile(path.join(root, 'node_modules', 'three', 'build', 'three.module.js'), 'one');
  await writeFile(path.join(root, 'node_modules', '.bin', 'three'), 'shim-one');
  await writeFile(path.join(root, 'node_modules', '.vite-cache', 'chunk.js'), 'cache-one');

  const first = await collectExecutionDependencyClosure(root);
  const repeated = await collectExecutionDependencyClosure(root);
  assert.equal(first.fileCount, 2);
  assert.equal(first.totalBytes, Buffer.byteLength('{"version":"1"}') + 3);
  assert.equal(executionDependencyClosuresMatch(first, repeated), true);

  await writeFile(path.join(root, 'node_modules', '.bin', 'three'), 'shim-two');
  await writeFile(path.join(root, 'node_modules', '.vite-cache', 'chunk.js'), 'cache-two');
  assert.equal(
    executionDependencyClosuresMatch(first, await collectExecutionDependencyClosure(root)),
    true,
  );

  await writeFile(path.join(root, 'node_modules', 'three', 'build', 'three.module.js'), 'two');
  const changed = await collectExecutionDependencyClosure(root);
  assert.notEqual(changed.sha256, first.sha256);
  assert.equal(executionDependencyClosuresMatch(first, changed), false);
});

async function createViteRuntimeFixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'candidate-vite-runtime-'));
  await mkdir(path.join(root, 'src', 'strategies'), { recursive: true });
  await mkdir(path.join(root, 'node_modules', 'fake-dep'), { recursive: true });
  await mkdir(path.join(root, 'node_modules', '.vite', 'deps'), { recursive: true });
  await writeFile(
    path.join(root, 'index.html'),
    '<script type="module" src="/src/main.js"></script>',
  );
  await writeFile(
    path.join(root, 'src', 'main.js'),
    "import { value } from 'fake-dep';\nimport { strategy } from './strategies/live-first-instance-crossover.js';\nglobalThis.fixtureValue = `${value}:${strategy}`;\n",
  );
  await writeFile(
    path.join(root, 'src', 'strategies', 'live-first-instance-crossover.js'),
    "export const strategy = 'LIVE_STRATEGY';\n",
  );
  await writeFile(
    path.join(root, 'node_modules', 'fake-dep', 'package.json'),
    JSON.stringify({ name: 'fake-dep', version: '1.0.0', type: 'module', exports: './index.js' }),
  );
  await writeFile(
    path.join(root, 'node_modules', 'fake-dep', 'index.js'),
    "export const value = 'CLEAN_UNOPTIMIZED_DEPENDENCY';\n",
  );
  await writeFile(
    path.join(root, 'node_modules', '.vite', 'deps', 'fake-dep.js'),
    "export const value = 'POISONED_PREEXISTING_VITE_CACHE';\n",
  );
  return root;
}

test('candidate Vite runtime ignores poisoned preexisting optimizer cache', async (t) => {
  const root = await createViteRuntimeFixture();
  const dependencyClosureBefore = await collectExecutionDependencyClosure(root);
  const guard = await createCandidateViteRuntimeGuard(root);
  let server = null;
  t.after(async () => {
    await server?.close();
    await guard.dispose();
    await rm(root, { recursive: true, force: true });
  });

  assert.equal(path.relative(path.join(root, 'node_modules'), guard.cacheDirectory).startsWith('..'), true);
  server = await createServer({
    root,
    configFile: false,
    server: {
      host: '127.0.0.1',
      port: 0,
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
    },
    logLevel: 'error',
    ...guard.viteConfig,
  });
  await server.listen();
  const baseUrl = server.resolvedUrls?.local?.[0];
  assert.ok(baseUrl);

  const rootResponse = await fetch(baseUrl);
  assert.equal(rootResponse.status, 200);
  assert.equal(rootResponse.headers.get('cross-origin-opener-policy'), 'same-origin');
  assert.equal(rootResponse.headers.get('cross-origin-embedder-policy'), 'require-corp');
  const rootHtml = await rootResponse.text();
  assert.equal(rootHtml, '<script type="module" src="/src/main.js"></script>');
  assert.doesNotMatch(rootHtml, /@vite\/client|wsToken/);
  const indexResponse = await fetch(new URL('/index.html?candidate-entry=1', baseUrl));
  assert.equal(await indexResponse.text(), rootHtml);

  const mainResponse = await fetch(new URL('/src/main.js', baseUrl));
  assert.equal(mainResponse.status, 200);
  const mainContents = await mainResponse.text();
  assert.doesNotMatch(mainContents, /POISONED_PREEXISTING_VITE_CACHE|\.vite\/deps/i);
  const dependencyUrl = mainContents.match(/from\s+["']([^"']*fake-dep[^"']*)["']/)?.[1];
  assert.ok(dependencyUrl, `Vite did not retain a resolvable fake-dep import: ${mainContents}`);

  const dependencyResponse = await fetch(new URL(dependencyUrl, baseUrl));
  assert.equal(dependencyResponse.status, 200);
  const dependencyContents = await dependencyResponse.text();
  assert.match(dependencyContents, /CLEAN_UNOPTIMIZED_DEPENDENCY/);
  assert.doesNotMatch(dependencyContents, /POISONED_PREEXISTING_VITE_CACHE|\.vite\/deps/i);
  const strategyResponse = await fetch(
    new URL('/src/strategies/live-first-instance-crossover.js', baseUrl),
  );
  assert.equal(strategyResponse.status, 200);
  const strategyContents = await strategyResponse.text();
  assert.match(strategyContents, /LIVE_STRATEGY/);
  const queriedStrategyResponse = await fetch(
    new URL('/src/strategies/live-first-instance-crossover.js?candidate-query=1', baseUrl),
  );
  assert.equal(queriedStrategyResponse.status, 200);
  assert.match(await queriedStrategyResponse.text(), /LIVE_STRATEGY/);

  await server.close();
  server = null;
  const audit = await guard.finalize();
  assert.equal(audit.configuration.optimizeDepsNoDiscovery, true);
  assert.deepEqual(audit.configuration.optimizeDepsInclude, []);
  assert.equal(audit.configuration.cachePreexisted, false);
  assert.equal(audit.cache.entryCount, 0);
  assert.equal(audit.configuration.appType, 'custom');
  assert.equal(audit.entryHtml.successfulResponseCount, 2);
  assert.equal(audit.entryHtml.responseVariants[0].responseCount, 2);
  assert.equal(audit.prohibitedOptimizedArtifactCount, 0);
  assert.equal(audit.dependencyModuleCount, 1);
  assert.deepEqual(
    audit.modules.map((record) => record.sourceRelativePath),
    [
      'node_modules/fake-dep/index.js',
      'src/main.js',
      'src/strategies/live-first-instance-crossover.js',
    ],
  );
  const mainModule = audit.modules.find(
    (record) => record.sourceRelativePath === 'src/main.js',
  );
  assert.equal(mainModule.successfulResponseCount, 1);
  assert.deepEqual(mainModule.transformedVariants, [{
    byteCount: Buffer.byteLength(mainContents),
    sha256: createHash('sha256').update(mainContents).digest('hex'),
    responseCount: 1,
  }]);
  assert.notEqual(mainModule.sourceSha256, mainModule.transformedVariants[0].sha256);
  const strategyModule = audit.modules.find(
    (record) => record.sourceRelativePath
      === 'src/strategies/live-first-instance-crossover.js',
  );
  assert.equal(strategyModule.successfulResponseCount, 2);
  assert.equal(
    strategyModule.transformedVariants.reduce(
      (sum, variant) => sum + variant.responseCount,
      0,
    ),
    2,
  );
  assert.deepEqual(await validateCandidateViteRuntimeAudit(audit, root), {
    pass: true,
    reasons: [],
  });
  await assert.doesNotReject(validateLiveCandidateViteRuntime({
    evidenceStatus: 'candidate',
    protocol: { matrixKind: 'first-instance-live' },
    candidateViteRuntimeAudit: audit,
  }, root));
  const wrongRequiredPath = structuredClone(audit);
  wrongRequiredPath.requiredModulePaths[1] = 'src/strategies/forged.js';
  await assert.rejects(
    validateLiveCandidateViteRuntime({
      evidenceStatus: 'candidate',
      protocol: { matrixKind: 'first-instance-live' },
      candidateViteRuntimeAudit: wrongRequiredPath,
    }, root),
    /frozen fail-closed policy|required runtime module/,
  );
  await writeFile(
    path.join(root, 'src', 'strategies', 'live-first-instance-crossover.js'),
    "export const strategy = 'TAMPERED_AFTER_RUNTIME';\n",
  );
  await assert.rejects(
    validateLiveCandidateViteRuntime({
      evidenceStatus: 'candidate',
      protocol: { matrixKind: 'first-instance-live' },
      candidateViteRuntimeAudit: audit,
    }, root),
    /differs from its retained source commitment/,
  );
  assert.equal(
    executionDependencyClosuresMatch(
      dependencyClosureBefore,
      await collectExecutionDependencyClosure(root),
    ),
    true,
  );
});

test('fresh candidate Vite servers emit byte-identical canonical runtime audits', async (t) => {
  const root = await createViteRuntimeFixture();
  t.after(() => rm(root, { recursive: true, force: true }));

  async function captureAudit() {
    const guard = await createCandidateViteRuntimeGuard(root);
    let server = null;
    try {
      server = await createServer({
        root,
        configFile: false,
        server: { host: '127.0.0.1', port: 0 },
        logLevel: 'error',
        ...guard.viteConfig,
      });
      await server.listen();
      const baseUrl = server.resolvedUrls?.local?.[0];
      assert.ok(baseUrl);
      const rootResponse = await fetch(baseUrl);
      assert.equal(rootResponse.status, 200);
      assert.doesNotMatch(await rootResponse.text(), /@vite\/client|wsToken/);
      const mainContents = await (await fetch(new URL('/src/main.js', baseUrl))).text();
      const dependencyUrl = mainContents.match(
        /from\s+["']([^"']*fake-dep[^"']*)["']/,
      )?.[1];
      assert.ok(dependencyUrl);
      for (const requestPath of [
        dependencyUrl,
        '/src/strategies/live-first-instance-crossover.js?candidate-query=1',
      ]) {
        const response = await fetch(new URL(requestPath, baseUrl));
        assert.equal(response.status, 200);
        await response.arrayBuffer();
      }
      await server.close();
      server = null;
      return await guard.finalize();
    } finally {
      await server?.close();
      await guard.dispose();
    }
  }

  const first = await captureAudit();
  const second = await captureAudit();
  assert.deepEqual(second, first);
});

test('candidate Vite runtime blocks and records any optimized-cache request', async (t) => {
  const root = await createViteRuntimeFixture();
  const guard = await createCandidateViteRuntimeGuard(root);
  let server = null;
  t.after(async () => {
    await server?.close();
    await guard.dispose();
    await rm(root, { recursive: true, force: true });
  });
  server = await createServer({
    root,
    configFile: false,
    server: { host: '127.0.0.1', port: 0 },
    logLevel: 'error',
    ...guard.viteConfig,
  });
  await server.listen();
  const baseUrl = server.resolvedUrls?.local?.[0];
  assert.ok(baseUrl);

  for (const requestPath of [
    '/node_modules/.vite/deps/fake-dep.js',
    '/node_modules/.vite-cache/deps/fake-dep.js',
    '/@vite/client',
    '/@id/unmapped-runtime.js',
  ]) {
    const response = await fetch(new URL(requestPath, baseUrl));
    assert.equal(response.status, 403);
    assert.doesNotMatch(await response.text(), /POISONED_PREEXISTING_VITE_CACHE/);
  }
  const missingResponse = await fetch(new URL('/src/missing-runtime.js', baseUrl), {
    headers: { accept: 'application/javascript' },
  });
  assert.equal(missingResponse.status, 404);
  await missingResponse.arrayBuffer();
  await server.close();
  server = null;
  await assert.rejects(
    guard.finalize(),
    /prohibited optimized artifact/,
  );
});

test('live candidate metadata binds its reservation to stable executed dependency bytes', () => {
  const closure = {
    schemaVersion: 1,
    kind: 'installed-execution-dependency-closure',
    root: 'node_modules',
    format: 'node-modules-sorted-path-size-content-sha256-v1',
    hashAlgorithm: 'sha256',
    exclusions: ['.bin/**', '.vite*/**'],
    fileCount: 5,
    totalBytes: 10,
    sha256: 'a'.repeat(64),
  };
  const metadata = {
    evidenceStatus: 'candidate',
    protocol: { matrixKind: 'first-instance-live' },
    sourceProvenance: {
      start: { commit: 'b'.repeat(40), tree: 'c'.repeat(40) },
    },
    candidateSeriesReservation: {
      schemaVersion: 1,
      kind: 'first-instance-live-candidate-series-reservation',
      seriesId: 'series-fixture',
      reservationEventSha256: 'd'.repeat(64),
      attemptOrdinal: 1,
      matrixOrdinal: 1,
      sourceCommit: 'b'.repeat(40),
      sourceTree: 'c'.repeat(40),
      executionDependencyClosureSha256: closure.sha256,
    },
    executionDependencyClosure: {
      start: { ...closure },
      end: { ...closure },
      stable: true,
    },
  };
  assert.doesNotThrow(() => validateLiveCandidateReservationAndDependencies(metadata));

  const changed = structuredClone(metadata);
  changed.executionDependencyClosure.end.sha256 = 'e'.repeat(64);
  assert.throws(
    () => validateLiveCandidateReservationAndDependencies(changed),
    /start\/end closures differ/,
  );

  const wrongReservation = structuredClone(metadata);
  wrongReservation.candidateSeriesReservation.sourceTree = 'f'.repeat(40);
  assert.throws(
    () => validateLiveCandidateReservationAndDependencies(wrongReservation),
    /reservation differs from executed source\/dependency bytes/,
  );
});

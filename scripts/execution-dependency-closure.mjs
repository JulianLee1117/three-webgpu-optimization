import { createHash } from 'node:crypto';
import {
  access,
  lstat,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rm,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const FORMAT = 'node-modules-sorted-path-size-content-sha256-v1';
const EXCLUSIONS = Object.freeze(['.bin/**', '.vite*/**']);
export const CANDIDATE_VITE_RUNTIME_POLICY_ID =
  'vite-unoptimized-fresh-cache-runtime-module-audit-v3';
export const CANDIDATE_VITE_REQUIRED_MODULE_PATHS = Object.freeze([
  'src/main.js',
  'src/strategies/live-first-instance-crossover.js',
]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const EXCLUDED_VITE_CACHE_PATH_PATTERN = /(?:^|[/\\])\.vite[^/\\]*(?:[/\\]|$)/i;
const AUDITABLE_MODULE_REQUEST_PATTERN = /\.(?:[cm]?[jt]sx?|css)$/i;

function comparePath(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function excludedTopLevel(name) {
  return name === '.bin' || name.startsWith('.vite');
}

function pathInside(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative === ''
    || (!path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`));
}

function normalizedModuleFilename(identifier) {
  const withoutNullPrefix = identifier.startsWith('\0') ? identifier.slice(1) : identifier;
  const queryIndex = withoutNullPrefix.indexOf('?');
  return queryIndex === -1 ? withoutNullPrefix : withoutNullPrefix.slice(0, queryIndex);
}

function decodedRequestPath(value) {
  let decoded = String(value ?? '');
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      break;
    }
  }
  return decoded;
}

function referencesOptimizedDependencyArtifact(value) {
  return EXCLUDED_VITE_CACHE_PATH_PATTERN.test(decodedRequestPath(value));
}

function entryHtmlContainsForbiddenRuntimeReference(contents) {
  const text = contents.toString('utf8');
  return text.includes('/@vite/')
    || text.includes('/@id/')
    || referencesOptimizedDependencyArtifact(text);
}

function sourceRelativePathFromRequest(requestUrl, root) {
  const pathname = decodedRequestPath(String(requestUrl ?? '')).split(/[?#]/, 1)[0];
  let absolutePath = null;
  if (pathname.startsWith('/@fs/')) {
    absolutePath = path.resolve(pathname.slice('/@fs/'.length));
  } else if (pathname.startsWith('/') && !pathname.startsWith('/@')) {
    absolutePath = path.resolve(root, ...pathname.slice(1).split('/'));
  }
  if (absolutePath === null || !pathInside(root, absolutePath)) return null;
  const relativePath = path.relative(root, absolutePath).split(path.sep).join('/');
  return relativePath.length > 0 && AUDITABLE_MODULE_REQUEST_PATTERN.test(relativePath)
    ? relativePath
    : null;
}

function responseChunkBuffer(chunk, encoding) {
  if (chunk === null || chunk === undefined || typeof chunk === 'function') return null;
  if (Buffer.isBuffer(chunk)) return Buffer.from(chunk);
  if (ArrayBuffer.isView(chunk)) {
    return Buffer.from(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
  }
  return Buffer.from(String(chunk), typeof encoding === 'string' ? encoding : undefined);
}

function exactKeys(value, expected) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length
    && actual.every((key, index) => key === wanted[index]);
}

async function emptyDirectoryAudit(directory) {
  try {
    const stats = await lstat(directory);
    if (!stats.isDirectory() || stats.isSymbolicLink()) {
      throw new Error('Candidate Vite cache path is not a regular directory.');
    }
    const entries = await readdir(directory);
    return { existedAtFinalization: true, entryCount: entries.length };
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return { existedAtFinalization: false, entryCount: 0 };
    }
    throw error;
  }
}

export function candidateViteRuntimeModulesSha256(modules) {
  const digest = createHash('sha256');
  digest.update('candidate-vite-runtime-modules-v2\0');
  for (const module of modules) {
    digest.update(`${Buffer.byteLength(module.sourceRelativePath)}:`);
    digest.update(module.sourceRelativePath);
    digest.update(
      `\0${module.sourceByteCount}:${module.sourceSha256}:`
        + `${module.successfulResponseCount}\0`,
    );
    for (const variant of module.transformedVariants) {
      digest.update(`${variant.byteCount}:${variant.sha256}:${variant.responseCount}\0`);
    }
  }
  return digest.digest('hex');
}

export async function validateCandidateViteRuntimeAudit(audit, projectRoot) {
  const reasons = [];
  const auditModules = Array.isArray(audit?.modules) ? audit.modules : [];
  const requiredModulePaths = Array.isArray(audit?.requiredModulePaths)
    ? audit.requiredModulePaths
    : [];
  if (!exactKeys(audit, [
    'schemaVersion',
    'kind',
    'policyId',
    'configuration',
    'cache',
    'entryHtml',
    'prohibitedOptimizedArtifactCount',
    'requiredModulePaths',
    'moduleCount',
    'dependencyModuleCount',
    'modulesSha256',
    'modules',
  ])) reasons.push('runtime audit has an unexpected schema');
  if (audit?.schemaVersion !== 3
    || audit?.kind !== 'candidate-vite-runtime-module-audit'
    || audit?.policyId !== CANDIDATE_VITE_RUNTIME_POLICY_ID) {
    reasons.push('runtime audit identity is invalid');
  }
  if (!exactKeys(audit?.configuration, [
    'configFile',
    'appType',
    'optimizeDepsNoDiscovery',
    'optimizeDepsInclude',
    'entryHtmlPolicy',
    'cacheDirectoryPolicy',
    'cachePreexisted',
  ])
    || audit?.configuration?.configFile !== false
    || audit?.configuration?.appType !== 'custom'
    || audit?.configuration?.optimizeDepsNoDiscovery !== true
    || !Array.isArray(audit?.configuration?.optimizeDepsInclude)
    || audit.configuration.optimizeDepsInclude.length !== 0
    || audit?.configuration?.entryHtmlPolicy
      !== 'exact-tracked-index-html-without-vite-client'
    || audit?.configuration?.cacheDirectoryPolicy
      !== 'unique-fresh-os-temporary-directory-outside-project'
    || audit?.configuration?.cachePreexisted !== false) {
    reasons.push('runtime audit Vite configuration is not the frozen fail-closed policy');
  }
  const entryHtml = audit?.entryHtml;
  if (!exactKeys(entryHtml, [
    'sourceRelativePath',
    'sourceByteCount',
    'sourceSha256',
    'successfulResponseCount',
    'responseHeaders',
    'responseVariants',
  ])
    || entryHtml?.sourceRelativePath !== 'index.html'
    || !Number.isSafeInteger(entryHtml?.sourceByteCount)
    || entryHtml.sourceByteCount < 1
    || !SHA256_PATTERN.test(entryHtml?.sourceSha256 ?? '')
    || !Number.isSafeInteger(entryHtml?.successfulResponseCount)
    || entryHtml.successfulResponseCount < 1
    || !exactKeys(entryHtml?.responseHeaders, [
      'crossOriginOpenerPolicy',
      'crossOriginEmbedderPolicy',
      'contentType',
      'cacheControl',
    ])
    || entryHtml.responseHeaders.crossOriginOpenerPolicy !== 'same-origin'
    || entryHtml.responseHeaders.crossOriginEmbedderPolicy !== 'require-corp'
    || entryHtml.responseHeaders.contentType !== 'text/html; charset=utf-8'
    || entryHtml.responseHeaders.cacheControl !== 'no-store'
    || !Array.isArray(entryHtml?.responseVariants)
    || entryHtml.responseVariants.length !== 1
    || !exactKeys(entryHtml.responseVariants[0], [
      'byteCount',
      'sha256',
      'responseCount',
    ])
    || entryHtml.responseVariants[0].byteCount !== entryHtml.sourceByteCount
    || entryHtml.responseVariants[0].sha256 !== entryHtml.sourceSha256
    || entryHtml.responseVariants[0].responseCount
      !== entryHtml.successfulResponseCount) {
    reasons.push('runtime audit entry HTML evidence is invalid');
  } else {
    try {
      const contents = await readFile(path.join(projectRoot, 'index.html'));
      if (contents.length !== entryHtml.sourceByteCount
        || createHash('sha256').update(contents).digest('hex') !== entryHtml.sourceSha256) {
        reasons.push('runtime entry HTML differs from its retained source commitment');
      }
      if (entryHtmlContainsForbiddenRuntimeReference(contents)) {
        reasons.push('runtime entry HTML references a forbidden Vite runtime or cache artifact');
      }
    } catch {
      reasons.push('runtime entry HTML is unavailable');
    }
  }
  if (!exactKeys(audit?.cache, ['existedAtFinalization', 'entryCount'])
    || typeof audit?.cache?.existedAtFinalization !== 'boolean'
    || audit?.cache?.entryCount !== 0) {
    reasons.push('candidate Vite cache was not empty at finalization');
  }
  if (audit?.prohibitedOptimizedArtifactCount !== 0) {
    reasons.push('an excluded .vite* cache artifact was requested, loaded, or referenced');
  }
  if (!Array.isArray(audit?.requiredModulePaths)
    || JSON.stringify(audit.requiredModulePaths)
      !== JSON.stringify(CANDIDATE_VITE_REQUIRED_MODULE_PATHS)
    || !Array.isArray(audit?.modules)
    || audit.modules.length === 0
    || audit?.moduleCount !== audit.modules.length) {
    reasons.push('runtime module inventory is missing or has an inconsistent count');
  }
  const root = path.resolve(projectRoot);
  const modulePaths = [];
  let dependencyModuleCount = 0;
  for (const [index, module] of auditModules.entries()) {
    if (!exactKeys(module, [
      'sourceRelativePath',
      'sourceByteCount',
      'sourceSha256',
      'successfulResponseCount',
      'transformedVariants',
    ])) {
      reasons.push(`runtime module ${index} has an unexpected schema`);
      continue;
    }
    const relativePath = module.sourceRelativePath;
    const absolutePath = typeof relativePath === 'string'
      ? path.resolve(root, ...relativePath.split('/'))
      : root;
    const canonicalRelativePath = path.relative(root, absolutePath).split(path.sep).join('/');
    if (typeof relativePath !== 'string'
      || relativePath.length === 0
      || relativePath.includes('\\')
      || path.isAbsolute(relativePath)
      || relativePath !== canonicalRelativePath
      || !pathInside(root, absolutePath)
      || referencesOptimizedDependencyArtifact(relativePath)) {
      reasons.push(`runtime module ${index} has an unsafe source path`);
      continue;
    }
    if (modulePaths.length > 0 && relativePath <= modulePaths[modulePaths.length - 1]) {
      reasons.push('runtime module paths are not unique and strictly sorted');
    }
    modulePaths.push(relativePath);
    if (relativePath.startsWith('node_modules/')) dependencyModuleCount += 1;
    if (!Number.isSafeInteger(module.sourceByteCount) || module.sourceByteCount < 0
      || !SHA256_PATTERN.test(module.sourceSha256 ?? '')
      || !Number.isSafeInteger(module.successfulResponseCount)
      || module.successfulResponseCount < 1
      || !Array.isArray(module.transformedVariants)
      || module.transformedVariants.length === 0) {
      reasons.push(`runtime module ${index} has invalid byte or transform commitments`);
      continue;
    }
    let variantsValid = true;
    for (const variant of module.transformedVariants) {
      if (!exactKeys(variant, ['byteCount', 'sha256', 'responseCount'])
        || !Number.isSafeInteger(variant.byteCount) || variant.byteCount < 0
        || !SHA256_PATTERN.test(variant.sha256 ?? '')
        || !Number.isSafeInteger(variant.responseCount) || variant.responseCount < 1) {
        reasons.push(`runtime module ${index} has an invalid transformed variant`);
        variantsValid = false;
      }
    }
    const variantsSorted = variantsValid && module.transformedVariants.every(
      (variant, variantIndex) => variantIndex === 0
        || module.transformedVariants[variantIndex - 1].sha256 < variant.sha256
        || (module.transformedVariants[variantIndex - 1].sha256 === variant.sha256
          && module.transformedVariants[variantIndex - 1].byteCount < variant.byteCount),
    );
    if (!variantsSorted) {
      reasons.push(`runtime module ${index} transformed variants are not unique and sorted`);
    }
    if (variantsValid && module.transformedVariants.reduce(
      (sum, variant) => sum + variant.responseCount,
      0,
    ) !== module.successfulResponseCount) {
      reasons.push(`runtime module ${index} response count is inconsistent`);
    }
    try {
      const contents = await readFile(absolutePath);
      if (contents.length !== module.sourceByteCount
        || createHash('sha256').update(contents).digest('hex') !== module.sourceSha256) {
        reasons.push(`runtime module ${relativePath} differs from its retained source commitment`);
      }
    } catch {
      reasons.push(`runtime module ${relativePath} is unavailable`);
    }
  }
  if (audit?.dependencyModuleCount !== dependencyModuleCount
    || dependencyModuleCount === 0) {
    reasons.push('runtime dependency-module count is absent or inconsistent');
  }
  for (const required of requiredModulePaths) {
    if (typeof required !== 'string'
      || required.includes('\\')
      || referencesOptimizedDependencyArtifact(required)
      || !modulePaths.includes(required)) {
      reasons.push(`required runtime module ${JSON.stringify(required)} was not observed`);
    }
  }
  let computedAggregate = null;
  try {
    computedAggregate = candidateViteRuntimeModulesSha256(auditModules);
  } catch {
    reasons.push('runtime module aggregate could not be reconstructed');
  }
  if (!SHA256_PATTERN.test(audit?.modulesSha256 ?? '')
    || audit.modulesSha256 !== computedAggregate) {
    reasons.push('runtime module aggregate digest is inconsistent');
  }
  return { pass: reasons.length === 0, reasons: [...new Set(reasons)] };
}

export async function createCandidateViteRuntimeGuard(projectRoot) {
  const root = path.resolve(projectRoot);
  const entryHtmlPath = path.join(root, 'index.html');
  const entryHtmlStats = await lstat(entryHtmlPath);
  if (!entryHtmlStats.isFile() || entryHtmlStats.isSymbolicLink()) {
    throw new Error('Candidate Vite entry HTML must be a regular non-symbolic-link file.');
  }
  const entryHtmlContents = await readFile(entryHtmlPath);
  if (entryHtmlContainsForbiddenRuntimeReference(entryHtmlContents)) {
    throw new Error('Candidate Vite entry HTML references a forbidden runtime artifact.');
  }
  const entryHtmlSha256 = createHash('sha256').update(entryHtmlContents).digest('hex');
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'three-webgpu-candidate-vite-'));
  const cacheDirectory = path.join(temporaryRoot, '.vite-candidate-cache');
  const [physicalRoot, physicalNodeModules, physicalTemporaryRoot] = await Promise.all([
    realpath(root),
    realpath(path.join(root, 'node_modules')),
    realpath(temporaryRoot),
  ]);
  if (pathInside(root, cacheDirectory)
    || pathInside(path.join(root, 'node_modules'), cacheDirectory)
    || pathInside(physicalRoot, physicalTemporaryRoot)
    || pathInside(physicalNodeModules, physicalTemporaryRoot)) {
    await rm(temporaryRoot, { recursive: true, force: true });
    throw new Error('Candidate Vite cache directory must be outside the project and node_modules.');
  }
  let cachePreexisted = true;
  try {
    await access(cacheDirectory);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    cachePreexisted = false;
  }
  if (cachePreexisted) throw new Error('Candidate Vite cache directory unexpectedly preexisted.');

  const modules = new Map();
  const requestedModulePaths = new Set();
  const servedVariantsByPath = new Map();
  const violations = [];
  let resolvedConfigurationObserved = false;
  let entryHtmlSuccessfulResponseCount = 0;
  let finalized = false;
  let disposed = false;

  function rejectOptimizedReference(code) {
    violations.push(code);
    throw new Error('Candidate Vite runtime rejected an excluded .vite* cache artifact.');
  }

  const plugin = {
    name: 'candidate-vite-runtime-module-audit',
    enforce: 'post',
    configResolved(configuration) {
      const exactConfiguration = path.resolve(configuration.root) === root
        && configuration.configFile === undefined
        && configuration.appType === 'custom'
        && path.resolve(configuration.cacheDir) === path.resolve(cacheDirectory)
        && configuration.optimizeDeps?.noDiscovery === true
        && Array.isArray(configuration.optimizeDeps?.include)
        && configuration.optimizeDeps.include.length === 0;
      if (!exactConfiguration) {
        violations.push('resolved-vite-configuration-mismatch');
        throw new Error('Candidate Vite runtime resolved outside the frozen configuration.');
      }
      resolvedConfigurationObserved = true;
    },
    configureServer(viteServer) {
      viteServer.middlewares.use((request, response, next) => {
        const requestPath = decodedRequestPath(
          String(request.url ?? ''),
        ).split(/[?#]/, 1)[0];
        if (requestPath.startsWith('/@vite/') || requestPath.startsWith('/@id/')) {
          violations.push('vite-internal-or-virtual-runtime-request');
          response.statusCode = 403;
          response.setHeader('content-type', 'text/plain; charset=utf-8');
          response.end('Candidate runtime forbids Vite-internal and virtual modules.');
          return;
        }
        if (!referencesOptimizedDependencyArtifact(request.url)) {
          const relativePath = sourceRelativePathFromRequest(request.url, root);
          if (relativePath === null) {
            if (requestPath.startsWith('/@')
              || AUDITABLE_MODULE_REQUEST_PATTERN.test(requestPath)) {
              violations.push('unmapped-runtime-module-request');
              response.statusCode = 403;
              response.setHeader('content-type', 'text/plain; charset=utf-8');
              response.end('Candidate runtime forbids unmapped module requests.');
              return;
            }
            next();
            return;
          }
          requestedModulePaths.add(relativePath);
          const chunks = [];
          const originalWrite = response.write;
          const originalEnd = response.end;
          response.write = function auditedWrite(chunk, encoding, callback) {
            const captured = responseChunkBuffer(chunk, encoding);
            if (captured !== null) chunks.push(captured);
            return Reflect.apply(originalWrite, this, arguments);
          };
          response.end = function auditedEnd(chunk, encoding, callback) {
            const captured = responseChunkBuffer(chunk, encoding);
            if (captured !== null) chunks.push(captured);
            return Reflect.apply(originalEnd, this, arguments);
          };
          response.once('finish', () => {
            if (response.statusCode < 200 || response.statusCode >= 300) {
              violations.push(
                `runtime-module-http-status:${relativePath}:${response.statusCode}`,
              );
              return;
            }
            const contents = Buffer.concat(chunks);
            const variant = {
              byteCount: contents.length,
              sha256: createHash('sha256').update(contents).digest('hex'),
              responseCount: 1,
            };
            const variants = servedVariantsByPath.get(relativePath) ?? [];
            const prior = variants.find(
              (entry) => entry.byteCount === variant.byteCount
                && entry.sha256 === variant.sha256,
            );
            if (prior === undefined) variants.push(variant);
            else prior.responseCount += 1;
            variants.sort((left, right) => (
              left.sha256.localeCompare(right.sha256) || left.byteCount - right.byteCount
            ));
            servedVariantsByPath.set(relativePath, variants);
          });
          next();
          return;
        }
        violations.push('optimized-dependency-http-request');
        response.statusCode = 403;
        response.setHeader('content-type', 'text/plain; charset=utf-8');
        response.end('Candidate runtime forbids excluded Vite cache artifacts.');
      });
      return () => {
        viteServer.middlewares.use((request, response, next) => {
          const requestPath = decodedRequestPath(
            String(request.url ?? ''),
          ).split(/[?#]/, 1)[0];
          if (request.method !== 'GET'
            || !['/', '/index.html'].includes(requestPath)) {
            next();
            return;
          }
          entryHtmlSuccessfulResponseCount += 1;
          response.statusCode = 200;
          response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          response.setHeader('content-type', 'text/html; charset=utf-8');
          response.setHeader('cache-control', 'no-store');
          response.setHeader('content-length', String(entryHtmlContents.length));
          response.end(entryHtmlContents);
        });
      };
    },
    async transform(code, identifier) {
      if (referencesOptimizedDependencyArtifact(identifier)) {
        rejectOptimizedReference('optimized-dependency-module-id');
      }
      if (referencesOptimizedDependencyArtifact(code)) {
        rejectOptimizedReference('optimized-dependency-transformed-reference');
      }
      const filename = normalizedModuleFilename(identifier);
      if (!path.isAbsolute(filename) || !pathInside(root, filename)) return null;
      const relativePath = path.relative(root, filename).split(path.sep).join('/');
      if (referencesOptimizedDependencyArtifact(relativePath)) {
        rejectOptimizedReference('optimized-dependency-source-path');
      }
      const contents = await readFile(filename);
      const sourceSha256 = createHash('sha256').update(contents).digest('hex');
      const prior = modules.get(relativePath);
      if (prior && (prior.sourceByteCount !== contents.length
        || prior.sourceSha256 !== sourceSha256)) {
        throw new Error(`Candidate runtime module changed while served: ${relativePath}`);
      }
      const record = prior ?? {
        sourceRelativePath: relativePath,
        sourceByteCount: contents.length,
        sourceSha256,
        successfulResponseCount: 0,
        transformedVariants: [],
      };
      modules.set(relativePath, record);
      return null;
    },
  };

  return {
    cacheDirectory,
    viteConfig: {
      appType: 'custom',
      cacheDir: cacheDirectory,
      optimizeDeps: { noDiscovery: true, include: [] },
      plugins: [plugin],
    },
    plugin,
    async finalize() {
      if (finalized) throw new Error('Candidate Vite runtime audit was already finalized.');
      finalized = true;
      for (const relativePath of requestedModulePaths) {
        const record = modules.get(relativePath);
        const variants = servedVariantsByPath.get(relativePath);
        if (record === undefined || variants === undefined || variants.length === 0) {
          violations.push(`incomplete-runtime-module-audit:${relativePath}`);
          continue;
        }
        record.transformedVariants = variants;
        record.successfulResponseCount = variants.reduce(
          (sum, variant) => sum + variant.responseCount,
          0,
        );
      }
      if (violations.length !== 0) {
        throw new Error(
          `Candidate Vite runtime observed ${violations.length} prohibited optimized artifact(s).`,
        );
      }
      if (!resolvedConfigurationObserved) {
        throw new Error('Candidate Vite runtime did not observe its resolved configuration.');
      }
      if (entryHtmlSuccessfulResponseCount < 1) {
        throw new Error('Candidate Vite runtime did not serve its exact entry HTML.');
      }
      const cache = await emptyDirectoryAudit(cacheDirectory);
      if (cache.entryCount !== 0) {
        throw new Error('Candidate Vite runtime created an unexpected cache entry.');
      }
      const moduleRecords = [...requestedModulePaths].map(
        (relativePath) => modules.get(relativePath),
      ).sort(
        (left, right) => comparePath(left.sourceRelativePath, right.sourceRelativePath),
      );
      const audit = {
        schemaVersion: 3,
        kind: 'candidate-vite-runtime-module-audit',
        policyId: CANDIDATE_VITE_RUNTIME_POLICY_ID,
        configuration: {
          configFile: false,
          appType: 'custom',
          optimizeDepsNoDiscovery: true,
          optimizeDepsInclude: [],
          entryHtmlPolicy: 'exact-tracked-index-html-without-vite-client',
          cacheDirectoryPolicy: 'unique-fresh-os-temporary-directory-outside-project',
          cachePreexisted,
        },
        cache,
        entryHtml: {
          sourceRelativePath: 'index.html',
          sourceByteCount: entryHtmlContents.length,
          sourceSha256: entryHtmlSha256,
          successfulResponseCount: entryHtmlSuccessfulResponseCount,
          responseHeaders: {
            crossOriginOpenerPolicy: 'same-origin',
            crossOriginEmbedderPolicy: 'require-corp',
            contentType: 'text/html; charset=utf-8',
            cacheControl: 'no-store',
          },
          responseVariants: [{
            byteCount: entryHtmlContents.length,
            sha256: entryHtmlSha256,
            responseCount: entryHtmlSuccessfulResponseCount,
          }],
        },
        prohibitedOptimizedArtifactCount: violations.length,
        requiredModulePaths: [...CANDIDATE_VITE_REQUIRED_MODULE_PATHS],
        moduleCount: moduleRecords.length,
        dependencyModuleCount: moduleRecords.filter(
          (record) => record.sourceRelativePath.startsWith('node_modules/'),
        ).length,
        modulesSha256: candidateViteRuntimeModulesSha256(moduleRecords),
        modules: moduleRecords,
      };
      const validation = await validateCandidateViteRuntimeAudit(audit, root);
      if (!validation.pass) {
        throw new Error(`Candidate Vite runtime audit failed: ${validation.reasons.join('; ')}`);
      }
      return audit;
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      if (!pathInside(os.tmpdir(), temporaryRoot)
        || path.resolve(temporaryRoot) === path.resolve(os.tmpdir())) {
        throw new Error('Refusing to remove an unsafe candidate Vite temporary directory.');
      }
      await rm(temporaryRoot, { recursive: true, force: true });
    },
  };
}

async function collectFiles(root, relativeDirectory, records) {
  const absoluteDirectory = path.join(root, ...relativeDirectory.split('/').filter(Boolean));
  const entries = await readdir(absoluteDirectory, { withFileTypes: true });
  entries.sort((left, right) => comparePath(left.name, right.name));
  for (const entry of entries) {
    if (relativeDirectory === '' && excludedTopLevel(entry.name)) continue;
    const relativePath = relativeDirectory === ''
      ? entry.name
      : `${relativeDirectory}/${entry.name}`;
    const absolutePath = path.join(root, ...relativePath.split('/'));
    if (entry.isSymbolicLink()) {
      throw new Error(
        `Execution dependency closure rejects symlink outside exclusions: ${relativePath}`,
      );
    }
    if (entry.isDirectory()) {
      await collectFiles(root, relativePath, records);
    } else if (entry.isFile()) {
      records.push({ relativePath, absolutePath });
    } else {
      throw new Error(
        `Execution dependency closure rejects unsupported entry: ${relativePath}`,
      );
    }
  }
}

export async function collectExecutionDependencyClosure(projectRoot) {
  const dependencyRoot = path.join(projectRoot, 'node_modules');
  const records = [];
  await collectFiles(dependencyRoot, '', records);
  records.sort((left, right) => comparePath(left.relativePath, right.relativePath));
  let cursor = 0;
  const workerCount = Math.min(32, Math.max(1, records.length));
  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (cursor < records.length) {
      const index = cursor;
      cursor += 1;
      const record = records[index];
      const contents = await readFile(record.absolutePath);
      record.bytes = contents.length;
      record.fileSha256 = createHash('sha256').update(contents).digest('hex');
    }
  }));
  const aggregate = createHash('sha256');
  aggregate.update(`${FORMAT}\0`);
  let totalBytes = 0;
  for (const record of records) {
    aggregate.update(`${Buffer.byteLength(record.relativePath)}:`);
    aggregate.update(record.relativePath);
    aggregate.update(`\0${record.bytes}:${record.fileSha256}\0`);
    totalBytes += record.bytes;
  }
  return {
    schemaVersion: 1,
    kind: 'installed-execution-dependency-closure',
    root: 'node_modules',
    format: FORMAT,
    hashAlgorithm: 'sha256',
    exclusions: [...EXCLUSIONS],
    fileCount: records.length,
    totalBytes,
    sha256: aggregate.digest('hex'),
  };
}

export function executionDependencyClosuresMatch(left, right) {
  return left?.schemaVersion === 1
    && right?.schemaVersion === 1
    && left.kind === right.kind
    && left.root === right.root
    && left.format === right.format
    && left.hashAlgorithm === right.hashAlgorithm
    && JSON.stringify(left.exclusions) === JSON.stringify(right.exclusions)
    && left.fileCount === right.fileCount
    && left.totalBytes === right.totalBytes
    && left.sha256 === right.sha256;
}

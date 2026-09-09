import { execFile as execFileCallback } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  chmod,
  mkdir,
  readFile,
  readdir,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidateRoot = path.join(
  repoRoot,
  '.local-research',
  'three-ort-gpu-buffer-current-dev-candidate',
);
const liveBuildRoot = path.join(
  repoRoot,
  '.generated',
  'three-ort-gpu-buffer-canary-v1-live-build',
);
const preflightBuildRoot = path.join(
  repoRoot,
  '.generated',
  'three-ort-gpu-buffer-canary-v1-preflight-build',
);
const protocolPath = path.join(repoRoot, 'protocols', 'three-ort-gpu-buffer-canary-v1.json');
const ortPackageRoot = path.join(repoRoot, 'node_modules', 'onnxruntime-web');

const CANDIDATE_COMMIT = '31fdc388bc2d001f9a7d7ca27f52ca2c411b1cdc';
const CANDIDATE_TREE = '8615edca00b47afe7fb32849182bd4a5ec3b687f';
const CANDIDATE_SUBJECT = 'WebGPU: Harden borrowed storage buffer adoption';
const ORT_VERSION = '1.29.0';
const GIT_COMMIT_PATTERN = /^[a-f0-9]{40}$/u;
const EXPECTED_BUILD_FILES = Object.freeze([
  'three.core.js',
  'three.module.js',
  'three.tsl.js',
  'three.webgpu.js',
  'three.webgpu.nodes.js',
]);
const ORT_RUNTIME_FILES = Object.freeze({
  'ort.webgpu.bundle.min.mjs': path.join('dist', 'ort.webgpu.bundle.min.mjs'),
  'ort-wasm-simd-threaded.asyncify.wasm': path.join(
    'dist',
    'ort-wasm-simd-threaded.asyncify.wasm',
  ),
});
const BUILD_ROUTE = '/__three_ort_bridge__';
const ORT_ROUTE = '/__ort_webgpu__';
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
});
const TEARDOWN_TIMEOUT_MS = 10_000;
const DEFAULT_PAGE_TIMEOUT_MS = 120_000;
const RUN_VECTORS = Object.freeze([
  Object.freeze({
    expectedBits: Object.freeze([
      0xbf400000, 0xbf400000,
      0x3f400000, 0xbf400000,
      0x00000000, 0x3f400000,
    ]),
    expectedOutput: Object.freeze([-0.75, -0.75, 0.75, -0.75, 0, 0.75]),
    input: Object.freeze([-0.75, -0.375, 0.25, -0.1875, 0, 0.125]),
  }),
  Object.freeze({
    expectedBits: Object.freeze([
      0xbf200000, 0xbf000000,
      0x3ec00000, 0xbf000000,
      0x3f200000, 0x3ec00000,
    ]),
    expectedOutput: Object.freeze([-0.625, -0.5, 0.375, -0.5, 0.625, 0.375]),
    input: Object.freeze([-0.625, -0.25, 0.125, -0.125, 0.125, 0.0625]),
  }),
]);

const execOptions = {
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
  windowsHide: true,
};

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonicalJson(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function serializeError(error) {
  return {
    message: String(error?.message ?? error).slice(0, 4_096),
    name: String(error?.name ?? 'Error').slice(0, 256),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 8_192),
  };
}

function exactArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function exists(target) {
  try {
    await stat(target);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function withDeadline(promise, timeoutMs, label) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(
      () => reject(new Error(`${label} exceeded ${timeoutMs} ms.`)),
      timeoutMs,
    );
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function pngDimensions(bytes) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  requireCondition(bytes.length >= 24, 'Screenshot is too small to be a PNG.');
  requireCondition(bytes.subarray(0, 8).equals(signature), 'Screenshot lacks a PNG signature.');
  requireCondition(bytes.toString('ascii', 12, 16) === 'IHDR', 'Screenshot lacks PNG IHDR.');
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

async function runGit(root, args) {
  return execFile(
    'git',
    [
      '-c',
      'core.autocrlf=false',
      '-c',
      'core.eol=lf',
      '-c',
      `safe.directory=${root}`,
      '-C',
      root,
      ...args,
    ],
    execOptions,
  );
}

async function gitSnapshot(root) {
  const [{ stdout: commit }, { stdout: tree }, { stdout: branch }, { stdout: status }] =
    await Promise.all([
      runGit(root, ['rev-parse', 'HEAD']),
      runGit(root, ['rev-parse', 'HEAD^{tree}']),
      runGit(root, ['branch', '--show-current']),
      runGit(root, ['status', '--porcelain=v1', '--untracked-files=all']),
    ]);
  return {
    branch: branch.trim(),
    commit: commit.trim(),
    status: status.trim(),
    tree: tree.trim(),
  };
}

async function readProtocol() {
  const bytes = await readFile(protocolPath);
  return { bytes, value: JSON.parse(bytes.toString('utf8')) };
}

function resolveRepoPath(relativePath, label) {
  requireCondition(typeof relativePath === 'string' && relativePath.length > 0,
    `${label} path is absent.`);
  const resolved = path.resolve(repoRoot, ...relativePath.replaceAll('\\', '/').split('/'));
  requireCondition(
    resolved.startsWith(`${repoRoot}${path.sep}`),
    `${label} path escapes the research repository.`,
  );
  return resolved;
}

async function validateCandidateIdentity(protocol) {
  const expected = protocol?.candidate ?? {};
  requireCondition(
    expected.commit === undefined || expected.commit === CANDIDATE_COMMIT,
    'Protocol candidate commit differs from the frozen canary commit.',
  );
  requireCondition(
    expected.tree === undefined || expected.tree === CANDIDATE_TREE,
    'Protocol candidate tree differs from the frozen canary tree.',
  );
  const snapshot = await gitSnapshot(candidateRoot);
  requireCondition(snapshot.commit === CANDIDATE_COMMIT, 'Candidate commit mismatch.');
  requireCondition(snapshot.tree === CANDIDATE_TREE, 'Candidate tree mismatch.');
  requireCondition(snapshot.status === '', 'Candidate repository is not clean.');
  const { stdout: subject } = await runGit(candidateRoot, ['show', '-s', '--format=%s', 'HEAD']);
  requireCondition(subject.trim() === CANDIDATE_SUBJECT, 'Candidate subject mismatch.');
  return snapshot;
}

export async function buildOrtGpuBufferCandidate(targetRoot) {
  requireCondition(
    await exists(targetRoot) === false,
    `Single-use build directory already exists: ${targetRoot}`,
  );
  const rollupModulePath = path.join(
    candidateRoot,
    'node_modules',
    'rollup',
    'dist',
    'es',
    'rollup.js',
  );
  requireCondition(await exists(rollupModulePath), 'Candidate Rollup dependency is not installed.');
  await mkdir(targetRoot, { recursive: true });
  const { rollup } = await import(pathToFileURL(rollupModulePath));
  const configurations = (await import(pathToFileURL(
    path.join(candidateRoot, 'utils', 'build', 'rollup.config.js'),
  ))).default;

  for (const configuration of configurations) {
    const input = typeof configuration.input === 'string'
      ? path.resolve(candidateRoot, configuration.input)
      : Object.fromEntries(Object.entries(configuration.input).map(([name, filename]) => (
        [name, path.resolve(candidateRoot, filename)]
      )));
    const bundle = await rollup({ ...configuration, input });
    try {
      const outputs = Array.isArray(configuration.output)
        ? configuration.output
        : [configuration.output];
      for (const output of outputs) {
        await bundle.write({ ...output, dir: targetRoot, file: undefined });
      }
    } finally {
      await bundle.close();
    }
  }

  const filenames = (await readdir(targetRoot)).sort();
  requireCondition(
    exactArray(filenames, [...EXPECTED_BUILD_FILES].sort()),
    `Candidate build emitted an unexpected inventory: ${filenames.join(', ')}.`,
  );
  const inventory = {};
  const modules = {};
  for (const filename of EXPECTED_BUILD_FILES) {
    const bytes = await readFile(path.join(targetRoot, filename));
    modules[filename] = bytes;
    inventory[filename] = { bytes: bytes.length, sha256: sha256(bytes) };
  }
  return { inventory, modules };
}

export function verifyBuiltOrtGpuBufferCandidate(built, expectedInventory = null) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  const webgpu = built?.modules?.['three.webgpu.js']?.toString('utf8') ?? '';
  const webgpuNodes = built?.modules?.['three.webgpu.nodes.js']?.toString('utf8') ?? '';
  const runtime = `${webgpu}\n${webgpuNodes}`;
  check(EXPECTED_BUILD_FILES.every((file) => built?.inventory?.[file] !== undefined),
    'Expected built module inventory is incomplete.');
  const expectedIsPinned = expectedInventory !== null
    && EXPECTED_BUILD_FILES.every((file) => (
      Number.isSafeInteger(expectedInventory?.[file]?.bytes)
        && /^[a-f0-9]{64}$/u.test(expectedInventory?.[file]?.sha256 ?? '')
    ));
  if (expectedIsPinned) {
    check(JSON.stringify(built?.inventory) === JSON.stringify(expectedInventory),
      'Built module inventory differs from the frozen offline preflight.');
  }
  check(webgpu.includes('GPUStorageBufferAttribute'),
    'Built Three.WebGPU.js lacks GPUStorageBufferAttribute.');
  check(webgpuNodes.includes('GPUStorageBufferAttribute'),
    'Built Three.WebGPU.Nodes.js lacks GPUStorageBufferAttribute.');
  check(runtime.includes('isGPUStorageBufferAttribute'),
    'Built runtime lacks borrowed storage-buffer type handling.');
  check(runtime.includes('bindingOffset') && runtime.includes('bindingSize'),
    'Built runtime lacks exact borrowed binding-range plumbing.');
  check(runtime.includes('ownsBuffer') && runtime.includes('false'),
    'Built runtime lacks borrowed ownership plumbing.');
  check(runtime.includes('declared GPUDevice does not match this renderer'),
    'Built runtime lacks shared-device identity validation.');
  return { inventoryPinned: expectedIsPinned, reasons, verified: reasons.length === 0 };
}

export async function readOrtRuntimeArtifacts() {
  const packageBytes = await readFile(path.join(ortPackageRoot, 'package.json'));
  const packageJson = JSON.parse(packageBytes.toString('utf8'));
  const lockBytes = await readFile(path.join(repoRoot, 'package-lock.json'));
  const lockJson = JSON.parse(lockBytes.toString('utf8'));
  const lockEntry = lockJson?.packages?.['node_modules/onnxruntime-web'];
  const files = {};
  const modules = {};
  for (const [filename, relativePath] of Object.entries(ORT_RUNTIME_FILES)) {
    const bytes = await readFile(path.join(ortPackageRoot, relativePath));
    files[filename] = { bytes: bytes.length, sha256: sha256(bytes) };
    modules[filename] = bytes;
  }
  return {
    files,
    lock: {
      bytes: lockBytes.length,
      integrity: lockEntry?.integrity ?? null,
      resolved: lockEntry?.resolved ?? null,
      sha256: sha256(lockBytes),
      version: lockEntry?.version ?? null,
    },
    modules,
    package: {
      bytes: packageBytes.length,
      name: packageJson.name,
      sha256: sha256(packageBytes),
      version: packageJson.version,
    },
  };
}

export function verifyOrtRuntimeArtifacts(runtime, expectedFiles = null, expectedPackage = null) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  check(runtime?.package?.name === 'onnxruntime-web', 'Runtime package name is not onnxruntime-web.');
  check(runtime?.package?.version === ORT_VERSION, `Runtime package is not exactly ${ORT_VERSION}.`);
  check(runtime?.lock?.version === ORT_VERSION,
    `package-lock.json does not resolve onnxruntime-web exactly ${ORT_VERSION}.`);
  if (expectedPackage !== null) {
    check(runtime?.lock?.integrity === expectedPackage.lockIntegrity,
      'package-lock.json ORT integrity differs from the protocol.');
  }
  check(Object.keys(ORT_RUNTIME_FILES).every((file) => runtime?.files?.[file] !== undefined),
    'ORT runtime artifact inventory is incomplete.');
  const expectedIsPinned = expectedFiles !== null
    && Object.keys(ORT_RUNTIME_FILES).every((file) => (
      Number.isSafeInteger(expectedFiles?.[file]?.bytes)
        && /^[a-f0-9]{64}$/u.test(expectedFiles?.[file]?.sha256 ?? '')
    ));
  if (expectedIsPinned) {
    check(JSON.stringify(runtime?.files) === JSON.stringify(expectedFiles),
      'ORT runtime artifacts differ from the protocol inventory.');
  }
  return { inventoryPinned: expectedIsPinned, reasons, verified: reasons.length === 0 };
}

export function findForbiddenArguments(arguments_, prefixes) {
  return arguments_.filter((argument) => {
    const normalized = String(argument).toLowerCase();
    return prefixes.some((prefix) => (
      normalized === String(prefix).toLowerCase()
        || normalized.startsWith(`${String(prefix).toLowerCase()}=`)
    ));
  });
}

async function captureBrowserArguments(browser) {
  try {
    const session = await browser.newBrowserCDPSession();
    const result = await session.send('Browser.getBrowserCommandLine');
    await session.detach();
    return { arguments: result.arguments ?? [], available: true };
  } catch (error) {
    return { arguments: [], available: false, error: serializeError(error) };
  }
}

/*
 * This function is serialized into the page by Playwright. Keep it entirely
 * self-contained: it deliberately closes over no Node-side values.
 *
 * Expected page/instrumentation interface:
 *   __setThreeOrtGpuBufferPhase('ort-run-1' | 'three-proof-1' | ...)
 *   __registerThreeOrtGpuBufferTarget(buffer, { logicalBytes: 24 }) -> numeric id
 *   __collectThreeOrtGpuBufferInstrumentation() -> one final serializable snapshot
 */
function installOrtGpuBufferInstrumentation() {
  const state = {
    acquisition: {
      adapterFeatures: [],
      adapterInfo: null,
      adapterRequests: [],
      deviceRequests: [],
    },
    device: {
      errorScopes: { pops: [], pushes: [] },
      pipelinePromiseRejections: [],
      uncapturedErrors: [],
      unexpectedLosses: [],
    },
    events: [],
    instrumentationFailures: [],
    kind: 'three-ort-gpu-buffer-instrumentation-v1',
    phase: 'bootstrap',
    phaseTransitions: [],
    resources: {
      adapters: [],
      bindGroupLayouts: [],
      bindGroups: [],
      buffers: [],
      commandBuffers: [],
      commandEncoders: [],
      devices: [],
      passes: [],
      pipelineLayouts: [],
      pipelines: [],
      queues: [],
      shaderModules: [],
    },
    setupError: null,
    target: null,
    unhandledRejections: [],
    windowErrors: [],
  };
  Object.defineProperty(window, '__threeOrtGpuBufferInstrumentation', { value: state });

  window.addEventListener('error', (event_) => {
    state.windowErrors.push({
      column: Number(event_.colno ?? 0),
      filename: String(event_.filename ?? ''),
      line: Number(event_.lineno ?? 0),
      message: String(event_.message ?? ''),
    });
  });
  window.addEventListener('unhandledrejection', (event_) => {
    state.unhandledRejections.push(String(event_.reason?.message ?? event_.reason));
  });

  const objectIds = new WeakMap();
  const objectKinds = new Map();
  const bufferOwners = new WeakMap();
  const bufferRecords = new WeakMap();
  const bindGroupResources = new WeakMap();
  const bindGroupLayoutRecords = new WeakMap();
  const deviceQueues = new Map();
  const instrumentedBuffers = new WeakSet();
  const instrumentedDevices = new WeakSet();
  const instrumentedPipelines = new WeakSet();
  const pendingErrorScopes = [];
  const scopeStacks = new WeakMap();
  const shaderModules = [];
  let nextObjectId = 1;
  let nextScopeId = 1;
  let nextSequence = 1;
  let collected = false;

  const idFor = (object, kind = 'object') => {
    if ((typeof object !== 'object' && typeof object !== 'function') || object === null) {
      return null;
    }
    let id = objectIds.get(object);
    if (id === undefined) {
      id = nextObjectId;
      nextObjectId += 1;
      objectIds.set(object, id);
      objectKinds.set(id, kind);
    }
    return id;
  };

  const emit = (method, details = {}) => {
    const record = {
      method,
      phase: state.phase,
      sequence: nextSequence,
      ...details,
    };
    nextSequence += 1;
    state.events.push(record);
    return record;
  };

  const numberOrNull = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };

  const simpleValue = (value, depth = 0) => {
    if (value === null || ['boolean', 'string'].includes(typeof value)) return value;
    if (typeof value === 'number') return Number.isFinite(value) ? value : String(value);
    if (typeof value === 'bigint') return String(value);
    if (depth >= 3) return value?.constructor?.name ?? typeof value;
    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
      return Array.from(value, (entry) => simpleValue(entry, depth + 1));
    }
    if (typeof value === 'object') {
      const result = {};
      for (const [key, entry] of Object.entries(value)) {
        if (entry === undefined || typeof entry === 'function') continue;
        result[key] = simpleValue(entry, depth + 1);
      }
      return result;
    }
    return String(value);
  };

  const replaceCallable = (target, name, factory, required = false) => {
    const original = target?.[name];
    if (typeof original !== 'function') {
      if (required) state.instrumentationFailures.push(`Missing callable ${name}.`);
      return false;
    }
    try {
      const replacement = factory(original.bind(target));
      Object.defineProperty(target, name, {
        configurable: true,
        value: replacement,
        writable: true,
      });
      const installed = target[name] === replacement;
      if (!installed && required) {
        state.instrumentationFailures.push(`Could not replace callable ${name}.`);
      }
      return installed;
    } catch (error) {
      if (required) {
        state.instrumentationFailures.push(
          `Could not replace callable ${name}: ${String(error?.message ?? error)}`,
        );
      }
      return false;
    }
  };

  const adapterInfo = (adapter) => {
    const info = adapter?.info;
    if (info === null || typeof info !== 'object') return null;
    const result = {};
    for (const key of [
      'vendor',
      'architecture',
      'device',
      'description',
      'backend',
      'type',
      'driver',
      'isFallbackAdapter',
    ]) {
      if (['string', 'boolean'].includes(typeof info[key])) result[key] = info[key];
    }
    return result;
  };

  const requestDescriptor = (descriptor) => {
    if (descriptor === undefined) return null;
    return {
      ...simpleValue(descriptor),
      requiredFeatures: descriptor.requiredFeatures === undefined
        ? []
        : Array.from(descriptor.requiredFeatures, String).sort(),
      requiredLimits: descriptor.requiredLimits === undefined
        ? {}
        : Object.fromEntries(
          Object.entries(descriptor.requiredLimits)
            .sort(([left], [right]) => left.localeCompare(right)),
        ),
    };
  };

  const bindingResource = (resource) => {
    if (resource !== null && typeof resource === 'object' && resource.buffer !== undefined) {
      return {
        bufferId: idFor(resource.buffer, 'buffer'),
        hasOwnOffset: Object.hasOwn(resource, 'offset'),
        hasOwnSize: Object.hasOwn(resource, 'size'),
        offset: Number(resource.offset ?? 0),
        size: resource.size === undefined ? null : Number(resource.size),
        type: 'buffer',
      };
    }
    return {
      objectId: idFor(resource, resource?.constructor?.name ?? 'binding-resource'),
      type: String(resource?.constructor?.name ?? typeof resource),
    };
  };

  const bindingEntries = (descriptor) => Array.from(
    descriptor?.entries ?? [],
    (entry) => ({
      binding: Number(entry.binding),
      resource: bindingResource(entry.resource),
    }),
  );

  const layoutEntries = (descriptor) => Array.from(
    descriptor?.entries ?? [],
    (entry) => ({
      binding: Number(entry.binding),
      buffer: entry.buffer === undefined ? null : simpleValue(entry.buffer),
      sampler: entry.sampler === undefined ? null : simpleValue(entry.sampler),
      storageTexture: entry.storageTexture === undefined
        ? null
        : simpleValue(entry.storageTexture),
      texture: entry.texture === undefined ? null : simpleValue(entry.texture),
      visibility: Number(entry.visibility ?? 0),
    }),
  );

  const bufferIdsInGroup = (bindGroup) => (
    (bindGroupResources.get(bindGroup) ?? [])
      .map((entry) => entry.resource?.bufferId)
      .filter((id) => Number.isSafeInteger(id))
  );

  const instrumentBuffer = (buffer, device, descriptor = null) => {
    if (instrumentedBuffers.has(buffer)) return bufferRecords.get(buffer);
    instrumentedBuffers.add(buffer);
    const bufferId = idFor(buffer, 'buffer');
    const deviceId = idFor(device, 'device');
    bufferOwners.set(buffer, device);
    const record = {
      bufferId,
      creationPhase: state.phase,
      creationSequence: nextSequence,
      deviceId,
      descriptor: descriptor === null ? null : {
        label: descriptor.label ?? null,
        mappedAtCreation: descriptor.mappedAtCreation === true,
        size: Number(descriptor.size),
        usage: Number(descriptor.usage),
      },
      label: String(buffer.label ?? descriptor?.label ?? ''),
      mapStateAtCreation: String(buffer.mapState ?? ''),
      size: Number(buffer.size),
      usage: Number(buffer.usage),
    };
    bufferRecords.set(buffer, record);
    state.resources.buffers.push(record);
    emit('createBuffer-result', { bufferId, deviceId });

    replaceCallable(buffer, 'mapAsync', (original) => (...args) => {
      emit('mapAsync', {
        bufferId,
        mode: Number(args[0]),
        offset: Number(args[1] ?? 0),
        size: args[2] === undefined ? null : Number(args[2]),
      });
      return original(...args);
    }, true);
    replaceCallable(buffer, 'getMappedRange', (original) => (...args) => {
      emit('getMappedRange', {
        bufferId,
        offset: Number(args[0] ?? 0),
        size: args[1] === undefined ? null : Number(args[1]),
      });
      return original(...args);
    }, true);
    replaceCallable(buffer, 'unmap', (original) => (...args) => {
      emit('unmap', { bufferId });
      return original(...args);
    }, true);
    replaceCallable(buffer, 'destroy', (original) => (...args) => {
      emit('destroyBuffer', { bufferId });
      return original(...args);
    }, true);
    return record;
  };

  const snapshotActiveBindings = (activeBindGroups) => {
    const groups = [...activeBindGroups.entries()]
      .sort(([left], [right]) => left - right)
      .map(([index, bindGroup]) => ({
        bindGroupId: idFor(bindGroup, 'bind-group'),
        bufferIds: bufferIdsInGroup(bindGroup),
        index,
      }));
    return {
      bindGroupIds: groups.map((group) => group.bindGroupId),
      boundBufferIds: [...new Set(groups.flatMap((group) => group.bufferIds))],
      groups,
    };
  };

  const numericArguments = (args) => args.map((value) => (
    typeof value === 'number' ? value : value?.constructor?.name ?? null
  ));

  const instrumentPass = (pass, kind, deviceId, commandEncoderId) => {
    const passId = idFor(pass, kind);
    const activeBindGroups = new Map();
    let activePipelineId = null;
    const record = {
      commandEncoderId,
      deviceId,
      kind,
      passId,
      phase: state.phase,
    };
    state.resources.passes.push(record);
    emit(`begin${kind === 'compute-pass' ? 'Compute' : 'Render'}Pass-result`, record);

    replaceCallable(pass, 'setBindGroup', (original) => (...args) => {
      const index = Number(args[0]);
      const bindGroup = args[1] ?? null;
      if (bindGroup === null) activeBindGroups.delete(index);
      else activeBindGroups.set(index, bindGroup);
      const groupResources = bindGroup === null ? [] : (bindGroupResources.get(bindGroup) ?? []);
      const dynamicOffsets = args[2] === undefined ? [] : Array.from(args[2], Number);
      const dynamicOffsetsDataStart = args[3] === undefined ? null : Number(args[3]);
      const dynamicOffsetsDataLength = args[4] === undefined ? null : Number(args[4]);
      const effectiveStart = dynamicOffsetsDataStart ?? 0;
      const effectiveLength = dynamicOffsetsDataLength ?? (dynamicOffsets.length - effectiveStart);
      emit('setBindGroup', {
        bindGroupId: idFor(bindGroup, 'bind-group'),
        bufferIds: groupResources
          .map((entry) => entry.resource?.bufferId)
          .filter((id) => Number.isSafeInteger(id)),
        commandEncoderId,
        deviceId,
        dynamicOffsets,
        dynamicOffsetsDataLength,
        dynamicOffsetsDataStart,
        effectiveDynamicOffsets: dynamicOffsets.slice(
          effectiveStart,
          effectiveStart + effectiveLength,
        ),
        index,
        passId,
      });
      return original(...args);
    }, true);
    replaceCallable(pass, 'setPipeline', (original) => (pipeline) => {
      activePipelineId = idFor(pipeline, 'pipeline');
      emit('setPipeline', {
        commandEncoderId,
        deviceId,
        passId,
        pipelineId: activePipelineId,
      });
      return original(pipeline);
    }, true);

    const directMethods = kind === 'compute-pass'
      ? ['dispatchWorkgroups']
      : ['draw', 'drawIndexed'];
    for (const method of directMethods) {
      replaceCallable(pass, method, (original) => (...args) => {
        emit(method, {
          ...snapshotActiveBindings(activeBindGroups),
          arguments: numericArguments(args),
          commandEncoderId,
          deviceId,
          passId,
          pipelineId: activePipelineId,
        });
        return original(...args);
      }, true);
    }
    if (kind === 'compute-pass') {
      replaceCallable(pass, 'dispatchWorkgroupsIndirect', (original) => (...args) => {
        emit('dispatchWorkgroupsIndirect', {
          ...snapshotActiveBindings(activeBindGroups),
          arguments: [idFor(args[0], 'buffer'), Number(args[1] ?? 0)],
          commandEncoderId,
          deviceId,
          indirectBufferId: idFor(args[0], 'buffer'),
          passId,
          pipelineId: activePipelineId,
        });
        return original(...args);
      }, true);
    } else {
      for (const method of ['drawIndirect', 'drawIndexedIndirect']) {
        replaceCallable(pass, method, (original) => (...args) => {
          emit(method, {
            ...snapshotActiveBindings(activeBindGroups),
            arguments: [idFor(args[0], 'buffer'), Number(args[1] ?? 0)],
            commandEncoderId,
            deviceId,
            indirectBufferId: idFor(args[0], 'buffer'),
            passId,
            pipelineId: activePipelineId,
          });
          return original(...args);
        }, true);
      }
    }
    if (kind !== 'compute-pass') {
      replaceCallable(pass, 'setVertexBuffer', (original) => (...args) => {
        emit('setVertexBuffer', {
          bufferId: idFor(args[1], 'buffer'),
          commandEncoderId,
          offset: Number(args[2] ?? 0),
          passId,
          size: args[3] === undefined ? null : Number(args[3]),
          slot: Number(args[0]),
        });
        return original(...args);
      }, true);
      replaceCallable(pass, 'setIndexBuffer', (original) => (...args) => {
        emit('setIndexBuffer', {
          bufferId: idFor(args[0], 'buffer'),
          commandEncoderId,
          format: String(args[1] ?? ''),
          offset: Number(args[2] ?? 0),
          passId,
          size: args[3] === undefined ? null : Number(args[3]),
        });
        return original(...args);
      }, true);
      replaceCallable(pass, 'executeBundles', (original) => (bundles) => {
        emit('executeBundles', {
          bundleIds: Array.from(bundles, (bundle) => idFor(bundle, 'render-bundle')),
          commandEncoderId,
          passId,
        });
        return original(bundles);
      }, kind === 'render-pass');
    }
    if (kind !== 'render-bundle') {
      replaceCallable(pass, 'end', (original) => (...args) => {
        emit('endPass', { commandEncoderId, passId });
        return original(...args);
      }, true);
    }
    return pass;
  };

  const instrumentCommandEncoder = (encoder, device) => {
    const commandEncoderId = idFor(encoder, 'command-encoder');
    const deviceId = idFor(device, 'device');
    state.resources.commandEncoders.push({
      commandEncoderId,
      deviceId,
      phase: state.phase,
    });

    replaceCallable(encoder, 'beginComputePass', (original) => (descriptor) => {
      emit('beginComputePass', {
        commandEncoderId,
        descriptor: simpleValue(descriptor ?? {}),
        deviceId,
      });
      return instrumentPass(
        original(descriptor),
        'compute-pass',
        deviceId,
        commandEncoderId,
      );
    }, true);
    replaceCallable(encoder, 'beginRenderPass', (original) => (descriptor) => {
      emit('beginRenderPass', {
        commandEncoderId,
        colorAttachmentCount: descriptor?.colorAttachments?.length ?? 0,
        deviceId,
        label: descriptor?.label ?? null,
      });
      return instrumentPass(
        original(descriptor),
        'render-pass',
        deviceId,
        commandEncoderId,
      );
    }, true);

    replaceCallable(encoder, 'copyBufferToBuffer', (original) => (...args) => {
      emit('copyBufferToBuffer', {
        commandEncoderId,
        destinationBufferId: idFor(args[2], 'buffer'),
        destinationOffset: Number(args[3]),
        size: Number(args[4]),
        sourceBufferId: idFor(args[0], 'buffer'),
        sourceOffset: Number(args[1]),
      });
      return original(...args);
    }, true);
    replaceCallable(encoder, 'copyBufferToTexture', (original) => (...args) => {
      emit('copyBufferToTexture', {
        commandEncoderId,
        destinationTextureId: idFor(args[1]?.texture, 'texture'),
        size: simpleValue(args[2]),
        sourceBufferId: idFor(args[0]?.buffer, 'buffer'),
        sourceOffset: Number(args[0]?.offset ?? 0),
      });
      return original(...args);
    }, true);
    replaceCallable(encoder, 'copyTextureToBuffer', (original) => (...args) => {
      emit('copyTextureToBuffer', {
        commandEncoderId,
        destinationBufferId: idFor(args[1]?.buffer, 'buffer'),
        destinationOffset: Number(args[1]?.offset ?? 0),
        size: simpleValue(args[2]),
        sourceTextureId: idFor(args[0]?.texture, 'texture'),
      });
      return original(...args);
    }, true);
    replaceCallable(encoder, 'clearBuffer', (original) => (...args) => {
      emit('clearBuffer', {
        bufferId: idFor(args[0], 'buffer'),
        commandEncoderId,
        offset: Number(args[1] ?? 0),
        size: args[2] === undefined ? null : Number(args[2]),
      });
      return original(...args);
    }, true);
    replaceCallable(encoder, 'resolveQuerySet', (original) => (...args) => {
      emit('resolveQuerySet', {
        commandEncoderId,
        destinationBufferId: idFor(args[3], 'buffer'),
        destinationOffset: Number(args[4]),
        firstQuery: Number(args[1]),
        queryCount: Number(args[2]),
      });
      return original(...args);
    });
    replaceCallable(encoder, 'finish', (original) => (...args) => {
      const commandBuffer = original(...args);
      const commandBufferId = idFor(commandBuffer, 'command-buffer');
      state.resources.commandBuffers.push({
        commandBufferId,
        commandEncoderId,
        deviceId,
        phase: state.phase,
      });
      emit('finishCommandEncoder', { commandBufferId, commandEncoderId, deviceId });
      return commandBuffer;
    }, true);
    return encoder;
  };

  const instrumentRenderBundleEncoder = (encoder, device) => {
    const commandEncoderId = idFor(encoder, 'render-bundle-encoder');
    const deviceId = idFor(device, 'device');
    instrumentPass(encoder, 'render-bundle', deviceId, commandEncoderId);
    replaceCallable(encoder, 'finish', (original) => (...args) => {
      const bundle = original(...args);
      emit('finishRenderBundle', {
        commandEncoderId,
        deviceId,
        renderBundleId: idFor(bundle, 'render-bundle'),
      });
      return bundle;
    }, true);
    return encoder;
  };

  const instrumentPipelineObject = (pipeline, record) => {
    if (pipeline === null || pipeline === undefined || instrumentedPipelines.has(pipeline)) return;
    instrumentedPipelines.add(pipeline);
    record.pipelineId = idFor(pipeline, 'pipeline');
    replaceCallable(pipeline, 'getBindGroupLayout', (original) => (index) => {
      const layout = original(index);
      const layoutId = idFor(layout, 'bind-group-layout');
      if (!bindGroupLayoutRecords.has(layout)) {
        const layoutRecord = {
          auto: true,
          bindGroupLayoutId: layoutId,
          index: Number(index),
          phase: state.phase,
          pipelineId: record.pipelineId,
        };
        bindGroupLayoutRecords.set(layout, layoutRecord);
        state.resources.bindGroupLayouts.push(layoutRecord);
      }
      emit('getBindGroupLayout', {
        bindGroupLayoutId: layoutId,
        index: Number(index),
        pipelineId: record.pipelineId,
      });
      return layout;
    });
  };

  const instrumentQueue = (queue, device) => {
    const queueId = idFor(queue, 'queue');
    const deviceId = idFor(device, 'device');
    if (deviceQueues.has(deviceId)) return;
    deviceQueues.set(deviceId, queue);
    state.resources.queues.push({ deviceId, queueId });

    replaceCallable(queue, 'writeBuffer', (original) => (...args) => {
      const data = args[2];
      const dataOffset = Number(args[3] ?? 0);
      let byteLength = null;
      if (args[4] !== undefined) {
        byteLength = Number(args[4]);
      } else if (ArrayBuffer.isView(data)) {
        byteLength = data.byteLength - dataOffset;
      } else if (data instanceof ArrayBuffer) {
        byteLength = data.byteLength - dataOffset;
      }
      emit('writeBuffer', {
        bufferId: idFor(args[0], 'buffer'),
        bufferOffset: Number(args[1]),
        byteLength,
        dataOffset,
        dataType: data?.constructor?.name ?? null,
        deviceId,
        queueId,
      });
      return original(...args);
    }, true);
    replaceCallable(queue, 'writeTexture', (original) => (...args) => {
      emit('writeTexture', {
        dataType: args[1]?.constructor?.name ?? null,
        deviceId,
        queueId,
        textureId: idFor(args[0]?.texture, 'texture'),
      });
      return original(...args);
    });
    replaceCallable(queue, 'copyExternalImageToTexture', (original) => (...args) => {
      emit('copyExternalImageToTexture', {
        deviceId,
        queueId,
        textureId: idFor(args[1]?.texture, 'texture'),
      });
      return original(...args);
    });
    replaceCallable(queue, 'submit', (original) => (commandBuffers) => {
      const commandBufferIds = Array.from(
        commandBuffers,
        (commandBuffer) => idFor(commandBuffer, 'command-buffer'),
      );
      emit('queueSubmit', { commandBufferIds, deviceId, queueId });
      return original(commandBuffers);
    }, true);
    replaceCallable(queue, 'onSubmittedWorkDone', (original) => (...args) => {
      emit('onSubmittedWorkDone', { deviceId, queueId });
      return original(...args);
    }, true);
  };

  const pipelineDescriptor = (descriptor, kind) => {
    const result = {
      label: descriptor?.label ?? null,
      layoutId: descriptor?.layout === 'auto'
        ? 'auto'
        : idFor(descriptor?.layout, 'pipeline-layout'),
    };
    if (kind === 'compute') {
      result.compute = {
        entryPoint: descriptor?.compute?.entryPoint ?? null,
        moduleId: idFor(descriptor?.compute?.module, 'shader-module'),
      };
    } else {
      result.fragment = descriptor?.fragment === undefined ? null : {
        entryPoint: descriptor.fragment.entryPoint ?? null,
        moduleId: idFor(descriptor.fragment.module, 'shader-module'),
      };
      result.vertex = descriptor?.vertex === undefined ? null : {
        entryPoint: descriptor.vertex.entryPoint ?? null,
        moduleId: idFor(descriptor.vertex.module, 'shader-module'),
      };
    }
    return result;
  };

  const instrumentDevice = (device) => {
    if (instrumentedDevices.has(device)) return device;
    instrumentedDevices.add(device);
    const deviceId = idFor(device, 'device');
    const record = {
      deviceId,
      features: Array.from(device.features, String).sort(),
      label: String(device.label ?? ''),
      limits: {
        maxStorageBufferBindingSize: numberOrNull(device.limits?.maxStorageBufferBindingSize),
        minStorageBufferOffsetAlignment: numberOrNull(
          device.limits?.minStorageBufferOffsetAlignment,
        ),
      },
      phase: state.phase,
    };
    state.resources.devices.push(record);
    emit('requestDevice-result', { deviceId });

    device.addEventListener?.('uncapturederror', (event_) => {
      event_.preventDefault();
      state.device.uncapturedErrors.push({
        deviceId,
        message: String(event_.error?.message ?? event_.error),
        name: String(event_.error?.constructor?.name ?? 'GPUError'),
        phase: state.phase,
        sequence: nextSequence,
      });
    });
    void device.lost.then((info) => {
      state.device.unexpectedLosses.push({
        deviceId,
        message: String(info.message ?? ''),
        phase: state.phase,
        reason: String(info.reason ?? ''),
        sequence: nextSequence,
      });
    });

    const scopeStack = [];
    scopeStacks.set(device, scopeStack);
    replaceCallable(device, 'pushErrorScope', (original) => (filter) => {
      const scope = {
        deviceId,
        filter: String(filter),
        id: nextScopeId,
        phase: state.phase,
        sequence: nextSequence,
      };
      nextScopeId += 1;
      scopeStack.push(scope);
      state.device.errorScopes.pushes.push(scope);
      emit('pushErrorScope', { deviceId, filter: scope.filter, scopeId: scope.id });
      return original(filter);
    }, true);
    replaceCallable(device, 'popErrorScope', (original) => () => {
      const pushed = scopeStack.pop() ?? null;
      const result = {
        deviceId,
        error: null,
        filter: pushed?.filter ?? null,
        id: pushed?.id ?? null,
        phase: state.phase,
        rejection: null,
        sequence: nextSequence,
        settled: false,
      };
      state.device.errorScopes.pops.push(result);
      emit('popErrorScope', { deviceId, scopeId: result.id });
      const promise = original();
      pendingErrorScopes.push(Promise.resolve(promise).then((error) => {
        result.error = error === null ? null : {
          message: String(error?.message ?? error),
          name: String(error?.constructor?.name ?? 'GPUError'),
        };
        result.settled = true;
      }, (error) => {
        result.rejection = {
          message: String(error?.message ?? error),
          name: String(error?.name ?? 'Error'),
        };
        result.settled = true;
      }));
      return promise;
    }, true);

    replaceCallable(device, 'createBuffer', (original) => (descriptor) => {
      const descriptorSnapshot = {
        label: descriptor?.label ?? null,
        mappedAtCreation: descriptor?.mappedAtCreation === true,
        size: Number(descriptor?.size),
        usage: Number(descriptor?.usage),
      };
      emit('createBuffer', { descriptor: descriptorSnapshot, deviceId });
      const buffer = original(descriptor);
      instrumentBuffer(buffer, device, descriptorSnapshot);
      return buffer;
    }, true);
    replaceCallable(device, 'createBindGroupLayout', (original) => (descriptor) => {
      const entries = layoutEntries(descriptor);
      const layout = original(descriptor);
      const bindGroupLayoutId = idFor(layout, 'bind-group-layout');
      const layoutRecord = {
        auto: false,
        bindGroupLayoutId,
        deviceId,
        entries,
        label: descriptor?.label ?? null,
        phase: state.phase,
      };
      bindGroupLayoutRecords.set(layout, layoutRecord);
      state.resources.bindGroupLayouts.push(layoutRecord);
      emit('createBindGroupLayout', { bindGroupLayoutId, deviceId, entries });
      return layout;
    }, true);
    replaceCallable(device, 'createPipelineLayout', (original) => (descriptor) => {
      const bindGroupLayoutIds = Array.from(
        descriptor?.bindGroupLayouts ?? [],
        (layout) => idFor(layout, 'bind-group-layout'),
      );
      const layout = original(descriptor);
      const pipelineLayoutId = idFor(layout, 'pipeline-layout');
      const layoutRecord = {
        bindGroupLayoutIds,
        deviceId,
        label: descriptor?.label ?? null,
        phase: state.phase,
        pipelineLayoutId,
      };
      state.resources.pipelineLayouts.push(layoutRecord);
      emit('createPipelineLayout', layoutRecord);
      return layout;
    }, true);
    replaceCallable(device, 'createBindGroup', (original) => (descriptor) => {
      // Three.js reuses and clears its descriptor entries, so this must happen
      // before invoking the native method.
      const entries = bindingEntries(descriptor);
      const bindGroup = original(descriptor);
      const bindGroupId = idFor(bindGroup, 'bind-group');
      bindGroupResources.set(bindGroup, entries);
      const groupRecord = {
        bindGroupId,
        deviceId,
        entries,
        label: descriptor?.label ?? null,
        layoutId: idFor(descriptor?.layout, 'bind-group-layout'),
        phase: state.phase,
      };
      state.resources.bindGroups.push(groupRecord);
      emit('createBindGroup', groupRecord);
      return bindGroup;
    }, true);
    replaceCallable(device, 'createShaderModule', (original) => (descriptor) => {
      const module = original(descriptor);
      const moduleId = idFor(module, 'shader-module');
      const moduleRecord = {
        code: String(descriptor?.code ?? ''),
        label: descriptor?.label ?? null,
        module,
        moduleId,
        phase: state.phase,
      };
      shaderModules.push(moduleRecord);
      state.resources.shaderModules.push({
        bytes: new TextEncoder().encode(moduleRecord.code).byteLength,
        label: moduleRecord.label,
        moduleId,
        phase: moduleRecord.phase,
      });
      emit('createShaderModule', {
        bytes: state.resources.shaderModules.at(-1).bytes,
        deviceId,
        moduleId,
      });
      return module;
    }, true);

    const instrumentPipelineCreation = (name, kind) => {
      replaceCallable(device, name, (original) => (descriptor) => {
        const pipelineRecord = {
          descriptor: pipelineDescriptor(descriptor, kind),
          deviceId,
          kind,
          method: name,
          phase: state.phase,
          pipelineId: null,
        };
        state.resources.pipelines.push(pipelineRecord);
        emit(name, { descriptor: pipelineRecord.descriptor, deviceId });
        const result = original(descriptor);
        if (result?.then instanceof Function) {
          void result.then((pipeline) => {
            instrumentPipelineObject(pipeline, pipelineRecord);
          }, (error) => {
            state.device.pipelinePromiseRejections.push({
              deviceId,
              message: String(error?.message ?? error),
              method: name,
              name: String(error?.name ?? 'Error'),
              phase: state.phase,
            });
          });
        } else {
          instrumentPipelineObject(result, pipelineRecord);
        }
        return result;
      }, true);
    };
    instrumentPipelineCreation('createComputePipeline', 'compute');
    instrumentPipelineCreation('createComputePipelineAsync', 'compute');
    instrumentPipelineCreation('createRenderPipeline', 'render');
    instrumentPipelineCreation('createRenderPipelineAsync', 'render');

    replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
      emit('createCommandEncoder', { deviceId, label: descriptor?.label ?? null });
      return instrumentCommandEncoder(original(descriptor), device);
    }, true);
    replaceCallable(device, 'createRenderBundleEncoder', (original) => (descriptor) => {
      emit('createRenderBundleEncoder', { deviceId, label: descriptor?.label ?? null });
      return instrumentRenderBundleEncoder(original(descriptor), device);
    }, true);
    replaceCallable(device, 'destroy', (original) => (...args) => {
      emit('destroyDevice', { deviceId });
      return original(...args);
    }, true);
    instrumentQueue(device.queue, device);
    return device;
  };

  try {
    const gpu = navigator.gpu;
    if (gpu === undefined) throw new Error('navigator.gpu is unavailable.');
    const requestAdapterInstalled = replaceCallable(
      gpu,
      'requestAdapter',
      (requestAdapter) => async (...args) => {
        const request = {
          argumentsCount: args.length,
          descriptor: requestDescriptor(args[0]),
          phase: state.phase,
        };
        state.acquisition.adapterRequests.push(request);
        emit('requestAdapter', request);
        const adapter = await requestAdapter(...args);
        if (adapter === null) return null;
        const adapterId = idFor(adapter, 'adapter');
        state.acquisition.adapterFeatures = Array.from(adapter.features, String).sort();
        state.acquisition.adapterInfo = adapterInfo(adapter);
        state.resources.adapters.push({
          adapterId,
          features: state.acquisition.adapterFeatures,
          info: state.acquisition.adapterInfo,
        });
        emit('requestAdapter-result', { adapterId });
        const requestDeviceInstalled = replaceCallable(
          adapter,
          'requestDevice',
          (requestDevice) => async (...deviceArgs) => {
            const deviceRequest = {
              adapterId,
              argumentsCount: deviceArgs.length,
              descriptor: requestDescriptor(deviceArgs[0]),
              phase: state.phase,
            };
            state.acquisition.deviceRequests.push(deviceRequest);
            emit('requestDevice', deviceRequest);
            const device = await requestDevice(...deviceArgs);
            instrumentDevice(device);
            return device;
          },
          true,
        );
        if (!requestDeviceInstalled) throw new Error('Could not instrument adapter.requestDevice.');
        return adapter;
      },
      true,
    );
    if (!requestAdapterInstalled) throw new Error('Could not instrument navigator.gpu.requestAdapter.');
  } catch (error) {
    state.setupError = {
      message: String(error?.message ?? error),
      name: String(error?.name ?? 'Error'),
    };
  }

  Object.defineProperty(window, '__setThreeOrtGpuBufferPhase', {
    value: (phase) => {
      if (typeof phase !== 'string' || phase.length === 0) {
        throw new TypeError('Instrumentation phase must be a non-empty string.');
      }
      const previous = state.phase;
      state.phase = phase;
      const transition = emit('phase', { from: previous, to: phase });
      state.phaseTransitions.push({
        from: previous,
        phase,
        sequence: transition.sequence,
      });
      return transition.sequence;
    },
  });
  Object.defineProperty(window, '__registerThreeOrtGpuBufferTarget', {
    value: (buffer, metadata = {}) => {
      const record = bufferRecords.get(buffer);
      if (record === undefined) {
        throw new Error('Target GPUBuffer was not created by the instrumented GPUDevice.');
      }
      if (state.target !== null && state.target.bufferId !== record.bufferId) {
        throw new Error('A different target GPUBuffer was already registered.');
      }
      if (state.target !== null) return state.target.bufferId;
      const registration = emit('registerTarget', {
        bufferId: record.bufferId,
        deviceId: record.deviceId,
      });
      state.target = {
        bufferId: record.bufferId,
        deviceId: record.deviceId,
        logicalBytes: Number(metadata.logicalBytes ?? metadata.byteLength ?? 0),
        metadata: simpleValue(metadata),
        physicalBytes: record.size,
        registrationPhase: state.phase,
        registrationSequence: registration.sequence,
      };
      return record.bufferId;
    },
  });
  Object.defineProperty(window, '__threeOrtGpuBufferObjectId', {
    value: (object) => idFor(object, object?.constructor?.name ?? 'object'),
  });

  Object.defineProperty(window, '__collectThreeOrtGpuBufferInstrumentation', {
    value: async () => {
      if (collected) throw new Error('ORT/Three instrumentation may only be collected once.');
      collected = true;
      await Promise.all(
        [...deviceQueues.values()].map((queue) => queue.onSubmittedWorkDone()),
      );
      await Promise.allSettled(pendingErrorScopes);
      await new Promise((resolve) => setTimeout(resolve, 50));
      const compilation = [];
      for (const moduleRecord of shaderModules) {
        try {
          const info = await moduleRecord.module.getCompilationInfo();
          compilation.push({
            code: moduleRecord.code,
            label: moduleRecord.label,
            messages: Array.from(info.messages, (message) => ({
              lineNum: Number(message.lineNum ?? 0),
              linePos: Number(message.linePos ?? 0),
              message: String(message.message),
              type: String(message.type),
            })),
            moduleId: moduleRecord.moduleId,
            phase: moduleRecord.phase,
          });
        } catch (error) {
          compilation.push({
            code: moduleRecord.code,
            label: moduleRecord.label,
            messages: [{
              message: String(error?.message ?? error),
              type: 'collection-error',
            }],
            moduleId: moduleRecord.moduleId,
            phase: moduleRecord.phase,
          });
        }
      }
      return structuredClone({
        ...state,
        collection: {
          finalObjectId: nextObjectId - 1,
          finalSequence: nextSequence - 1,
          objectKinds: Object.fromEntries(objectKinds),
        },
        shaderModules: compilation,
      });
    },
  });
}

function expectedTupleList(vector, sentinel) {
  return [0, 1, 2].map((instance) => [
    vector.expectedBits[instance * 2],
    vector.expectedBits[instance * 2 + 1],
    instance,
    sentinel,
  ]);
}

function sequenceIsStrict(records) {
  return Array.isArray(records)
    && records.every((record, index) => (
      Number.isSafeInteger(record?.sequence)
        && (index === 0 || records[index - 1].sequence < record.sequence)
    ));
}

function groupContainsExactTargetRange(group, targetId, logicalBytes) {
  return (group?.entries ?? []).some((entry) => (
    entry?.resource?.type === 'buffer'
      && entry.resource.bufferId === targetId
      && entry.resource.hasOwnOffset === true
      && entry.resource.hasOwnSize === true
      && entry.resource.offset === 0
      && entry.resource.size === logicalBytes
  ));
}

function groupContainsTarget(group, targetId) {
  return (group?.entries ?? []).some((entry) => (
    entry?.resource?.type === 'buffer' && entry.resource.bufferId === targetId
  ));
}

function commandUsesTarget(event, targetId) {
  return Array.isArray(event?.boundBufferIds) && event.boundBufferIds.includes(targetId);
}

export function reconstructActiveBindGroups(events, command, bindGroupById) {
  const slots = new Map();
  for (const event of events) {
    if (event?.sequence >= command?.sequence) break;
    if (event?.method !== 'setBindGroup' || event?.passId !== command?.passId) continue;
    if (event.bindGroupId === null) slots.delete(event.index);
    else slots.set(event.index, event);
  }
  return [...slots.entries()]
    .sort(([left], [right]) => left - right)
    .map(([index, setEvent]) => {
      const { bindGroupId } = setEvent;
      const group = bindGroupById.get(bindGroupId);
      const bufferIds = (group?.entries ?? [])
        .map((entry) => entry?.resource?.bufferId)
        .filter((id) => Number.isSafeInteger(id));
      return {
        bindGroupId,
        bufferIds,
        commandEncoderId: setEvent.commandEncoderId,
        deviceId: setEvent.deviceId,
        dynamicOffsets: setEvent.dynamicOffsets ?? null,
        dynamicOffsetsDataLength: setEvent.dynamicOffsetsDataLength ?? null,
        dynamicOffsetsDataStart: setEvent.dynamicOffsetsDataStart ?? null,
        effectiveDynamicOffsets: setEvent.effectiveDynamicOffsets ?? null,
        index,
        setSequence: setEvent.sequence,
      };
    });
}

function reconstructActivePipeline(events, command) {
  let pipeline = null;
  for (const event of events) {
    if (event?.sequence >= command?.sequence) break;
    if (event?.method === 'setPipeline' && event?.passId === command?.passId) pipeline = event;
  }
  return pipeline;
}

function submittedCommandClosure(events, command, targetDeviceId) {
  const finishes = events.filter((event) => (
    event.method === 'finishCommandEncoder'
      && event.commandEncoderId === command.commandEncoderId
      && event.deviceId === targetDeviceId
      && event.sequence > command.sequence
  ));
  if (finishes.length !== 1) return null;
  const finish = finishes[0];
  const submits = events.filter((event) => (
    event.method === 'queueSubmit'
      && event.deviceId === targetDeviceId
      && event.sequence > finish.sequence
      && event.commandBufferIds?.includes(finish.commandBufferId)
  ));
  if (submits.length !== 1) return null;
  return {
    commandBufferId: finish.commandBufferId,
    commandSequence: command.sequence,
    finishSequence: finish.sequence,
    queueId: submits[0].queueId,
    submitSequence: submits[0].sequence,
  };
}

function targetGroupHasNoDynamicOffsets(activeGroup) {
  return exactArray(activeGroup?.dynamicOffsets, [])
    && exactArray(activeGroup?.effectiveDynamicOffsets, [])
    && activeGroup?.dynamicOffsetsDataStart === null
    && activeGroup?.dynamicOffsetsDataLength === null;
}

function stripWgslComments(code) {
  let result = '';
  let blockDepth = 0;
  let inLineComment = false;
  for (let index = 0; index < code.length; index += 1) {
    const character = code[index];
    const next = code[index + 1];
    if (inLineComment) {
      if (character === '\n' || character === '\r') {
        inLineComment = false;
        result += character;
      } else {
        result += ' ';
      }
      continue;
    }
    if (blockDepth > 0) {
      if (character === '/' && next === '*') {
        blockDepth += 1;
        result += '  ';
        index += 1;
      } else if (character === '*' && next === '/') {
        blockDepth -= 1;
        result += '  ';
        index += 1;
      } else {
        result += character === '\n' || character === '\r' ? character : ' ';
      }
      continue;
    }
    if (character === '/' && next === '/') {
      inLineComment = true;
      result += '  ';
      index += 1;
    } else if (character === '/' && next === '*') {
      blockDepth = 1;
      result += '  ';
      index += 1;
    } else {
      result += character;
    }
  }
  return result;
}

function escapeRegularExpression(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

function identifierOccurrenceCount(source, identifier) {
  return [...source.matchAll(new RegExp(
    `(?<![A-Za-z0-9_])${escapeRegularExpression(identifier)}(?![A-Za-z0-9_])`,
    'gu',
  ))].length;
}

function readOnlyStorageBindingWitness(code, groupIndex, bindingIndex) {
  if (typeof code !== 'string') return null;
  const withoutComments = stripWgslComments(code);
  const binding = `@binding\\s*\\(\\s*${bindingIndex}\\s*\\)`;
  const group = `@group\\s*\\(\\s*${groupIndex}\\s*\\)`;
  const declaration = 'var\\s*<\\s*storage\\s*,\\s*read\\s*>\\s*'
    + '([A-Za-z_][A-Za-z0-9_]*)\\s*:';
  const match = new RegExp(
    `(?:${binding}\\s*${group}|${group}\\s*${binding})\\s*${declaration}`,
    'u',
  ).exec(withoutComments);
  if (match === null) return null;
  const variableIdentifier = match[1];
  const identifierOccurrences = identifierOccurrenceCount(
    withoutComments,
    variableIdentifier,
  );
  const postDeclarationOccurrences = identifierOccurrenceCount(
    withoutComments.slice(match.index + match[0].length),
    variableIdentifier,
  );
  if (identifierOccurrences < 3 || postDeclarationOccurrences < 2) return null;
  return {
    declaration: `@group(${groupIndex}) @binding(${bindingIndex}) `
      + `var<storage, read> ${variableIdentifier}`,
    identifierOccurrences,
    postDeclarationOccurrences,
    variableIdentifier,
  };
}

function readOnlyStorageWitnesses(
  command,
  activeGroups,
  instrumentation,
  targetId,
) {
  const pipelines = instrumentation?.resources?.pipelines ?? [];
  const pipelineLayouts = instrumentation?.resources?.pipelineLayouts ?? [];
  const bindGroupLayouts = instrumentation?.resources?.bindGroupLayouts ?? [];
  const bindGroups = instrumentation?.resources?.bindGroups ?? [];
  const shaderModules = instrumentation?.shaderModules ?? [];
  const pipeline = pipelines.find((record) => record.pipelineId === command.pipelineId);
  const pipelineLayout = pipelineLayouts.find((record) => (
    record.pipelineLayoutId === pipeline?.descriptor?.layoutId
  ));
  if (pipeline?.kind !== 'render' || pipelineLayout === undefined) return [];
  const witnesses = [];
  for (const activeGroup of activeGroups) {
    const bindGroup = bindGroups.find((record) => (
      record.bindGroupId === activeGroup.bindGroupId
    ));
    const targetEntries = (bindGroup?.entries ?? []).filter((entry) => (
      entry?.resource?.bufferId === targetId
    ));
    if (targetEntries.length === 0) continue;
    const expectedLayoutId = pipelineLayout.bindGroupLayoutIds?.[activeGroup.index];
    const bindGroupLayout = bindGroupLayouts.find((record) => (
      record.bindGroupLayoutId === bindGroup?.layoutId
    ));
    if (expectedLayoutId !== bindGroup?.layoutId || bindGroupLayout === undefined) continue;
    for (const targetEntry of targetEntries) {
      const layoutEntry = (bindGroupLayout.entries ?? []).find((entry) => (
        entry.binding === targetEntry.binding
      ));
      if (layoutEntry?.buffer?.type !== 'read-only-storage'
          || layoutEntry.buffer.hasDynamicOffset === true) continue;
      const stageModuleIds = [
        ['vertex', pipeline.descriptor?.vertex?.moduleId],
        ['fragment', pipeline.descriptor?.fragment?.moduleId],
      ];
      if (!stageModuleIds.every(([, moduleId]) => Number.isSafeInteger(moduleId))) continue;
      const matchingModules = stageModuleIds.map(([stage, moduleId]) => {
        const module = shaderModules.find((record) => record.moduleId === moduleId);
        const bindingWitness = readOnlyStorageBindingWitness(
          module?.code,
          activeGroup.index,
          targetEntry.binding,
        );
        if (bindingWitness === null) return null;
        return {
          bytes: Buffer.byteLength(module.code, 'utf8'),
          ...bindingWitness,
          moduleId,
          sha256: sha256(Buffer.from(module.code, 'utf8')),
          stage,
        };
      });
      if (matchingModules.some((module) => module === null)) continue;
      if (!exactArray(matchingModules.map((module) => module.stage), ['vertex', 'fragment'])) {
        continue;
      }
      witnesses.push({
        bindGroupId: bindGroup.bindGroupId,
        bindGroupLayoutId: bindGroup.layoutId,
        binding: targetEntry.binding,
        bufferType: layoutEntry.buffer.type,
        commandSequence: command.sequence,
        group: activeGroup.index,
        hasDynamicOffset: layoutEntry.buffer.hasDynamicOffset === true,
        modules: matchingModules,
        pipelineId: pipeline.pipelineId,
        pipelineLayoutId: pipelineLayout.pipelineLayoutId,
      });
    }
  }
  return witnesses;
}

function commandSnapshotMatchesActiveTrace(command, activeGroups) {
  const commandGroups = command?.groups ?? [];
  return exactArray(
    command?.bindGroupIds,
    activeGroups.map((group) => group.bindGroupId),
  )
    && exactArray(
      [...(command?.boundBufferIds ?? [])].sort((left, right) => left - right),
      [...new Set(activeGroups.flatMap((group) => group.bufferIds))]
        .sort((left, right) => left - right),
    )
    && commandGroups.length === activeGroups.length
    && commandGroups.every((group, index) => (
      group.index === activeGroups[index].index
        && group.bindGroupId === activeGroups[index].bindGroupId
        && exactArray(group.bufferIds, activeGroups[index].bufferIds)
    ));
}

function assignedFnBody(source, objectName, propertyName) {
  const object = escapeRegularExpression(objectName);
  const property = escapeRegularExpression(propertyName);
  return new RegExp(
    `${object}\\s*\\.\\s*${property}\\s*=\\s*Fn\\s*\\(\\s*\\(\\s*\\)\\s*=>\\s*`
      + '\\{([\\s\\S]*?)\\}\\s*\\)\\s*\\(\\s*\\)\\s*;',
    'u',
  ).exec(source)?.[1] ?? null;
}

function hasExactIndexedXyReads(body, storageIdentifier) {
  if (typeof body !== 'string') return false;
  const storage = escapeRegularExpression(storageIdentifier);
  return /const\s+base\s*=\s*instanceIndex\s*\.\s*mul\s*\(\s*uint\s*\(\s*2\s*\)\s*\)\s*;/u
    .test(body)
    && new RegExp(
      `const\\s+x\\s*=\\s*${storage}\\s*\\.\\s*element\\s*\\(\\s*base\\s*\\)`
        + '\\s*\\.\\s*toVar\\s*\\(',
      'u',
    ).test(body)
    && new RegExp(
      `const\\s+y\\s*=\\s*${storage}\\s*\\.\\s*element\\s*\\(`
        + '\\s*base\\s*\\.\\s*add\\s*\\(\\s*uint\\s*\\(\\s*1\\s*\\)\\s*\\)\\s*\\)'
        + '\\s*\\.\\s*toVar\\s*\\(',
      'u',
    ).test(body);
}

function storageXyDriveMarkerPosition(body) {
  if (typeof body !== 'string') return false;
  const positionComponent = (axis) => new RegExp(
    `positionGeometry\\s*\\.\\s*${axis}\\s*\\.\\s*mul\\s*\\(`
      + '\\s*float\\s*\\(\\s*ORT_MARKER_NDC_SIZE\\s*\\)\\s*\\)'
      + `\\s*\\.\\s*add\\s*\\(\\s*${axis}\\s*\\)`,
    'u',
  ).test(body);
  return positionComponent('x') && positionComponent('y');
}

function storageXyDriveExactBitOracle(body) {
  return typeof body === 'string'
    && /floatBitsToUint\s*\(\s*x\s*\)/u.test(body)
    && /floatBitsToUint\s*\(\s*y\s*\)/u.test(body)
    && /uint\s*\(\s*ORT_ORACLE_SENTINEL\s*\)/u.test(body);
}

function storageXyDriveVisualColor(body) {
  if (typeof body !== 'string') return false;
  const channel = (axis) => new RegExp(
    `${axis}\\s*\\.\\s*add\\s*\\(\\s*1\\s*\\)\\s*`
      + '\\.\\s*mul\\s*\\(\\s*0\\.5\\s*\\)',
    'u',
  ).test(body);
  return channel('x') && channel('y');
}

export function verifyOrtGpuBufferPageSource(source) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  check(typeof source === 'string' && source.length > 0, 'Canary page module is empty.');
  check(source.includes("from '/__ort_webgpu__/ort.webgpu.bundle.min.mjs'"),
    'Canary does not import the pinned local ORT WebGPU bundle route.');
  check(source.includes("from '/__three_ort_bridge__/three.webgpu.js'"),
    'Canary does not import the frozen local Three WebGPU bundle route.');
  check(source.includes("from '/__three_ort_bridge__/three.tsl.js'"),
    'Canary does not import the frozen local Three TSL bundle route.');
  check(source.includes('Tensor.fromGpuBuffer'),
    'Canary does not construct its output tensor from the target GPUBuffer.');
  check(source.includes('session.run'), 'Canary does not execute an ORT session.');
  check(source.includes('results.Y.gpuBuffer'),
    'Canary does not assert the returned ORT GPUBuffer identity.');
  const readOnlyStorageUses = source.match(
    /storage\s*\(\s*attribute\s*\)\s*\.\s*toReadOnly\s*\(\s*\)/gu,
  ) ?? [];
  check(readOnlyStorageUses.length === 2,
    'Canary must construct exactly two read-only target storage-node paths.');
  check(/const\s+values\s*=\s*storage\s*\(\s*attribute\s*\)\s*\.\s*toReadOnly\s*\(\s*\)\s*;/u
    .test(source), 'Canary exact oracle values are not sourced from read-only target storage.');
  check(/const\s+visualValues\s*=\s*storage\s*\(\s*attribute\s*\)\s*\.\s*toReadOnly\s*\(\s*\)\s*;/u
    .test(source), 'Canary visual values are not sourced from read-only target storage.');
  check(!/storage\s*\(\s*attribute\s*\)(?!\s*\.\s*toReadOnly\s*\()/gu.test(source),
    'Canary constructs a writable or unclassified target storage-node path.');
  check(!/\.\s*toReadWrite\s*\(/gu.test(source),
    'Canary source contains a writable storage-node path.');
  check(!/\.getData\s*\(/u.test(source),
    'Canary source calls Tensor.getData(), which would download GPU output.');
  check(!/outputTensor\s*\.\s*dispose\s*\(/u.test(source),
    'Canary source disposes the preallocated output tensor in the v1 proof.');
  check(!/(?:outputTensor|results\s*\.\s*Y)\s*\.\s*data\b/u.test(source),
    'Canary source accesses output tensor CPU data.');
  check(!source.includes('preferredOutputLocation'),
    'Canary uses preferredOutputLocation instead of the stable preallocated fetch path.');
  const oraclePositionBody = assignedFnBody(source, 'positionMaterial', 'positionNode');
  const oracleFragmentBody = assignedFnBody(source, 'positionMaterial', 'fragmentNode');
  const visualPositionBody = assignedFnBody(source, 'visualMaterial', 'positionNode');
  const visualFragmentBody = assignedFnBody(source, 'visualMaterial', 'fragmentNode');
  check(hasExactIndexedXyReads(oraclePositionBody, 'values')
      && storageXyDriveMarkerPosition(oraclePositionBody),
  'Canary exact oracle position path does not drive x/y from target elements base/base+1.');
  check(hasExactIndexedXyReads(oracleFragmentBody, 'values')
      && storageXyDriveExactBitOracle(oracleFragmentBody),
  'Canary exact oracle fragment path does not encode target x/y with floatBitsToUint.');
  check(hasExactIndexedXyReads(visualPositionBody, 'visualValues')
      && storageXyDriveMarkerPosition(visualPositionBody),
  'Canary visual position path does not drive x/y from target elements base/base+1.');
  check(hasExactIndexedXyReads(visualFragmentBody, 'visualValues')
      && storageXyDriveVisualColor(visualFragmentBody),
  'Canary visual fragment path does not derive its x/y channels from target storage.');
  return { reasons, verified: reasons.length === 0 };
}

export function summarizeTargetTrace(instrumentation) {
  const targetId = instrumentation?.target?.bufferId;
  const events = instrumentation?.events ?? [];
  const targetEvents = events.filter((event) => (
    event?.bufferId === targetId
      || event?.sourceBufferId === targetId
      || event?.destinationBufferId === targetId
      || event?.indirectBufferId === targetId
      || event?.bufferIds?.includes?.(targetId)
      || event?.boundBufferIds?.includes?.(targetId)
  ));
  return {
    targetId,
    targetEvents,
    byMethod: Object.fromEntries(
      [...new Set(targetEvents.map((event) => event.method))]
        .sort()
        .map((method) => [
          method,
          targetEvents.filter((event) => event.method === method).length,
        ]),
    ),
  };
}

export function validateOrtGpuBufferEvidence(evidence, protocol, observations = {}) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  const page = evidence?.page;
  const instrumentation = evidence?.instrumentation;
  const runs = page?.runs ?? [];
  const targetId = instrumentation?.target?.bufferId;
  const physicalBytes = protocol?.target?.physicalBytes ?? 32;
  const logicalBytes = protocol?.target?.logicalBytes ?? 24;
  const oracleWidth = protocol?.oracle?.width ?? 256;
  const oracleHeight = protocol?.oracle?.height ?? 256;
  const minimumPixels = protocol?.oracle?.minimumPixelsPerTuple ?? 300;
  const maximumPixels = protocol?.oracle?.maximumPixelsPerTuple ?? 800;
  const maximumCentroidError = protocol?.oracle?.maximumCentroidErrorPixels ?? 2;
  const sentinel = protocol?.oracle?.sentinel ?? 0x4f525433;
  const expectedRuns = protocol?.runs?.length === 2 ? protocol.runs : RUN_VECTORS;
  const pageKind = protocol?.page?.resultKind
    ?? 'three-webgpu-onnxruntime-borrowed-gpu-buffer-correctness-canary';

  check(page?.kind === pageKind, 'Page kind is incorrect.');
  check(page?.status === 'passed' && page?.error === null, 'Page did not pass.');
  check(page?.model?.verified === true, 'Embedded ONNX model was not verified.');
  check(page?.model?.actualBytes === protocol?.model?.bytes,
    'Embedded ONNX model byte length differs from the protocol.');
  check(page?.model?.actualSha256 === protocol?.model?.sha256,
    'Embedded ONNX model hash differs from the protocol.');
  check(page?.routes?.ortBundle === `${ORT_ROUTE}/ort.webgpu.bundle.min.mjs`,
    'Page ORT bundle route differs.');
  check(page?.routes?.ortWasm === `${ORT_ROUTE}/ort-wasm-simd-threaded.asyncify.wasm`,
    'Page ORT wasm route differs.');
  check(page?.routes?.threeBundle === `${BUILD_ROUTE}/three.webgpu.js`,
    'Page Three WebGPU route differs.');
  check(page?.routes?.threeTsl === `${BUILD_ROUTE}/three.tsl.js`,
    'Page Three TSL route differs.');

  const environment = page?.environment;
  check(environment?.navigatorGpu === true, 'Page did not expose WebGPU.');
  check(environment?.adapterAcquired === true && environment?.deviceAcquired === true,
    'Page did not acquire one WebGPU adapter and device.');
  check(environment?.ortProxy === protocol?.onnxRuntime?.proxy,
    'ORT proxy mode differs from the protocol.');
  check(environment?.ortThreads === protocol?.onnxRuntime?.numThreads,
    'ORT thread count differs from the protocol.');
  check(environment?.ortAdapterSame === true, 'ORT did not use the page adapter.');
  check(environment?.ortExecutionProviderDeviceSupplied === true
      && environment?.ortExecutionProviderDeviceSame === true,
  'ORT session creation did not receive the exact explicit GPUDevice.');
  check(environment?.ortDeviceSame === null || typeof environment?.ortDeviceSame === 'boolean',
    'ORT env.webgpu.device telemetry is malformed.');
  check(environment?.deviceLost === null, 'The page observed device loss.');
  check((environment?.uncapturedErrors?.length ?? 0) === 0,
    'The page observed an uncaptured GPU error.');
  check(environment?.instrumentation?.phaseHook === true
      && environment?.instrumentation?.targetHook === true
      && environment?.instrumentation?.collector === true,
  'The page did not find all mandatory instrumentation hooks.');
  check(environment?.instrumentation?.targetRegistrationResult === targetId,
    'Page target registration does not match the instrumented target ID.');

  const pageTarget = page?.target;
  check(pageTarget?.physicalBytes === physicalBytes, 'Page target physical size differs.');
  check(pageTarget?.logicalBytes === logicalBytes, 'Page target logical size differs.');
  check(pageTarget?.mapState === 'unmapped', 'Target buffer was not initially unmapped.');
  check(pageTarget?.outputTensorType === 'float32', 'Output tensor type is not float32.');
  check(exactArray(pageTarget?.outputTensorDims, protocol?.model?.dims ?? [3, 2]),
    'Output tensor dimensions differ.');
  check(pageTarget?.outputTensorBufferSame === true,
    'Output tensor does not expose the exact target GPUBuffer.');

  check(Array.isArray(runs) && runs.length === 2 && runs.every(Boolean),
    'Page does not contain exactly two completed runs.');
  for (let index = 0; index < 2; index += 1) {
    const run = runs[index];
    const expected = expectedRuns[index] ?? RUN_VECTORS[index];
    const expectedBits = expected.expectedBits ?? RUN_VECTORS[index].expectedBits;
    const expectedOutput = expected.expectedOutput ?? RUN_VECTORS[index].expectedOutput;
    const expectedInput = expected.input ?? RUN_VECTORS[index].input;
    const expectedTuples = expectedTupleList({ expectedBits }, sentinel);
    const oracle = run?.three?.oracle;

    check(run?.runIndex === index, `Run ${index + 1} index differs.`);
    check(exactArray(run?.input, expectedInput), `Run ${index + 1} input differs.`);
    check(exactArray(run?.expectedOutput, expectedOutput),
      `Run ${index + 1} expected float output differs.`);
    check(exactArray(run?.expectedBits, expectedBits),
      `Run ${index + 1} expected float bits differ.`);
    check(JSON.stringify(run?.expectedTuples) === JSON.stringify(expectedTuples),
      `Run ${index + 1} expected tuple oracle differs.`);
    check(run?.ort?.returnedSameTensor === true,
      `ORT run ${index + 1} did not return the supplied tensor.`);
    check(run?.ort?.returnedSameBuffer === true,
      `ORT run ${index + 1} did not return the target GPUBuffer.`);
    check(run?.ort?.type === 'float32', `ORT run ${index + 1} output type differs.`);
    check(exactArray(run?.ort?.dims, protocol?.model?.dims ?? [3, 2]),
      `ORT run ${index + 1} output dimensions differ.`);
    check(run?.ort?.usedPreallocatedFetch === true && run?.ort?.calledGetData === false,
      `ORT run ${index + 1} did not retain the stable preallocated no-download path.`);

    const three = run?.three;
    check(three?.rendererDeviceSame === true,
      `Three run ${index + 1} did not use the shared GPUDevice.`);
    check(three?.backendIsWebGPU === true, `Three run ${index + 1} was not WebGPU.`);
    check(three?.backendBufferSame === true,
      `Three run ${index + 1} backend did not cache the exact target object.`);
    check(three?.backendBindingOffset === protocol?.target?.byteOffset,
      `Three run ${index + 1} backend binding offset differs.`);
    check(three?.backendBindingSize === logicalBytes,
      `Three run ${index + 1} backend binding size differs.`);
    check(three?.backendOwnsBuffer === false,
      `Three run ${index + 1} incorrectly owns the target buffer.`);
    check(three?.attribute?.isGpuStorageBufferAttribute === true
        && three?.attribute?.isStorageBufferAttribute === true,
    `Three run ${index + 1} did not use GPUStorageBufferAttribute.`);
    check(three?.attribute?.itemSize === protocol?.target?.itemSize
        && three?.attribute?.count === protocol?.target?.count
        && three?.attribute?.byteOffset === protocol?.target?.byteOffset
        && three?.attribute?.byteLength === logicalBytes,
    `Three run ${index + 1} attribute range differs.`);
    check(three?.attribute?.storageOnly === true,
      `Three run ${index + 1} attribute was not storage-only.`);
    check(three?.disposed === true, `Three run ${index + 1} was not disposed.`);
    const release = three?.borrowedAttributeRelease;
    check(release?.rendererAttributesHadBefore === true
        && release?.backendHadBefore === true,
    `Three run ${index + 1} did not prove both borrowed-attribute caches before release.`);
    check(release?.rendererAttributesHasAfter === false
        && release?.backendHasAfter === false,
    `Three run ${index + 1} did not explicitly release both borrowed-attribute caches.`);
    check(release?.targetDestroyCountBeforeRendererDispose === 0
        && release?.targetDestroyCountAfterRendererDispose === 0,
    `Three run ${index + 1} destroyed the borrowed target during disposal.`);

    check(run?.validationScope === null, `Run ${index + 1} validation scope returned an error.`);
    check(oracle?.typedArray === protocol?.oracle?.typedArray,
      `Run ${index + 1} oracle is not Uint32Array.`);
    check(oracle?.width === oracleWidth && oracle?.height === oracleHeight,
      `Run ${index + 1} oracle dimensions differ.`);
    check(oracle?.byteLength === oracleWidth * oracleHeight * 16,
      `Run ${index + 1} oracle byte length differs.`);
    check(oracle?.minimumPixelsPerMarker === minimumPixels
        && oracle?.maximumPixelsPerMarker === maximumPixels
        && oracle?.spatialTolerancePixels === maximumCentroidError,
    `Run ${index + 1} oracle thresholds differ.`);
    check(JSON.stringify(oracle?.expectedPixelTuples) === JSON.stringify(expectedTuples),
      `Run ${index + 1} oracle expected tuples differ.`);
    check(oracle?.exact === true && (oracle?.unexpected?.length ?? -1) === 0,
      `Run ${index + 1} oracle contains missing or unexpected tuples.`);
    check(oracle?.backgroundPixels > 0, `Run ${index + 1} oracle lacks background pixels.`);
    check((oracle?.spatialMismatches?.length ?? -1) === 0,
      `Run ${index + 1} oracle has a spatial mismatch.`);
    check(Array.isArray(oracle?.tuples) && oracle.tuples.length === 3
        && oracle.tuples.every((tuple) => (
          tuple?.sufficient === true
            && tuple?.notOversized === true
            && tuple?.spatialMismatch === false
            && tuple?.count >= minimumPixels
            && tuple?.count <= maximumPixels
            && tuple?.centroidError?.x <= maximumCentroidError
            && tuple?.centroidError?.y <= maximumCentroidError
        )),
    `Run ${index + 1} tuple coverage or centroid proof differs.`);
    check(/^[a-f0-9]{64}$/u.test(oracle?.readbackSha256 ?? ''),
      `Run ${index + 1} readback hash is absent.`);
  }
  check(runs?.[0]?.three?.oracle?.readbackSha256
      !== runs?.[1]?.three?.oracle?.readbackSha256,
  'The two distinct output runs produced the same readback hash.');

  const lifecycle = page?.lifecycle;
  check(lifecycle?.screenshotReady === true && lifecycle?.continuationCalls === 1,
    'Screenshot continuation handshake was not exact.');
  check(lifecycle?.firstThreeDisposed === true && lifecycle?.secondThreeDisposed === true,
    'Both Three lifecycles did not complete disposal.');
  check(lifecycle?.targetReusedByOrt === true && lifecycle?.targetReusedByThree === true,
    'The target was not reused by both ORT and a fresh Three renderer.');
  check(lifecycle?.ortReleasedAfterThree === true,
    'ORT was not released after both Three lifecycles.');
  check(lifecycle?.outputTensorDisposed === false,
    'The stable v1 proof disposed its preallocated output tensor.');
  check(lifecycle?.ownerDestroyedTarget === true,
    'The external owner did not explicitly destroy the target.');
  check(lifecycle?.targetDestroyCountBeforeOwnerCleanup === 0
      && lifecycle?.targetDestroyCountAfterOwnerCleanup === 1,
  'Lifecycle did not prove zero early destroys followed by one owner destroy.');
  check((lifecycle?.cleanupErrors?.length ?? -1) === 0,
    'Page lifecycle cleanup recorded errors.');

  check(instrumentation?.kind === 'three-ort-gpu-buffer-instrumentation-v1',
    'Instrumentation kind is incorrect.');
  check(instrumentation?.setupError === null, 'GPU instrumentation setup failed.');
  check((instrumentation?.instrumentationFailures?.length ?? -1) === 0,
    'One or more required WebGPU hooks could not be installed.');
  check(Number.isSafeInteger(targetId) && targetId > 0,
    'Instrumentation target ID is absent.');
  check(instrumentation?.target?.physicalBytes === physicalBytes
      && instrumentation?.target?.logicalBytes === logicalBytes,
  'Instrumented target range differs.');
  check((instrumentation?.resources?.devices?.length ?? 0) >= 1,
    'No GPUDevice was instrumented.');
  check((instrumentation?.acquisition?.adapterRequests?.length ?? 0) >= 1,
    'No adapter request was observed.');
  check((instrumentation?.acquisition?.deviceRequests?.length ?? 0) >= 1,
    'No device request was observed.');
  check((instrumentation?.resources?.devices ?? []).some((device) => (
    device.deviceId === instrumentation?.target?.deviceId
  )), 'Target GPUDevice is absent from the instrumented device inventory.');
  const targetCreation = (instrumentation?.resources?.buffers ?? [])
    .find((record) => record.bufferId === targetId);
  check(targetCreation?.size === physicalBytes
      && targetCreation?.descriptor?.size === physicalBytes,
  'Target allocation is not exactly 32 bytes.');
  check(targetCreation?.mapStateAtCreation === 'unmapped'
      && targetCreation?.descriptor?.mappedAtCreation === false,
  'Target allocation began mapped.');
  check(targetCreation?.usage === 0x8c && targetCreation?.descriptor?.usage === 0x8c,
    'Target usage is not exactly STORAGE | COPY_SRC | COPY_DST.');

  const events = instrumentation?.events ?? [];
  check(sequenceIsStrict(events), 'Instrumentation event sequence is not strictly increasing.');
  const expectedPhases = [
    'ort-run-1',
    'three-proof-1',
    'screenshot-ready',
    'three-dispose-1',
    'ort-run-2',
    'three-proof-2',
    'three-dispose-2',
    'ort-cleanup',
    'owner-cleanup',
  ];
  const transitions = instrumentation?.phaseTransitions ?? [];
  check(sequenceIsStrict(transitions), 'Instrumentation phase sequence is not strictly increasing.');
  check(exactArray(transitions.map((transition) => transition.phase), expectedPhases),
    'Critical ORT/Three/lifecycle phase order differs.');

  const bindGroups = instrumentation?.resources?.bindGroups ?? [];
  const bindGroupById = new Map(bindGroups.map((group) => [group.bindGroupId, group]));
  const executionProof = { ort: [], three: [] };
  const activeGroupsFor = (command) => reconstructActiveBindGroups(
    events,
    command,
    bindGroupById,
  );
  const hasActivelyBoundTarget = (command) => activeGroupsFor(command).some((group) => (
    group.bufferIds.includes(targetId)
      && groupContainsTarget(bindGroupById.get(group.bindGroupId), targetId)
  ));
  const targetDispatchesEverywhere = events.filter((event) => (
    ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect'].includes(event.method)
      && hasActivelyBoundTarget(event)
  ));
  const targetDrawsEverywhere = events.filter((event) => (
    ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect'].includes(event.method)
      && hasActivelyBoundTarget(event)
  ));
  check(targetDispatchesEverywhere.every((dispatch) => (
    ['ort-run-1', 'ort-run-2'].includes(dispatch.phase)
  )), 'A target-active compute dispatch occurred outside its allowed phase (ORT run 1/2).');
  check(targetDrawsEverywhere.every((draw) => (
    ['three-proof-1', 'three-proof-2'].includes(draw.phase)
  )), 'A target-active draw occurred outside its allowed phase (Three proof 1/2).');

  for (let runIndex = 1; runIndex <= 2; runIndex += 1) {
    const ortPhase = `ort-run-${runIndex}`;
    const dispatches = events.filter((event) => (
      event.phase === ortPhase
        && ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect'].includes(event.method)
    ));
    const targetDispatches = dispatches.filter((event) => (
      commandUsesTarget(event, targetId)
        && event.deviceId === instrumentation?.target?.deviceId
    ));
    check(dispatches.length > 0, `ORT run ${runIndex} issued no compute dispatch.`);
    check(targetDispatches.length >= 1,
      `ORT run ${runIndex} issued no dispatch with the target actively bound.`);
    const ortProofs = targetDispatches.map((dispatch) => {
      const activeGroups = activeGroupsFor(dispatch);
      const pipelineEvent = reconstructActivePipeline(events, dispatch);
      const pipeline = (instrumentation?.resources?.pipelines ?? []).find((record) => (
        record.pipelineId === pipelineEvent?.pipelineId
      ));
      const submission = submittedCommandClosure(
        events,
        dispatch,
        instrumentation?.target?.deviceId,
      );
      const activeTarget = activeGroups.some((activeGroup) => (
        activeGroup.bufferIds?.includes(targetId)
          && activeGroup.deviceId === instrumentation?.target?.deviceId
          && activeGroup.commandEncoderId === dispatch.commandEncoderId
          && groupContainsTarget(bindGroupById.get(activeGroup.bindGroupId), targetId)
          && bindGroupById.get(activeGroup.bindGroupId)?.deviceId
            === instrumentation?.target?.deviceId
      ));
      return {
        activeTarget,
        commandSequence: dispatch.sequence,
        pipeline: pipeline === undefined ? null : {
          deviceId: pipeline.deviceId,
          kind: pipeline.kind,
          pipelineId: pipeline.pipelineId,
          setSequence: pipelineEvent?.sequence ?? null,
        },
        pipelineMatchesActiveTrace: dispatch.pipelineId === pipelineEvent?.pipelineId
          && pipelineEvent?.deviceId === instrumentation?.target?.deviceId
          && pipelineEvent?.commandEncoderId === dispatch.commandEncoderId,
        snapshotMatchesActiveTrace: commandSnapshotMatchesActiveTrace(dispatch, activeGroups),
        submission,
      };
    });
    check(ortProofs.every((proof) => proof.activeTarget),
      `ORT run ${runIndex} did not retain an active target binding at dispatch.`);
    check(ortProofs.every((proof) => proof.snapshotMatchesActiveTrace),
      `ORT run ${runIndex} command binding snapshot differs from its active bind state.`);
    check(ortProofs.every((proof) => proof.pipelineMatchesActiveTrace),
      `ORT run ${runIndex} command pipeline snapshot differs from its active pipeline.`);
    check(ortProofs.every((proof) => (
      proof.pipeline?.kind === 'compute'
        && proof.pipeline.deviceId === instrumentation?.target?.deviceId
        && proof.pipeline.setSequence < proof.commandSequence
    )), `ORT run ${runIndex} lacks a current target-device compute pipeline.`);
    check(ortProofs.every((proof) => proof.submission !== null),
      `ORT run ${runIndex} decisive dispatch was not finished and submitted.`);
    executionProof.ort.push({
      allDispatchCount: dispatches.length,
      proofs: ortProofs,
      targetDispatchCount: targetDispatches.length,
    });

    const threePhase = `three-proof-${runIndex}`;
    const draws = events.filter((event) => (
      event.phase === threePhase
        && ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect'].includes(event.method)
    ));
    const targetDraws = draws.filter((event) => (
      commandUsesTarget(event, targetId)
        && event.deviceId === instrumentation?.target?.deviceId
    ));
    check(draws.length > 0, `Three proof ${runIndex} issued no draw.`);
    check(targetDraws.length === draws.length,
      `A Three proof ${runIndex} draw lacked the target buffer binding.`);
    const threeProofs = targetDraws.map((draw) => {
      const activeGroups = activeGroupsFor(draw);
      const pipelineEvent = reconstructActivePipeline(events, draw);
      const pipeline = (instrumentation?.resources?.pipelines ?? []).find((record) => (
        record.pipelineId === pipelineEvent?.pipelineId
      ));
      const targetGroups = activeGroups.filter((activeGroup) => (
        activeGroup.bufferIds?.includes(targetId)
          && activeGroup.deviceId === instrumentation?.target?.deviceId
          && activeGroup.commandEncoderId === draw.commandEncoderId
          && groupContainsExactTargetRange(
            bindGroupById.get(activeGroup.bindGroupId),
            targetId,
            logicalBytes,
          )
      ));
      const readOnlyWitness = readOnlyStorageWitnesses(
        draw,
        activeGroups,
        instrumentation,
        targetId,
      );
      return {
        commandSequence: draw.sequence,
        dynamicOffsetsAbsent: targetGroups.every(targetGroupHasNoDynamicOffsets),
        exactTargetRange: targetGroups.length > 0,
        pipeline: pipeline === undefined ? null : {
          deviceId: pipeline.deviceId,
          kind: pipeline.kind,
          pipelineId: pipeline.pipelineId,
          setSequence: pipelineEvent?.sequence ?? null,
        },
        pipelineMatchesActiveTrace: draw.pipelineId === pipelineEvent?.pipelineId
          && pipelineEvent?.deviceId === instrumentation?.target?.deviceId
          && pipelineEvent?.commandEncoderId === draw.commandEncoderId,
        readOnlyStorageWitness: readOnlyWitness,
        snapshotMatchesActiveTrace: commandSnapshotMatchesActiveTrace(draw, activeGroups),
        submission: submittedCommandClosure(
          events,
          draw,
          instrumentation?.target?.deviceId,
        ),
      };
    });
    check(threeProofs.every((proof) => proof.snapshotMatchesActiveTrace),
      `Three proof ${runIndex} command binding snapshot differs from its active bind state.`);
    check(threeProofs.every((proof) => proof.pipelineMatchesActiveTrace),
      `Three proof ${runIndex} command pipeline snapshot differs from its active pipeline.`);
    check(threeProofs.every((proof) => proof.exactTargetRange),
      `Three proof ${runIndex} did not explicitly bind target range {offset:0,size:24}.`);
    check(threeProofs.every((proof) => proof.dynamicOffsetsAbsent),
      `Three proof ${runIndex} used a dynamic offset with the exact target range.`);
    check(threeProofs.every((proof) => (
      proof.pipeline?.kind === 'render'
        && proof.pipeline.deviceId === instrumentation?.target?.deviceId
        && proof.pipeline.setSequence < proof.commandSequence
    )), `Three proof ${runIndex} lacks a current target-device render pipeline.`);
    check(threeProofs.every((proof) => proof.submission !== null),
      `Three proof ${runIndex} decisive draw was not finished and submitted.`);
    check(threeProofs.every((proof) => proof.readOnlyStorageWitness.length > 0),
      `Three proof ${runIndex} lacks a dynamic read-only storage WGSL/layout witness.`);
    executionProof.three.push({ proofs: threeProofs, targetDrawCount: targetDraws.length });
  }

  const targetWrites = events.filter((event) => (
    event.method === 'writeBuffer' && event.bufferId === targetId
  ));
  const targetMaps = events.filter((event) => (
    ['mapAsync', 'getMappedRange', 'unmap'].includes(event.method)
      && event.bufferId === targetId
  ));
  const targetCopies = events.filter((event) => (
    ['copyBufferToBuffer', 'copyBufferToTexture', 'copyTextureToBuffer', 'resolveQuerySet']
      .includes(event.method)
      && (event.sourceBufferId === targetId || event.destinationBufferId === targetId)
  ));
  const targetClears = events.filter((event) => (
    event.method === 'clearBuffer' && event.bufferId === targetId
  ));
  const targetVertexOrIndex = events.filter((event) => (
    ['setVertexBuffer', 'setIndexBuffer'].includes(event.method) && event.bufferId === targetId
  ));
  const targetIndirect = events.filter((event) => (
    ['dispatchWorkgroupsIndirect', 'drawIndirect', 'drawIndexedIndirect'].includes(event.method)
      && event.indirectBufferId === targetId
  ));
  const targetDestroys = events.filter((event) => (
    event.method === 'destroyBuffer' && event.bufferId === targetId
  ));
  check(targetWrites.length === 0, 'Target buffer received a CPU queue.writeBuffer upload.');
  check(targetMaps.length === 0, 'Target buffer was mapped or exposed as a mapped range.');
  check(targetCopies.length === 0, 'A GPU copy or query resolve involved the target buffer.');
  check(targetClears.length === 0, 'Target buffer was explicitly cleared.');
  check(targetVertexOrIndex.length === 0, 'Target buffer was used as vertex or index input.');
  check(targetIndirect.length === 0, 'Target buffer was used for an indirect command.');
  check(targetDestroys.length === 1 && targetDestroys[0]?.phase === 'owner-cleanup',
    'Target was not destroyed exactly once by its owner during owner-cleanup.');
  const ownerCleanupTransition = transitions.find((transition) => (
    transition.phase === 'owner-cleanup'
  ));
  check(targetDestroys[0]?.sequence > ownerCleanupTransition?.sequence,
    'Target destruction preceded the explicit owner-cleanup phase.');
  check(events.filter((event) => (
    event.method === 'destroyDevice'
      && event.deviceId === instrumentation?.target?.deviceId
  )).length === 0, 'The externally supplied target GPUDevice was destroyed.');

  const pushes = instrumentation?.device?.errorScopes?.pushes ?? [];
  const pops = instrumentation?.device?.errorScopes?.pops ?? [];
  check(pushes.length > 0 && pushes.some((record) => record.filter === 'validation'),
    'No validation error scope surrounded the proofs.');
  check(pushes.length === pops.length, 'GPU error scopes are unbalanced.');
  check(pops.every((record) => (
    record.settled === true && record.error === null && record.rejection === null
  )), 'A GPU error scope returned an error.');
  check((instrumentation?.device?.pipelinePromiseRejections?.length ?? -1) === 0,
    'A GPU pipeline promise rejected.');
  check((instrumentation?.device?.uncapturedErrors?.length ?? -1) === 0,
    'Instrumentation observed an uncaptured GPU error.');
  check((instrumentation?.device?.unexpectedLosses ?? []).filter((loss) => (
    loss.deviceId === instrumentation?.target?.deviceId
  )).length === 0, 'Instrumentation observed loss of the target GPUDevice.');
  check((instrumentation?.shaderModules?.length ?? 0) > 0
      && instrumentation.shaderModules.every((module) => module.messages?.length === 0),
  'Shader compilation messages were observed.');
  check((instrumentation?.windowErrors?.length ?? -1) === 0,
    'Window errors were observed.');
  check((instrumentation?.unhandledRejections?.length ?? -1) === 0,
    'Unhandled page rejections were observed.');

  for (const key of [
    'browserDisconnects',
    'consoleErrors',
    'consoleWarnings',
    'httpErrors',
    'pageCrashes',
    'pageErrors',
    'requestFailures',
    'unexpectedRequests',
  ]) {
    check((observations?.[key]?.length ?? 0) === 0, `Browser observation ${key} is not empty.`);
  }
  const requestedPaths = observations?.requestedPaths ?? [];
  const serverRequests = observations?.serverRequests ?? [];
  check(serverRequests.length === requestedPaths.length
      && serverRequests.every((request) => request.status === 200),
  'Allowlisted browser requests did not all traverse the strict Vite middleware.');
  const pathCounts = (paths) => Object.fromEntries(
    [...new Set(paths)].sort().map((pathname) => [
      pathname,
      paths.filter((candidate) => candidate === pathname).length,
    ]),
  );
  check(JSON.stringify(pathCounts(serverRequests.map((request) => request.path)))
      === JSON.stringify(pathCounts(requestedPaths)),
  'Playwright allowlist and Vite middleware request inventories differ.');
  check(requestedPaths.filter((pathname) => (
    pathname === `${ORT_ROUTE}/ort.webgpu.bundle.min.mjs`
  )).length === 1, 'Pinned ORT WebGPU bundle was not requested exactly once.');
  check(requestedPaths.filter((pathname) => (
    pathname === `${ORT_ROUTE}/ort-wasm-simd-threaded.asyncify.wasm`
  )).length === 1, 'Pinned ORT asyncify wasm was not requested exactly once.');
  const ortBundleRequest = serverRequests.filter((request) => (
    request.path === `${ORT_ROUTE}/ort.webgpu.bundle.min.mjs`
  ));
  const ortWasmRequest = serverRequests.filter((request) => (
    request.path === `${ORT_ROUTE}/ort-wasm-simd-threaded.asyncify.wasm`
  ));
  check(ortBundleRequest.length === 1
      && ortBundleRequest[0].contentType === 'text/javascript; charset=utf-8'
      && ortBundleRequest[0].bytes === protocol?.onnxRuntime?.files
        ?.['ort.webgpu.bundle.min.mjs']?.bytes,
  'Vite did not serve the exact pinned ORT JavaScript artifact.');
  check(ortWasmRequest.length === 1
      && ortWasmRequest[0].contentType === 'application/wasm'
      && ortWasmRequest[0].bytes === protocol?.onnxRuntime?.files
        ?.['ort-wasm-simd-threaded.asyncify.wasm']?.bytes,
  'Vite did not serve the exact pinned ORT wasm artifact.');

  return {
    executionProof,
    reasons,
    targetOperations: {
      clears: targetClears,
      copies: targetCopies,
      destroys: targetDestroys,
      indirect: targetIndirect,
      maps: targetMaps,
      vertexOrIndex: targetVertexOrIndex,
      writes: targetWrites,
    },
    targetTrace: summarizeTargetTrace(instrumentation),
    verified: reasons.length === 0,
  };
}

export function validateOrtGpuBufferScreenshotReady(page, protocol) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  const expectedStatus = protocol?.page?.screenshotStage ?? 'screenshot-ready';
  const run = page?.runs?.[0];
  check(page?.kind === 'three-webgpu-onnxruntime-borrowed-gpu-buffer-correctness-canary',
    'Screenshot-ready page kind is incorrect.');
  check(page?.status === expectedStatus && page?.error === null,
    'Page did not reach the screenshot-ready checkpoint.');
  check(page?.lifecycle?.screenshotReady === true,
    'Page did not record screenshot readiness.');
  check(page?.lifecycle?.continuationCalls === 0,
    'Lifecycle continuation began before the screenshot.');
  check(run?.runIndex === 0 && run?.ort?.returnedSameBuffer === true,
    'Screenshot checkpoint lacks exact ORT run-1 identity.');
  check(run?.three?.backendBufferSame === true && run?.three?.oracle?.exact === true,
    'Screenshot checkpoint lacks exact Three run-1 proof.');
  check(page?.runs?.[1] === null, 'Run 2 began before screenshot capture.');
  check(page?.lifecycle?.ownerDestroyedTarget === false,
    'Target was destroyed before screenshot capture.');
  return { reasons, verified: reasons.length === 0 };
}

function summarizeInstrumentation(instrumentation) {
  return {
    ...instrumentation,
    shaderModules: (instrumentation?.shaderModules ?? []).map((module) => ({
      bytes: Buffer.byteLength(module.code ?? '', 'utf8'),
      label: module.label,
      messages: module.messages,
      moduleId: module.moduleId,
      phase: module.phase,
      sha256: sha256(Buffer.from(module.code ?? '', 'utf8')),
    })),
  };
}

export function targetCriticalInstrumentationProjection(instrumentation) {
  const projected = JSON.parse(JSON.stringify(instrumentation));
  const targetDeviceId = projected?.target?.deviceId;
  if (projected?.device !== undefined) {
    projected.device.unexpectedLosses = (projected.device.unexpectedLosses ?? [])
      .filter((loss) => loss.deviceId === targetDeviceId);
  }
  projected.events = (projected?.events ?? []).filter((event) => (
    event.method !== 'destroyDevice' || event.deviceId === targetDeviceId
  ));
  return projected;
}

export function parseOrtGpuBufferArguments(args = process.argv.slice(2)) {
  if (exactArray(args, ['--offline-preflight'])) return { mode: 'offline-preflight' };
  requireCondition(args.length === 4, 'Expected --browser PATH --expected-research-commit HASH.');
  requireCondition(args[0] === '--browser', 'First argument must be --browser.');
  requireCondition(args[2] === '--expected-research-commit',
    'Third argument must be --expected-research-commit.');
  requireCondition(GIT_COMMIT_PATTERN.test(args[3]),
    'Expected research commit must be a full Git object ID.');
  return {
    browserPath: path.resolve(args[1]),
    expectedResearchCommit: args[3],
    mode: 'browser',
  };
}

export const parseThreeOrtGpuBufferArguments = parseOrtGpuBufferArguments;

function protocolExpectedBuildRoot(protocol, mode) {
  const relative = mode === 'offline-preflight'
    ? protocol?.build?.offlinePreflightDirectory
    : protocol?.build?.liveDirectory;
  return resolveRepoPath(relative, `${mode} build`);
}

function verifyProtocolHarnessConstants(protocol) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  check(protocol?.schemaVersion === 1, 'Protocol schemaVersion is not 1.');
  check(protocol?.candidate?.commit === CANDIDATE_COMMIT, 'Protocol candidate commit differs.');
  check(protocol?.candidate?.tree === CANDIDATE_TREE, 'Protocol candidate tree differs.');
  check(protocol?.candidate?.subject === CANDIDATE_SUBJECT, 'Protocol candidate subject differs.');
  check(protocol?.onnxRuntime?.package === 'onnxruntime-web',
    'Protocol ORT package name differs.');
  check(protocol?.onnxRuntime?.version === ORT_VERSION, 'Protocol ORT version differs.');
  check(/^sha512-[A-Za-z0-9+/]+={0,2}$/u.test(protocol?.onnxRuntime?.lockIntegrity ?? ''),
    'Protocol ORT lockfile integrity pin is invalid.');
  check(protocol?.build?.route === `${BUILD_ROUTE}/`, 'Protocol Three build route differs.');
  check(protocol?.onnxRuntime?.route === `${ORT_ROUTE}/`, 'Protocol ORT route differs.');
  check(path.resolve(protocolExpectedBuildRoot(protocol, 'browser')) === liveBuildRoot,
    'Protocol live build directory differs.');
  check(path.resolve(protocolExpectedBuildRoot(protocol, 'offline-preflight'))
      === preflightBuildRoot,
  'Protocol preflight build directory differs.');
  check(protocol?.target?.physicalBytes === 32 && protocol?.target?.logicalBytes === 24,
    'Protocol target size differs from the v1 proof.');
  check(protocol?.model?.bytes === 130
      && /^[a-f0-9]{64}$/u.test(protocol?.model?.sha256 ?? ''),
  'Protocol model content pin is invalid.');
  check(protocol?.runs?.length === 2
      && protocol.runs.every((run, index) => (
        exactArray(run.input, RUN_VECTORS[index].input)
          && exactArray(run.expectedOutput, RUN_VECTORS[index].expectedOutput)
          && exactArray(run.expectedBits, RUN_VECTORS[index].expectedBits)
      )),
  'Protocol run vectors differ from the exact v1 vectors.');
  return { reasons, verified: reasons.length === 0 };
}

async function readPageArtifacts(protocol) {
  const entries = [
    ['html', protocol?.page?.path, 'text/html; charset=utf-8'],
    ['module', protocol?.page?.module, 'text/javascript; charset=utf-8'],
    ['contract', protocol?.page?.contract, 'text/javascript; charset=utf-8'],
  ];
  const result = {};
  for (const [name, relativePath, contentType] of entries) {
    const localPath = resolveRepoPath(relativePath, `page ${name}`);
    const bytes = await readFile(localPath);
    result[name] = {
      bytes,
      contentType,
      inventory: { bytes: bytes.length, sha256: sha256(bytes) },
      localPath,
      route: `/${relativePath.replaceAll('\\', '/').replace(/^\/+/, '')}`,
    };
  }
  return result;
}

async function offlinePreflight() {
  const protocol = (await readProtocol()).value;
  const protocolValidation = verifyProtocolHarnessConstants(protocol);
  requireCondition(protocolValidation.verified, protocolValidation.reasons.join(' '));
  const before = await validateCandidateIdentity(protocol);
  const built = await buildOrtGpuBufferCandidate(preflightBuildRoot);
  const buildValidation = verifyBuiltOrtGpuBufferCandidate(built, protocol.build.files);
  requireCondition(buildValidation.verified && buildValidation.inventoryPinned,
    buildValidation.reasons.join(' ') || 'Three build inventory is not pinned.');
  const runtime = await readOrtRuntimeArtifacts();
  const runtimeValidation = verifyOrtRuntimeArtifacts(
    runtime,
    protocol.onnxRuntime.files,
    protocol.onnxRuntime,
  );
  requireCondition(runtimeValidation.verified && runtimeValidation.inventoryPinned,
    runtimeValidation.reasons.join(' ') || 'ORT runtime inventory is not pinned.');
  const pageArtifacts = await readPageArtifacts(protocol);
  const pageValidation = verifyOrtGpuBufferPageSource(
    pageArtifacts.module.bytes.toString('utf8'),
  );
  requireCondition(pageValidation.verified, pageValidation.reasons.join(' '));
  const after = await validateCandidateIdentity(protocol);
  requireCondition(JSON.stringify(before) === JSON.stringify(after),
    'Candidate changed during offline preflight.');
  console.log(JSON.stringify({
    build: built.inventory,
    candidate: after,
    mode: 'offline-preflight',
    page: Object.fromEntries(Object.entries(pageArtifacts).map(([name, artifact]) => (
      [name, artifact.inventory]
    ))),
    protocolValidation,
    runtime: { files: runtime.files, lock: runtime.lock, package: runtime.package },
    validation: { build: buildValidation, page: pageValidation, runtime: runtimeValidation },
  }, null, 2));
}

async function browserCanary({ browserPath, expectedResearchCommit }) {
  const { bytes: protocolBytes, value: protocol } = await readProtocol();
  const protocolValidation = verifyProtocolHarnessConstants(protocol);
  requireCondition(protocolValidation.verified, protocolValidation.reasons.join(' '));
  const runRoot = resolveRepoPath(protocol.output.directory, 'result directory');
  const reportPath = path.join(runRoot, protocol.output.report);
  const checksumPath = `${reportPath}.sha256`;
  const screenshotPath = path.join(runRoot, protocol.output.screenshot);
  requireCondition(browserPath === path.resolve(protocol.browser.requiredExecutablePath),
    'Browser path differs from the protocol.');
  requireCondition(exactArray(protocol.browser.playwrightArguments, []),
    'Protocol attempts to pass browser arguments.');
  requireCondition(protocol.output.exclusiveDirectory === true
      && protocol.output.overwrite === false,
  'Protocol does not require a single-use, non-overwriting result directory.');
  requireCondition(await exists(runRoot) === false, 'Single-use result directory already exists.');
  requireCondition(await exists(liveBuildRoot) === false,
    'Single-use live build directory already exists.');

  const researchBefore = await gitSnapshot(repoRoot);
  requireCondition(researchBefore.commit === expectedResearchCommit,
    'Research HEAD differs from the approved commit.');
  requireCondition(researchBefore.status === '', 'Research repository is not clean.');
  const candidateBefore = await validateCandidateIdentity(protocol);
  const browserBytes = await readFile(browserPath);
  const browserFile = await stat(browserPath);
  const pageArtifacts = await readPageArtifacts(protocol);
  const pageSourceValidation = verifyOrtGpuBufferPageSource(
    pageArtifacts.module.bytes.toString('utf8'),
  );
  requireCondition(pageSourceValidation.verified, pageSourceValidation.reasons.join(' '));
  const runtime = await readOrtRuntimeArtifacts();
  const runtimeValidation = verifyOrtRuntimeArtifacts(
    runtime,
    protocol.onnxRuntime.files,
    protocol.onnxRuntime,
  );
  requireCondition(runtimeValidation.verified && runtimeValidation.inventoryPinned,
    runtimeValidation.reasons.join(' ') || 'ORT runtime inventory is not pinned.');

  await mkdir(path.dirname(runRoot), { recursive: true });
  await mkdir(runRoot);
  let browser = null;
  let browserCloseRequested = false;
  let context = null;
  let finalRawEvidence = null;
  let server = null;
  const observations = {
    browserDisconnects: [],
    consoleErrors: [],
    consoleWarnings: [],
    httpErrors: [],
    pageCrashes: [],
    pageErrors: [],
    requestFailures: [],
    requestedPaths: [],
    serverRequests: [],
    unexpectedRequests: [],
  };
  const report = {
    browser: null,
    build: null,
    candidate: { after: null, before: candidateBefore },
    claimBoundary: protocol.claimBoundary,
    cleanup: {
      browser: { attempted: false, closed: false, error: null },
      context: { attempted: false, closed: false, error: null },
      server: { attempted: false, closed: false, error: null },
    },
    evidence: { final: null, late: null, screenshotReady: null },
    failure: null,
    kind: 'three-ort-webgpu-borrowed-buffer-canary-report',
    observations,
    onnxRuntime: {
      files: runtime.files,
      lock: runtime.lock,
      package: runtime.package,
      validation: runtimeValidation,
    },
    pageSource: Object.fromEntries(Object.entries(pageArtifacts).map(([name, artifact]) => (
      [name, artifact.inventory]
    ))),
    protocol: { bytes: protocolBytes.length, sha256: sha256(protocolBytes) },
    protocolValidation,
    research: { after: null, before: researchBefore, expectedCommit: expectedResearchCommit },
    schemaVersion: 1,
    screenshot: null,
    status: 'running',
    validation: null,
  };

  try {
    const built = await buildOrtGpuBufferCandidate(liveBuildRoot);
    const buildValidation = verifyBuiltOrtGpuBufferCandidate(built, protocol.build.files);
    requireCondition(buildValidation.verified && buildValidation.inventoryPinned,
      buildValidation.reasons.join(' ') || 'Three build inventory is not pinned.');
    report.build = { files: built.inventory, validation: buildValidation };

    const served = new Map();
    for (const artifact of Object.values(pageArtifacts)) {
      served.set(artifact.route, { body: artifact.bytes, type: artifact.contentType });
    }
    for (const [filename, bytes] of Object.entries(built.modules)) {
      served.set(`${BUILD_ROUTE}/${filename}`, {
        body: bytes,
        type: 'text/javascript; charset=utf-8',
      });
    }
    for (const [filename, bytes] of Object.entries(runtime.modules)) {
      served.set(`${ORT_ROUTE}/${filename}`, {
        body: bytes,
        type: filename.endsWith('.wasm')
          ? 'application/wasm'
          : 'text/javascript; charset=utf-8',
      });
    }

    server = await createServer({
      appType: 'custom',
      cacheDir: path.join(liveBuildRoot, '.vite-cache'),
      configFile: false,
      logLevel: 'error',
      optimizeDeps: { noDiscovery: true },
      plugins: [{
        configureServer(viteServer) {
          // Registered in configureServer so this exact-byte middleware runs
          // before Vite's transform and filesystem middleware.
          viteServer.middlewares.use((request, response) => {
            const requestUrl = new URL(request.url ?? '/', 'http://127.0.0.1');
            const artifact = served.get(requestUrl.pathname);
            const exact = request.method === 'GET'
              && requestUrl.search === ''
              && artifact !== undefined;
            observations.serverRequests.push({
              bytes: exact ? artifact.body.length : 0,
              contentType: exact ? artifact.type : 'text/plain; charset=utf-8',
              method: request.method ?? null,
              path: requestUrl.pathname,
              status: exact ? 200 : 404,
            });
            response.statusCode = exact ? 200 : 404;
            response.setHeader('Cache-Control', 'no-store');
            for (const [name, value] of Object.entries(ISOLATION_HEADERS)) {
              response.setHeader(name, value);
            }
            if (!exact) {
              response.setHeader('Content-Type', 'text/plain; charset=utf-8');
              response.end('Not found.');
              return;
            }
            response.setHeader('Content-Type', artifact.type);
            response.setHeader('Content-Length', String(artifact.body.length));
            response.end(artifact.body);
          });
        },
        name: 'strict-ort-three-canary-artifacts',
      }],
      root: repoRoot,
      server: {
        fs: { allow: [repoRoot], strict: true },
        headers: ISOLATION_HEADERS,
        hmr: false,
        host: '127.0.0.1',
        port: 0,
        strictPort: false,
      },
    });
    await server.listen();
    const address = server.httpServer.address();
    requireCondition(address !== null && typeof address !== 'string', 'Vite port unavailable.');
    const baseUrl = `http://127.0.0.1:${address.port}`;

    browser = await chromium.launch({
      args: [],
      executablePath: browserPath,
      headless: true,
    });
    browser.on('disconnected', () => {
      if (!browserCloseRequested) observations.browserDisconnects.push('unexpected');
    });
    const effectiveArguments = await captureBrowserArguments(browser);
    const forbiddenArguments = findForbiddenArguments(
      effectiveArguments.arguments,
      protocol.browser.forbiddenArgumentPrefixes,
    );
    requireCondition(effectiveArguments.available, 'Effective browser arguments unavailable.');
    requireCondition(forbiddenArguments.length === 0, 'Forbidden browser GPU arguments observed.');
    report.browser = {
      effectiveArguments,
      executable: browserPath,
      file: { bytes: browserFile.size, sha256: sha256(browserBytes) },
      forbiddenArguments,
      playwrightArguments: [],
      productVersion: browser.version(),
    };

    context = await browser.newContext({
      deviceScaleFactor: protocol.page.deviceScaleFactor,
      serviceWorkers: 'block',
      viewport: {
        height: protocol.page.viewport[1],
        width: protocol.page.viewport[0],
      },
    });
    await context.route('**/*', async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      observations.requestedPaths.push(url.pathname);
      const exactRequest = url.origin === baseUrl
        && url.search === ''
        && url.hash === ''
        && request.method() === 'GET'
        && served.has(url.pathname);
      if (!exactRequest) {
        observations.unexpectedRequests.push({
          method: request.method(),
          url: request.url(),
        });
        await route.abort('blockedbyclient');
        return;
      }
      await route.continue();
    });

    const page = await context.newPage();
    page.on('console', (message) => {
      if (message.type() === 'error') observations.consoleErrors.push(message.text());
      if (message.type() === 'warning') observations.consoleWarnings.push(message.text());
    });
    page.on('crash', () => observations.pageCrashes.push('crash'));
    page.on('pageerror', (error) => observations.pageErrors.push(serializeError(error)));
    page.on('requestfailed', (request) => observations.requestFailures.push({
      errorText: request.failure()?.errorText ?? null,
      url: request.url(),
    }));
    page.on('response', (response) => {
      if (response.status() >= 400) {
        observations.httpErrors.push({ status: response.status(), url: response.url() });
      }
    });
    await page.addInitScript(installOrtGpuBufferInstrumentation);
    await page.goto(`${baseUrl}${pageArtifacts.html.route}`, {
      timeout: 60_000,
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      (screenshotStage) => {
        const status = window.__threeOrtGpuBufferCanary?.status;
        return status === screenshotStage
          || status === 'failed'
          || (window.__threeOrtGpuBufferInstrumentation?.windowErrors?.length ?? 0) > 0
          || (window.__threeOrtGpuBufferInstrumentation?.unhandledRejections?.length ?? 0) > 0;
      },
      protocol.page.screenshotStage,
      { timeout: protocol.page.timeoutMilliseconds ?? DEFAULT_PAGE_TIMEOUT_MS },
    );
    const readyPage = await page.evaluate(() => (
      structuredClone(window.__threeOrtGpuBufferCanary)
    ));
    const readyInstrumentation = await page.evaluate(() => (
      structuredClone(window.__threeOrtGpuBufferInstrumentation)
    ));
    report.evidence.screenshotReady = {
      instrumentation: readyInstrumentation,
      page: readyPage,
    };
    const readyValidation = validateOrtGpuBufferScreenshotReady(readyPage, protocol);
    requireCondition(readyValidation.verified, readyValidation.reasons.join(' '));

    const screenshotBytes = await page.screenshot({ fullPage: false });
    const screenshotDimensions = pngDimensions(screenshotBytes);
    requireCondition(
      screenshotDimensions.width === protocol.page.viewport[0]
        && screenshotDimensions.height === protocol.page.viewport[1],
      'Screenshot dimensions differ from the protocol viewport.',
    );
    requireCondition(screenshotBytes.length > 10_000, 'Screenshot is unexpectedly small.');
    await writeFile(screenshotPath, screenshotBytes, { flag: 'wx' });
    report.screenshot = {
      bytes: screenshotBytes.length,
      ...screenshotDimensions,
      path: path.relative(repoRoot, screenshotPath).replaceAll('\\', '/'),
      sha256: sha256(screenshotBytes),
      stage: protocol.page.screenshotStage,
    };

    await page.evaluate(() => {
      void window.__continueThreeOrtGpuBufferCanary();
    });
    await page.waitForFunction(
      () => ['passed', 'failed'].includes(window.__threeOrtGpuBufferCanary?.status),
      undefined,
      { timeout: protocol.page.timeoutMilliseconds ?? DEFAULT_PAGE_TIMEOUT_MS },
    );
    const finalPage = await page.evaluate(() => (
      structuredClone(window.__threeOrtGpuBufferCanary)
    ));
    const instrumentation = await withDeadline(
      page.evaluate(() => window.__collectThreeOrtGpuBufferInstrumentation()),
      protocol.page.timeoutMilliseconds ?? DEFAULT_PAGE_TIMEOUT_MS,
      'GPU evidence collection',
    );
    const rawEvidence = { instrumentation, page: finalPage };
    finalRawEvidence = rawEvidence;
    report.evidence.final = {
      instrumentation: summarizeInstrumentation(instrumentation),
      page: finalPage,
    };
    report.validation = validateOrtGpuBufferEvidence(rawEvidence, protocol, observations);
    requireCondition(report.validation.verified, report.validation.reasons.join(' '));

    await page.waitForTimeout(100);
    const lateInstrumentation = await page.evaluate(() => (
      structuredClone(window.__threeOrtGpuBufferInstrumentation)
    ));
    const {
      collection: ignoredCollection,
      shaderModules: ignoredShaderModules,
      ...collectedState
    } = instrumentation;
    void ignoredCollection;
    void ignoredShaderModules;
    requireCondition(
      JSON.stringify(targetCriticalInstrumentationProjection(lateInstrumentation))
        === JSON.stringify(targetCriticalInstrumentationProjection(collectedState)),
      'Target-critical GPU or page instrumentation changed after final evidence collection.',
    );
    report.evidence.late = lateInstrumentation;
    report.validation = validateOrtGpuBufferEvidence(rawEvidence, protocol, observations);
    requireCondition(report.validation.verified,
      `Late validation failed: ${report.validation.reasons.join(' ')}`);

    const candidateAfter = await validateCandidateIdentity(protocol);
    requireCondition(JSON.stringify(candidateAfter) === JSON.stringify(candidateBefore),
      'Candidate changed during browser canary.');
    report.candidate.after = candidateAfter;
    const researchAfter = await gitSnapshot(repoRoot);
    requireCondition(JSON.stringify(researchAfter) === JSON.stringify(researchBefore),
      'Research repository changed during browser canary.');
    report.research.after = researchAfter;
    report.status = 'passed';
  } catch (error) {
    report.failure = serializeError(error);
    report.status = 'failed';
  } finally {
    try {
      report.cleanup.context.attempted = context !== null;
      if (context !== null) {
        await withDeadline(context.close(), TEARDOWN_TIMEOUT_MS, 'Browser context teardown');
        report.cleanup.context.closed = true;
      }
    } catch (error) {
      report.cleanup.context.error = serializeError(error);
      report.failure ??= serializeError(error);
      report.status = 'failed';
    }
    try {
      report.cleanup.browser.attempted = browser !== null;
      if (browser !== null) {
        browserCloseRequested = true;
        await withDeadline(browser.close(), TEARDOWN_TIMEOUT_MS, 'Browser teardown');
        report.cleanup.browser.closed = true;
      }
    } catch (error) {
      report.cleanup.browser.error = serializeError(error);
      report.failure ??= serializeError(error);
      report.status = 'failed';
    }
    try {
      report.cleanup.server.attempted = server !== null;
      if (server !== null) {
        await withDeadline(server.close(), TEARDOWN_TIMEOUT_MS, 'Vite teardown');
        report.cleanup.server.closed = true;
      }
    } catch (error) {
      report.cleanup.server.error = serializeError(error);
      report.failure ??= serializeError(error);
      report.status = 'failed';
    }
  }

  if (report.status === 'passed') {
    const cleanupComplete = Object.values(report.cleanup).every((entry) => (
      entry.attempted === true && entry.closed === true && entry.error === null
    ));
    if (!cleanupComplete) {
      report.failure = serializeError(new Error('Browser canary teardown was incomplete.'));
      report.status = 'failed';
    }
  }
  if (report.status === 'passed' && finalRawEvidence !== null) {
    report.validation = validateOrtGpuBufferEvidence(
      finalRawEvidence,
      protocol,
      observations,
    );
    if (!report.validation.verified) {
      report.failure = serializeError(new Error(
        `Post-teardown validation failed: ${report.validation.reasons.join(' ')}`,
      ));
      report.status = 'failed';
    }
  }
  const reportBytes = canonicalJson(report);
  await writeFile(reportPath, reportBytes, { flag: 'wx' });
  const reportHash = sha256(reportBytes);
  await writeFile(checksumPath, `${reportHash}  report.json\n`, { flag: 'wx' });
  if (await exists(screenshotPath)) await chmod(screenshotPath, 0o444);
  await chmod(reportPath, 0o444);
  await chmod(checksumPath, 0o444);
  if (report.status !== 'passed') {
    console.error(JSON.stringify({ failure: report.failure, report: reportPath }, null, 2));
    process.exitCode = 1;
  } else {
    console.log(JSON.stringify({
      report: reportPath,
      reportSha256: reportHash,
      screenshot: report.screenshot,
      status: report.status,
      verified: report.validation.verified,
    }, null, 2));
  }
}

async function main() {
  const arguments_ = parseOrtGpuBufferArguments();
  if (arguments_.mode === 'offline-preflight') {
    await offlinePreflight();
  } else {
    await browserCanary(arguments_);
  }
}

if (process.argv[1] !== undefined
    && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}

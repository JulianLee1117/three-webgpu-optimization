import { execFile as execFileCallback } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  chmod,
  mkdir,
  readFile,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { isDeepStrictEqual, promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import {
  PRIMITIVE_INDEX_CANARY_KIND,
  PRIMITIVE_INDEX_INDICES,
  PRIMITIVE_INDEX_MIN_PIXELS_PER_TUPLE,
  PRIMITIVE_INDEX_ORACLE_SIZE,
  PRIMITIVE_INDEX_SENTINEL,
  expectedPrimitiveIndexTuples,
} from '../src/three-primitive-index-canary-contract.js';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidateRoot = path.join(
  repoRoot,
  '.generated',
  'three-primitive-index-current-dev-candidate',
);
const liveBuildRoot = path.join(
  repoRoot,
  '.generated',
  'three-primitive-index-canary-v3-live-build',
);
const preflightBuildRoot = path.join(
  repoRoot,
  '.generated',
  'three-primitive-index-canary-v3-preflight-build',
);
const protocolPath = path.join(repoRoot, 'protocols', 'three-primitive-index-canary-v3.json');
const pagePath = path.join(repoRoot, 'three-primitive-index-canary.html');
const pageModulePath = path.join(repoRoot, 'src', 'three-primitive-index-canary.js');
const contractModulePath = path.join(
  repoRoot,
  'src',
  'three-primitive-index-canary-contract.js',
);
const GIT_COMMIT_PATTERN = /^[a-f0-9]{40}$/u;
const EXPECTED_BUILD_FILES = Object.freeze([
  'three.core.js',
  'three.module.js',
  'three.tsl.js',
  'three.webgpu.js',
  'three.webgpu.nodes.js',
]);
const BUILD_ROUTE = '/__three_primitive_index__';
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
});
const TEARDOWN_TIMEOUT_MS = 10_000;

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
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 8_192),
  };
}

function exactArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function occurrences(source, token) {
  return typeof source === 'string' ? source.split(token).length - 1 : 0;
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

async function validateCandidateIdentity(protocol) {
  const snapshot = await gitSnapshot(candidateRoot);
  requireCondition(snapshot.commit === protocol.candidate.commit, 'Candidate commit mismatch.');
  requireCondition(snapshot.tree === protocol.candidate.tree, 'Candidate tree mismatch.');
  requireCondition(snapshot.status === '', 'Candidate repository is not clean.');
  const { stdout: subject } = await runGit(candidateRoot, ['show', '-s', '--format=%s', 'HEAD']);
  requireCondition(subject.trim() === protocol.candidate.subject, 'Candidate subject mismatch.');
  return snapshot;
}

export async function buildPrimitiveIndexCandidate(targetRoot) {
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
  await mkdir(targetRoot);
  const { rollup } = await import(pathToFileURL(rollupModulePath));
  const configUrl = pathToFileURL(
    path.join(candidateRoot, 'utils', 'build', 'rollup.config.js'),
  );
  const configurations = (await import(configUrl)).default;

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

  const inventory = {};
  const modules = {};
  for (const filename of EXPECTED_BUILD_FILES) {
    const bytes = await readFile(path.join(targetRoot, filename));
    modules[filename] = bytes;
    inventory[filename] = { bytes: bytes.length, sha256: sha256(bytes) };
  }
  return { inventory, modules };
}

export function verifyBuiltPrimitiveIndexCandidate(built, expectedInventory = null) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  const webgpu = built?.modules?.['three.webgpu.js']?.toString('utf8') ?? '';
  const webgpuNodes = built?.modules?.['three.webgpu.nodes.js']?.toString('utf8') ?? '';
  const tsl = built?.modules?.['three.tsl.js']?.toString('utf8') ?? '';
  const runtime = `${webgpu}\n${webgpuNodes}`;
  check(EXPECTED_BUILD_FILES.every((file) => built?.inventory?.[file] !== undefined),
    'Expected built module inventory is incomplete.');
  if (expectedInventory !== null) {
    check(JSON.stringify(built?.inventory) === JSON.stringify(expectedInventory),
      'Built module inventory differs from the frozen offline preflight.');
  }
  check(runtime.includes("PrimitiveIndex: 'primitive-index'"),
    'Built WebGPU runtime lacks the primitive-index feature constant.');
  check(runtime.includes("enableDirective( 'primitive_index' )"),
    'Built WebGPU runtime lacks primitive_index directive generation.');
  check(runtime.includes("getBuiltin( 'primitive_index', 'primitiveIndex', 'u32' )"),
    'Built WebGPU runtime lacks the primitive_index builtin generation.');
  check(tsl.includes('primitiveIndex'), 'Built TSL module lacks primitiveIndex export.');
  return { reasons, verified: reasons.length === 0 };
}

export function findForbiddenArguments(arguments_, prefixes) {
  return arguments_.filter((argument) => {
    const normalized = String(argument).toLowerCase();
    return prefixes.some((prefix) => (
      normalized === prefix || normalized.startsWith(`${prefix}=`)
    ));
  });
}

async function captureBrowserArguments(browser) {
  try {
    const session = await browser.newBrowserCDPSession();
    const result = await session.send('Browser.getBrowserCommandLine');
    await session.detach();
    return { available: true, arguments: result.arguments ?? [] };
  } catch (error) {
    return { available: false, arguments: [], error: serializeError(error) };
  }
}

function installPrimitiveIndexInstrumentation() {
  const state = {
    acquisition: {
      adapterFeatures: [],
      adapterInfo: null,
      adapterRequests: [],
      deviceFeatures: [],
      deviceRequests: [],
    },
    device: {
      errorScopes: { pops: [], pushes: [] },
      pipelinePromiseRejections: [],
      uncapturedErrors: [],
      unexpectedLoss: null,
    },
    drawEncoders: [],
    setupError: null,
    textures: [],
    wgslLanguageFeatures: [],
    windowErrors: [],
    unhandledRejections: [],
  };
  Object.defineProperty(window, '__threePrimitiveIndexInstrumentation', { value: state });

  window.addEventListener('error', (event) => {
    state.windowErrors.push({
      column: Number(event.colno ?? 0),
      filename: String(event.filename ?? ''),
      line: Number(event.lineno ?? 0),
      message: String(event.message ?? ''),
    });
  });
  window.addEventListener('unhandledrejection', (event) => {
    state.unhandledRejections.push(String(event.reason?.message ?? event.reason));
  });

  const modules = [];
  const pendingScopes = [];
  const scopeStack = [];
  let nextScopeId = 0;
  let activeDevice = null;

  const replaceCallable = (target, name, factory) => {
    const original = target?.[name];
    if (typeof original !== 'function') return false;
    const replacement = factory(original.bind(target));
    Object.defineProperty(target, name, {
      configurable: true,
      value: replacement,
      writable: true,
    });
    return target[name] === replacement;
  };

  const descriptorData = (descriptor) => {
    if (descriptor === undefined) return null;
    const result = {};
    for (const [key, value] of Object.entries(descriptor)) {
      if (value === undefined) continue;
      if (key === 'requiredFeatures') {
        result.requiredFeatures = Array.from(value, String).sort();
      } else if (key === 'requiredLimits') {
        result.requiredLimits = Object.fromEntries(
          Object.entries(value).sort(([left], [right]) => left.localeCompare(right)),
        );
      } else if (['boolean', 'number', 'string'].includes(typeof value)) {
        result[key] = value;
      }
    }
    return result;
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

  const instrumentDrawEncoder = (encoder, kind) => {
    const record = { kind, events: [] };
    state.drawEncoders.push(record);
    replaceCallable(encoder, 'setIndexBuffer', (original) => (...args) => {
      record.events.push({
        method: 'setIndexBuffer',
        format: String(args[1] ?? ''),
        offset: Number(args[2] ?? 0),
        size: args[3] === undefined ? null : Number(args[3]),
      });
      return original(...args);
    });
    for (const method of ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
      replaceCallable(encoder, method, (original) => (...args) => {
        record.events.push({
          method,
          arguments: args.map((value) => (
            typeof value === 'number' ? value : value?.constructor?.name ?? null
          )),
        });
        return original(...args);
      });
    }
    return encoder;
  };

  const instrumentDevice = (device) => {
    activeDevice = device;
    state.acquisition.deviceFeatures = Array.from(device.features, String).sort();
    device.addEventListener?.('uncapturederror', (event) => {
      event.preventDefault();
      state.device.uncapturedErrors.push({
        name: String(event.error?.constructor?.name ?? 'GPUError'),
        message: String(event.error?.message ?? event.error),
      });
    });
    void device.lost.then((info) => {
      state.device.unexpectedLoss = {
        message: String(info.message ?? ''),
        reason: String(info.reason ?? ''),
      };
    });

    const pushInstalled = replaceCallable(device, 'pushErrorScope', (original) => (filter) => {
      const record = { filter: String(filter), id: nextScopeId };
      nextScopeId += 1;
      scopeStack.push(record);
      state.device.errorScopes.pushes.push(record);
      return original(filter);
    });
    const popInstalled = replaceCallable(device, 'popErrorScope', (original) => () => {
      const pushed = scopeStack.pop() ?? null;
      const record = {
        error: null,
        filter: pushed?.filter ?? null,
        id: pushed?.id ?? null,
        rejection: null,
        settled: false,
      };
      state.device.errorScopes.pops.push(record);
      const promise = original();
      pendingScopes.push(Promise.resolve(promise).then((error) => {
        record.error = error === null ? null : {
          message: String(error?.message ?? error),
          name: String(error?.constructor?.name ?? 'GPUError'),
        };
        record.settled = true;
      }, (error) => {
        record.rejection = {
          message: String(error?.message ?? error),
          name: String(error?.name ?? 'Error'),
        };
        record.settled = true;
      }));
      return promise;
    });
    if (!pushInstalled || !popInstalled) {
      throw new Error('Could not instrument GPU device error scopes.');
    }

    replaceCallable(device, 'createShaderModule', (original) => (descriptor) => {
      const module = original(descriptor);
      modules.push({
        code: String(descriptor?.code ?? ''),
        label: descriptor?.label ?? null,
        module,
      });
      return module;
    });
    replaceCallable(device, 'createTexture', (original) => (descriptor) => {
      state.textures.push({
        dimension: String(descriptor?.dimension ?? '2d'),
        format: String(descriptor?.format ?? ''),
        label: descriptor?.label ?? null,
        sampleCount: Number(descriptor?.sampleCount ?? 1),
        usage: Number(descriptor?.usage ?? 0),
      });
      return original(descriptor);
    });
    const instrumentPipeline = (name) => {
      replaceCallable(device, name, (original) => (descriptor) => {
        const result = original(descriptor);
        if (result?.then instanceof Function) {
          void result.catch((error) => {
            state.device.pipelinePromiseRejections.push({
              message: String(error?.message ?? error),
              name: String(error?.name ?? 'Error'),
            });
          });
        }
        return result;
      });
    };
    instrumentPipeline('createRenderPipeline');
    instrumentPipeline('createRenderPipelineAsync');
    replaceCallable(device, 'createRenderBundleEncoder', (original) => (descriptor) => (
      instrumentDrawEncoder(original(descriptor), 'render-bundle')
    ));
    replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
      const commandEncoder = original(descriptor);
      replaceCallable(commandEncoder, 'beginRenderPass', (begin) => (passDescriptor) => (
        instrumentDrawEncoder(begin(passDescriptor), 'render-pass')
      ));
      return commandEncoder;
    });
  };

  try {
    const gpu = navigator.gpu;
    state.wgslLanguageFeatures = gpu?.wgslLanguageFeatures == null
      ? []
      : Array.from(gpu.wgslLanguageFeatures, String).sort();
    if (gpu === undefined) throw new Error('navigator.gpu is unavailable.');
    const installed = replaceCallable(gpu, 'requestAdapter', (requestAdapter) => async (...args) => {
      state.acquisition.adapterRequests.push({
        argumentsCount: args.length,
        descriptor: descriptorData(args[0]),
      });
      const adapter = await requestAdapter(...args);
      if (adapter === null) return null;
      state.acquisition.adapterFeatures = Array.from(adapter.features, String).sort();
      state.acquisition.adapterInfo = adapterInfo(adapter);
      const deviceInstalled = replaceCallable(
        adapter,
        'requestDevice',
        (requestDevice) => async (...deviceArgs) => {
          state.acquisition.deviceRequests.push({
            argumentsCount: deviceArgs.length,
            descriptor: descriptorData(deviceArgs[0]),
          });
          const device = await requestDevice(...deviceArgs);
          instrumentDevice(device);
          return device;
        },
      );
      if (!deviceInstalled) throw new Error('Could not instrument adapter.requestDevice.');
      return adapter;
    });
    if (!installed) throw new Error('Could not instrument navigator.gpu.requestAdapter.');
  } catch (error) {
    state.setupError = {
      message: String(error?.message ?? error),
      name: String(error?.name ?? 'Error'),
    };
  }

  let collected = false;
  window.__collectThreePrimitiveIndexInstrumentation = async () => {
    if (collected) throw new Error('primitiveIndex instrumentation may only be collected once.');
    collected = true;
    if (activeDevice !== null) await activeDevice.queue.onSubmittedWorkDone();
    await Promise.allSettled(pendingScopes);
    const shaderModules = [];
    for (const record of modules) {
      try {
        const info = await record.module.getCompilationInfo();
        shaderModules.push({
          code: record.code,
          label: record.label,
          messages: Array.from(info.messages, (message) => ({
            lineNum: Number(message.lineNum ?? 0),
            linePos: Number(message.linePos ?? 0),
            message: String(message.message),
            type: String(message.type),
          })),
        });
      } catch (error) {
        shaderModules.push({
          code: record.code,
          label: record.label,
          messages: [{ message: String(error), type: 'collection-error' }],
        });
      }
    }
    return structuredClone({ ...state, shaderModules });
  };
}

export function validatePrimitiveIndexEvidence(evidence, protocol, observations) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  const page = evidence?.page;
  const instrumentation = evidence?.instrumentation;
  const geometry = page?.geometry;
  const oracle = page?.oracle;

  check(page?.kind === PRIMITIVE_INDEX_CANARY_KIND, 'Page kind is incorrect.');
  check(page?.status === 'passed' && page?.error === null, 'Page did not pass.');
  check(page?.renderer?.backendIsWebGPU === true, 'Page did not use the WebGPU backend.');
  check(page?.renderer?.devicePrimitiveIndex === true,
    'Renderer device does not expose primitive-index.');
  check(exactArray(geometry?.attributes, protocol.geometry.attributes),
    'Geometry attributes differ or include a primitive-ID attribute.');
  check(geometry?.indexArrayType === protocol.geometry.indexArrayType,
    'Geometry index type is not Uint16Array.');
  check(geometry?.indexCount === protocol.geometry.indexCount,
    'Geometry index count differs.');
  check(exactArray(geometry?.indexValues, protocol.geometry.indexValues),
    'Geometry index values differ.');
  check(geometry?.positionCount === protocol.geometry.positionCount,
    'Geometry position count differs.');
  check(geometry?.instanceCount === protocol.geometry.instanceCount,
    'Geometry instance count differs.');

  check(oracle?.typedArray === protocol.oracle.typedArray, 'Oracle is not a Uint32Array.');
  check(oracle?.width === protocol.oracle.width && oracle?.height === protocol.oracle.height,
    'Oracle dimensions differ.');
  check(oracle?.byteLength === protocol.oracle.width * protocol.oracle.height * 16,
    'Oracle readback byte length differs.');
  check(oracle?.minimumPixelsPerTuple === protocol.oracle.minimumPixelsPerTuple,
    'Oracle coverage threshold differs.');
  check(oracle?.exact === true && oracle?.unexpected?.length === 0,
    'Oracle contains missing or unexpected integer tuples.');
  check(oracle?.resetProven === true, 'Oracle does not prove per-instance reset.');
  check(oracle?.instanceSpatialMismatches === 0,
    'Oracle instance indices do not stay in their expected screen halves.');
  check(
    Array.isArray(oracle?.primitiveIdsByInstance)
      && oracle.primitiveIdsByInstance.length === 2
      && oracle.primitiveIdsByInstance.every((ids, index) => (
        exactArray(ids, protocol.oracle.expectedPrimitiveIdsByInstance[index])
      )),
    'Primitive IDs are not exactly 0..3 in both instances.',
  );
  check(
    JSON.stringify(oracle?.expectedTuples) === JSON.stringify(expectedPrimitiveIndexTuples()),
    'Page expected tuple oracle differs from the shared contract.',
  );
  check(
    oracle?.tuples?.length === 8
      && oracle.tuples.every((record) => (
        record.sufficient === true && record.count >= protocol.oracle.minimumPixelsPerTuple
      )),
    'At least one exact primitive/instance tuple lacks sufficient pixel coverage.',
  );
  check(oracle?.backgroundPixels > 0, 'Oracle lacks a distinct zero background.');
  check(/^[a-f0-9]{64}$/u.test(oracle?.readbackSha256 ?? ''),
    'Oracle readback SHA-256 is absent.');

  check(instrumentation?.setupError === null, 'GPU instrumentation setup failed.');
  const acquisition = instrumentation?.acquisition;
  check(acquisition?.adapterRequests?.length === 1, 'Expected one adapter request.');
  check(acquisition?.deviceRequests?.length === 1, 'Expected one device request.');
  check(acquisition?.adapterFeatures?.includes(protocol.wgsl.feature),
    'Adapter does not advertise primitive-index.');
  check(acquisition?.deviceFeatures?.includes(protocol.wgsl.feature),
    'Created device does not enable primitive-index.');
  const requiredFeatures = acquisition?.deviceRequests?.[0]?.descriptor?.requiredFeatures ?? [];
  check(requiredFeatures.includes(protocol.wgsl.feature),
    'Three did not negotiate primitive-index in requiredFeatures.');
  check(requiredFeatures.filter((feature) => feature === protocol.wgsl.feature).length === 1,
    'primitive-index requiredFeatures negotiation is not unique.');

  const allEvents = (instrumentation?.drawEncoders ?? [])
    .flatMap((encoder) => encoder.events ?? []);
  const indexedEncoders = (instrumentation?.drawEncoders ?? []).filter((encoder) => (
    (encoder.events ?? []).some((event) => event.method === 'drawIndexed')
  ));
  const indexedDraws = allEvents.filter((event) => event.method === 'drawIndexed');
  check(indexedDraws.length === 2, `Expected two indexed draws, observed ${indexedDraws.length}.`);
  check(indexedDraws.every((draw) => exactArray(draw.arguments, protocol.geometry.gpuDraw)),
    'An indexed GPU draw differs from [12, 2, 0, 0, 0].');
  check(
    indexedEncoders.length === 2 && indexedEncoders.every((encoder) => {
      const events = encoder.events ?? [];
      return events.length === 2
        && events[0].method === 'setIndexBuffer'
        && events[0].format === protocol.geometry.gpuIndexFormat
        && events[1].method === 'drawIndexed'
        && exactArray(events[1].arguments, protocol.geometry.gpuDraw);
    }),
    `Expected each indexed encoder to pair one ${protocol.geometry.gpuIndexFormat} binding `
      + 'with drawIndexed(12, 2, 0, 0, 0).',
  );
  check((instrumentation?.textures ?? []).some((texture) => (
    texture.format === protocol.oracle.textureFormat && texture.sampleCount === 1
  )), 'No single-sampled rgba32uint oracle texture was created.');

  const shaderModules = instrumentation?.shaderModules ?? [];
  const primitiveModules = shaderModules.filter((module) => (
    module.code.includes(protocol.wgsl.directive)
  ));
  check(primitiveModules.length === 2,
    `Expected two primitiveIndex fragment modules, observed ${primitiveModules.length}.`);
  check(primitiveModules.every((module) => (
    occurrences(module.code, protocol.wgsl.directive) === 1
      && occurrences(module.code, protocol.wgsl.builtin) === 1
  )), 'Generated primitiveIndex WGSL directive or builtin is not exact.');
  check(shaderModules.length > 0 && shaderModules.every((module) => module.messages.length === 0),
    'Shader compilation messages were observed.');

  const pushes = instrumentation?.device?.errorScopes?.pushes ?? [];
  const pops = instrumentation?.device?.errorScopes?.pops ?? [];
  check(pushes.some((record) => record.filter === 'validation'),
    'No validation error scope surrounded the proof.');
  check(pushes.length === pops.length, 'GPU error scopes are unbalanced.');
  check(pops.every((record) => (
    record.settled === true && record.error === null && record.rejection === null
  )), 'A GPU error scope returned an error.');
  check(page?.validationScope === null, 'Page validation scope returned an error.');
  check(instrumentation?.device?.pipelinePromiseRejections?.length === 0,
    'A render pipeline promise rejected.');
  check(instrumentation?.device?.uncapturedErrors?.length === 0,
    'Uncaptured GPU errors were observed.');
  check(instrumentation?.device?.unexpectedLoss === null, 'Unexpected device loss was observed.');
  check(instrumentation?.windowErrors?.length === 0, 'Window errors were observed.');
  check(instrumentation?.unhandledRejections?.length === 0,
    'Unhandled page rejections were observed.');

  for (const key of [
    'browserDisconnects',
    'consoleErrors',
    'consoleWarnings',
    'httpErrors',
    'pageCrashes',
    'pageErrors',
    'requestFailures',
  ]) {
    check((observations?.[key]?.length ?? 0) === 0, `Browser observation ${key} is not empty.`);
  }

  return {
    indexedDraws,
    primitiveShaderModuleCount: primitiveModules.length,
    reasons,
    verified: reasons.length === 0,
  };
}

function summarizeInstrumentation(instrumentation) {
  return {
    ...instrumentation,
    shaderModules: instrumentation.shaderModules.map((module) => ({
      ...module,
      bytes: Buffer.byteLength(module.code),
      builtinCount: occurrences(
        module.code,
        '@builtin( primitive_index ) primitiveIndex : u32',
      ),
      directiveCount: occurrences(module.code, 'enable primitive_index;'),
      sha256: sha256(Buffer.from(module.code, 'utf8')),
    })),
  };
}

export function parsePrimitiveIndexArguments(args = process.argv.slice(2)) {
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

async function offlinePreflight() {
  const protocol = (await readProtocol()).value;
  const before = await validateCandidateIdentity(protocol);
  const built = await buildPrimitiveIndexCandidate(preflightBuildRoot);
  const validation = verifyBuiltPrimitiveIndexCandidate(built, protocol.build.files);
  requireCondition(validation.verified, validation.reasons.join(' '));
  const after = await validateCandidateIdentity(protocol);
  requireCondition(JSON.stringify(before) === JSON.stringify(after),
    'Candidate changed during offline preflight.');
  console.log(JSON.stringify({
    candidate: after,
    build: built.inventory,
    mode: 'offline-preflight',
    validation,
  }, null, 2));
}

async function browserCanary({ browserPath, expectedResearchCommit }) {
  const { bytes: protocolBytes, value: protocol } = await readProtocol();
  const runRoot = path.join(repoRoot, ...protocol.output.directory.split('/'));
  const reportPath = path.join(runRoot, protocol.output.report);
  const screenshotPath = path.join(runRoot, protocol.output.screenshot);
  requireCondition(browserPath === path.resolve(protocol.browser.requiredExecutablePath),
    'Browser path differs from protocol.');
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
  const [htmlBytes, moduleBytes, contractBytes] = await Promise.all([
    readFile(pagePath),
    readFile(pageModulePath),
    readFile(contractModulePath),
  ]);

  await mkdir(path.dirname(runRoot), { recursive: true });
  await mkdir(runRoot);
  let browser = null;
  let browserCloseRequested = false;
  let context = null;
  let server = null;
  const observations = {
    browserDisconnects: [],
    consoleErrors: [],
    consoleWarnings: [],
    httpErrors: [],
    pageCrashes: [],
    pageErrors: [],
    requestFailures: [],
  };
  const report = {
    schemaVersion: 1,
    kind: 'three-webgpu-primitive-index-canary-report',
    status: 'running',
    claimBoundary: protocol.claimBoundary,
    protocol: { bytes: protocolBytes.length, sha256: sha256(protocolBytes) },
    research: { before: researchBefore, expectedCommit: expectedResearchCommit },
    candidate: { before: candidateBefore, after: null },
    build: null,
    browser: null,
    pageSource: {
      html: { bytes: htmlBytes.length, sha256: sha256(htmlBytes) },
      module: { bytes: moduleBytes.length, sha256: sha256(moduleBytes) },
      contract: { bytes: contractBytes.length, sha256: sha256(contractBytes) },
    },
    evidence: { initial: null, late: null },
    validation: null,
    screenshot: null,
    observations,
    cleanup: {
      browser: { attempted: false, closed: false, error: null },
      context: { attempted: false, closed: false, error: null },
      server: { attempted: false, closed: false, error: null },
    },
    failure: null,
  };

  try {
    const built = await buildPrimitiveIndexCandidate(liveBuildRoot);
    const buildValidation = verifyBuiltPrimitiveIndexCandidate(built, protocol.build.files);
    requireCondition(buildValidation.verified, buildValidation.reasons.join(' '));
    report.build = { files: built.inventory, validation: buildValidation };

    server = await createServer({
      appType: 'custom',
      cacheDir: path.join(liveBuildRoot, '.vite-cache'),
      configFile: false,
      logLevel: 'error',
      optimizeDeps: { noDiscovery: true },
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
      productVersion: browser.version(),
    };

    context = await browser.newContext({
      deviceScaleFactor: protocol.page.deviceScaleFactor,
      serviceWorkers: 'block',
      viewport: {
        width: protocol.page.viewport[0],
        height: protocol.page.viewport[1],
      },
    });
    const served = new Map([
      [`/${protocol.page.path}`, { body: htmlBytes, type: 'text/html; charset=utf-8' }],
      [`/${protocol.page.module}`, { body: moduleBytes, type: 'text/javascript; charset=utf-8' }],
      [`/${protocol.page.contract}`, { body: contractBytes, type: 'text/javascript; charset=utf-8' }],
    ]);
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== baseUrl) {
        await route.abort('blockedbyclient');
        return;
      }
      if (served.has(url.pathname)) {
        const response = served.get(url.pathname);
        await route.fulfill({
          body: response.body,
          contentType: response.type,
          headers: ISOLATION_HEADERS,
          status: 200,
        });
        return;
      }
      if (url.pathname.startsWith(`${BUILD_ROUTE}/`)) {
        const filename = url.pathname.slice(BUILD_ROUTE.length + 1);
        requireCondition(Object.hasOwn(built.modules, filename),
          `Unexpected built module request: ${filename}`);
        await route.fulfill({
          body: built.modules[filename],
          contentType: 'text/javascript; charset=utf-8',
          headers: ISOLATION_HEADERS,
          status: 200,
        });
        return;
      }
      await route.abort('blockedbyclient');
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
    await page.addInitScript(installPrimitiveIndexInstrumentation);
    await page.goto(`${baseUrl}/${protocol.page.path}`, {
      timeout: 60_000,
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      () => (window.__threePrimitiveIndexCanary !== undefined
        && window.__threePrimitiveIndexCanary.status !== 'running')
        || (window.__threePrimitiveIndexInstrumentation?.windowErrors?.length ?? 0) > 0
        || (window.__threePrimitiveIndexInstrumentation?.unhandledRejections?.length ?? 0) > 0,
      undefined,
      { timeout: protocol.page.timeoutMilliseconds },
    );
    const pageEvidence = await page.evaluate(() => (
      structuredClone(window.__threePrimitiveIndexCanary)
    ));
    const instrumentation = await withDeadline(
      page.evaluate(() => window.__collectThreePrimitiveIndexInstrumentation()),
      protocol.page.timeoutMilliseconds,
      'GPU evidence collection',
    );
    const rawEvidence = { page: pageEvidence, instrumentation };
    report.evidence.initial = {
      page: pageEvidence,
      instrumentation: summarizeInstrumentation(instrumentation),
    };
    const initializationError = instrumentation.windowErrors?.[0]
      ?? observations.pageErrors[0]
      ?? null;
    if (pageEvidence === undefined && initializationError !== null) {
      throw new Error(`Page initialization failed: ${initializationError.message}`);
    }
    report.validation = validatePrimitiveIndexEvidence(rawEvidence, protocol, observations);
    requireCondition(report.validation.verified, report.validation.reasons.join(' '));

    const screenshotBytes = await page.screenshot({ fullPage: true });
    const screenshotDimensions = pngDimensions(screenshotBytes);
    requireCondition(
      screenshotDimensions.width === protocol.page.viewport[0]
        && screenshotDimensions.height === protocol.page.viewport[1],
      'Screenshot dimensions differ from protocol viewport.',
    );
    requireCondition(screenshotBytes.length > 10_000, 'Screenshot is unexpectedly small.');
    await writeFile(screenshotPath, screenshotBytes, { flag: 'wx' });
    report.screenshot = {
      bytes: screenshotBytes.length,
      ...screenshotDimensions,
      path: path.relative(repoRoot, screenshotPath).replaceAll('\\', '/'),
      sha256: sha256(screenshotBytes),
    };
    await page.waitForTimeout(100);
    const lateInstrumentation = await page.evaluate(() => (
      structuredClone(window.__threePrimitiveIndexInstrumentation)
    ));
    const { shaderModules: ignoredShaderModules, ...initialInstrumentationState } =
      instrumentation;
    void ignoredShaderModules;
    requireCondition(
      isDeepStrictEqual(lateInstrumentation, initialInstrumentationState),
      'Instrumented GPU or page state changed after initial evidence collection.',
    );
    report.evidence.late = lateInstrumentation;
    report.validation = validatePrimitiveIndexEvidence(rawEvidence, protocol, observations);
    requireCondition(report.validation.verified, report.validation.reasons.join(' '));

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
    report.status = 'failed';
    report.failure = serializeError(error);
  } finally {
    try {
      report.cleanup.context.attempted = context !== null;
      if (context !== null) {
        await withDeadline(context.close(), TEARDOWN_TIMEOUT_MS, 'Browser context teardown');
        report.cleanup.context.closed = true;
      }
    } catch (error) {
      report.cleanup.context.error = serializeError(error);
      report.status = 'failed';
      report.failure ??= serializeError(error);
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
      report.status = 'failed';
      report.failure ??= serializeError(error);
    }
    try {
      report.cleanup.server.attempted = server !== null;
      if (server !== null) {
        await withDeadline(server.close(), TEARDOWN_TIMEOUT_MS, 'Vite teardown');
        report.cleanup.server.closed = true;
      }
    } catch (error) {
      report.cleanup.server.error = serializeError(error);
      report.status = 'failed';
      report.failure ??= serializeError(error);
    }
  }

  if (report.status === 'passed') {
    const cleanupComplete = Object.values(report.cleanup).every((entry) => (
      entry.attempted === true && entry.closed === true && entry.error === null
    ));
    if (!cleanupComplete) {
      report.status = 'failed';
      report.failure = serializeError(new Error('Browser canary teardown was incomplete.'));
    }
  }
  const reportBytes = canonicalJson(report);
  await writeFile(reportPath, reportBytes, { flag: 'wx' });
  const reportHash = sha256(reportBytes);
  await writeFile(`${reportPath}.sha256`, `${reportHash}  report.json\n`, { flag: 'wx' });
  if (await exists(screenshotPath)) await chmod(screenshotPath, 0o444);
  await chmod(reportPath, 0o444);
  await chmod(`${reportPath}.sha256`, 0o444);
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
  const arguments_ = parsePrimitiveIndexArguments();
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

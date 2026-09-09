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

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidateRoot = path.join(
  repoRoot,
  '.generated',
  'three-draw-index-upstream-candidate',
);
const buildRoot = path.join(repoRoot, '.generated', 'three-draw-index-v3-live-build');
const protocolPath = path.join(
  repoRoot,
  'protocols',
  'three-draw-index-v3-owned-device-v1.json',
);
const manifestPath = path.join(
  repoRoot,
  'patches',
  'three-webgpu-draw-index-immediates-v2.json',
);
const exampleRelativePath = 'examples/webgpu_tsl_draw_index.html';
const examplePath = path.join(candidateRoot, ...exampleRelativePath.split('/'));
const GIT_COMMIT_PATTERN = /^[a-f0-9]{40}$/u;
const EXPECTED_PATCH_FILES = 15;
const EXPECTED_BUILD_FILES = [
  'three.core.js',
  'three.module.js',
  'three.tsl.js',
  'three.webgpu.js',
  'three.webgpu.nodes.js',
];
const INITIAL_COLLECTION_TIMEOUT_MS = 120_000;
const SNAPSHOT_TIMEOUT_MS = 30_000;
const FAILURE_FORENSICS_TIMEOUT_MS = 5_000;
const TEARDOWN_TIMEOUT_MS = 10_000;
const isolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
};

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
  return source.split(token).length - 1;
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function withDeadline(promise, timeoutMs, label) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(`${label} exceeded ${timeoutMs} ms.`)), timeoutMs);
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
  requireCondition(bytes.subarray(0, 8).equals(signature), 'Screenshot lacks the PNG signature.');
  requireCondition(bytes.toString('ascii', 12, 16) === 'IHDR', 'Screenshot lacks a PNG IHDR.');
  return { height: bytes.readUInt32BE(20), width: bytes.readUInt32BE(16) };
}

async function readBoundArtifact(specification, label) {
  const artifactPath = path.join(repoRoot, ...specification.path.split('/'));
  const bytes = await readFile(artifactPath);
  requireCondition(bytes.length === specification.bytes, `${label} byte count mismatch.`);
  requireCondition(sha256(bytes) === specification.sha256, `${label} hash mismatch.`);
  return { bytes, path: artifactPath };
}

async function validatePriorAttempt(priorAttempt) {
  const reportArtifact = await readBoundArtifact(priorAttempt.report, 'Prior v2 report');
  const sidecarArtifact = await readBoundArtifact(priorAttempt.sidecar, 'Prior v2 report sidecar');
  const screenshotArtifact = await readBoundArtifact(
    priorAttempt.screenshot,
    'Prior v2 screenshot',
  );
  const priorReport = JSON.parse(reportArtifact.bytes.toString('utf8'));
  requireCondition(priorReport.status === priorAttempt.expectedStatus, 'Prior v2 status mismatch.');
  requireCondition(
    priorReport.failure?.message === priorAttempt.expectedFailure,
    'Prior v2 failure identity mismatch.',
  );
  requireCondition(
    sidecarArtifact.bytes.toString('utf8') === `${priorAttempt.report.sha256}  report.json\n`,
    'Prior v2 report sidecar content mismatch.',
  );
  const dimensions = pngDimensions(screenshotArtifact.bytes);
  requireCondition(
    dimensions.width === priorAttempt.screenshot.width
      && dimensions.height === priorAttempt.screenshot.height,
    'Prior v2 screenshot dimensions mismatch.',
  );
  return {
    relationship: priorAttempt.relationship,
    report: priorAttempt.report,
    sidecar: priorAttempt.sidecar,
    screenshot: priorAttempt.screenshot,
    status: priorReport.status,
    failure: {
      name: priorReport.failure?.name ?? null,
      message: priorReport.failure?.message ?? null,
    },
  };
}

export function prepareExampleSource(source, buildUrlRoot) {
  const webGPUImport = '../build/three.webgpu.js';
  const tslImport = '../build/three.tsl.js';
  const renderNeedle = '\t\t\t\trenderer.render( scene, camera );';
  requireCondition(occurrences(source, webGPUImport) === 2, 'Example WebGPU imports are ambiguous.');
  requireCondition(occurrences(source, tslImport) === 1, 'Example TSL import is ambiguous.');
  requireCondition(occurrences(source, renderNeedle) === 1, 'Example render hook is ambiguous.');

  const body = source
    .replaceAll(webGPUImport, `${buildUrlRoot}/three.webgpu.js`)
    .replaceAll(tslImport, `${buildUrlRoot}/three.tsl.js`)
    .replace(renderNeedle, `${renderNeedle}\n\n\t\t\t\tconst renderedBatches = scene.children.filter( ( child ) => child.isBatchedMesh === true );\n\t\t\t\twindow.__drawIndexV3State.ready = {\n\t\t\t\t\tbackendIsWebGPU: renderer.backend.isWebGPUBackend === true,\n\t\t\t\t\tbatchCounts: renderedBatches.map( ( batchMesh ) => batchMesh._multiDrawCount ),\n\t\t\t\t\tbatchOrders: renderedBatches.map( ( batchMesh ) => Array.from( batchMesh._indirectTexture.image.data.slice( 0, batchMesh._multiDrawCount ) ) ),\n\t\t\t\t\tcapability: renderer.backend.capabilities.hasImmediateData(),\n\t\t\t\t\tdeviceLimit: renderer.backend.device.limits.maxImmediateSize,\n\t\t\t\t\townedDevice: window.__drawIndexV3OwnsDevice( renderer.backend.device ),\n\t\t\t\t\tphase\n\t\t\t\t};\n\t\t\t\trenderer.setAnimationLoop( null );`);

  requireCondition(occurrences(body, `${buildUrlRoot}/three.webgpu.js`) === 2, 'WebGPU import rewrite failed.');
  requireCondition(occurrences(body, `${buildUrlRoot}/three.tsl.js`) === 1, 'TSL import rewrite failed.');
  requireCondition(occurrences(body, 'window.__drawIndexV3State.ready = {') === 1, 'Ready hook injection failed.');
  return body;
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

async function runGit(root, args, options = {}) {
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
    { ...execOptions, ...options },
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
    commit: commit.trim(),
    tree: tree.trim(),
    branch: branch.trim(),
    status: status.trim(),
  };
}

async function candidatePatchBytes() {
  const { stdout } = await runGit(
    candidateRoot,
    ['diff', 'HEAD', '--full-index', '--binary'],
    { encoding: null },
  );
  return stdout;
}

export async function buildCandidate(targetRoot = buildRoot) {
  requireCondition(
    await exists(targetRoot) === false,
    'The single-use live build directory already exists.',
  );

  const rollupModulePath = path.join(
    candidateRoot,
    'node_modules',
    'rollup',
    'dist',
    'es',
    'rollup.js',
  );
  requireCondition(
    await exists(rollupModulePath),
    'Candidate dependencies are not installed.',
  );
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

export function findForbiddenArguments(arguments_, prefixes) {
  return arguments_.filter((argument) => {
    const normalized = String(argument).toLowerCase();
    return prefixes.some((prefix) => (
      normalized === prefix || normalized.startsWith(`${prefix}=`)
    ));
  });
}

function installPageInstrumentation() {
  const PAGE_STAGE_TIMEOUT_MS = 30_000;
  const state = {
    acquisition: {
      adapterRequests: [],
      deviceRequests: [],
      adapterFeatures: [],
      adapterInfo: null,
    },
    capabilities: {
      wgslLanguageFeatures: [],
      maxImmediateSize: null,
    },
    device: {
      callableSetImmediates: false,
      errorScopes: { pops: [], pushes: [] },
      pipelinePromiseRejections: [],
      uncapturedErrors: [],
      unexpectedLoss: null,
    },
    pipelineLayouts: [],
    renderPipelines: [],
    renderPasses: [],
    ready: null,
    setupError: null,
    snapshot: {
      collectionCalls: 0,
      compilationInfo: [],
      progress: { stage: 'idle', status: 'not-started' },
    },
    windowErrors: [],
    unhandledRejections: [],
  };
  Object.defineProperty(window, '__drawIndexV3State', { value: state });

  window.addEventListener('error', (event) => {
    state.windowErrors.push({
      message: String(event.message ?? ''),
      filename: String(event.filename ?? ''),
      line: Number(event.lineno ?? 0),
      column: Number(event.colno ?? 0),
    });
  });
  window.addEventListener('unhandledrejection', (event) => {
    state.unhandledRejections.push(String(event.reason?.message ?? event.reason));
  });

  const shaderModules = [];
  const layoutSizes = new WeakMap();
  const pipelineSizes = new WeakMap();
  const pendingErrorScopes = [];
  const scopeStack = [];
  let nextScopeId = 0;
  let activeDevice = null;

  const serializeDescriptor = (descriptor) => {
    if (descriptor === undefined) return null;
    const result = {};
    for (const [key, value] of Object.entries(descriptor)) {
      if (value === undefined) continue;
      if (key === 'requiredFeatures') {
        result[key] = Array.from(value, String).sort();
      } else if (key === 'requiredLimits') {
        result[key] = Object.fromEntries(
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

  const selectedImmediate = (args) => {
    const [rangeOffset, values, suppliedOffset, suppliedSize] = args;
    const dataOffset = suppliedOffset ?? 0;
    const dataSize = suppliedSize ?? (values.length - dataOffset);
    return {
      rangeOffset: Number(rangeOffset),
      dataOffset: Number(dataOffset),
      dataSize: Number(dataSize),
      sourceType: values?.constructor?.name ?? null,
      values: values instanceof Uint32Array
        ? Array.from(values.slice(dataOffset, dataOffset + dataSize))
        : null,
    };
  };

  const instrumentPass = (pass) => {
    const record = {
      callableSetImmediates: typeof pass.setImmediates === 'function',
      events: [],
    };
    state.renderPasses.push(record);
    state.device.callableSetImmediates ||= record.callableSetImmediates;
    let currentPipelineSize = null;
    let lastMethod = null;
    let lastImmediate = null;

    replaceCallable(pass, 'setPipeline', (original) => (pipeline) => {
      currentPipelineSize = pipelineSizes.get(pipeline) ?? null;
      record.events.push({ method: 'setPipeline', immediateSize: currentPipelineSize });
      lastMethod = 'setPipeline';
      return original(pipeline);
    });
    replaceCallable(pass, 'setImmediates', (original) => (...args) => {
      lastImmediate = selectedImmediate(args);
      record.events.push({
        method: 'setImmediates',
        immediateSize: currentPipelineSize,
        ...lastImmediate,
      });
      lastMethod = 'setImmediates';
      return original(...args);
    });
    for (const method of ['draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']) {
      replaceCallable(pass, method, (original) => (...args) => {
        record.events.push({
          method,
          arguments: args.map((value) => (
            typeof value === 'number' ? value : value?.constructor?.name ?? null
          )),
          immediateSize: currentPipelineSize,
          adjacentImmediate: lastMethod === 'setImmediates',
          immediate: lastImmediate,
        });
        lastMethod = method;
        return original(...args);
      });
    }
    return record;
  };

  const instrumentDevice = (device) => {
    activeDevice = device;
    state.capabilities.maxImmediateSize = Number.isFinite(
      Number(device.limits?.maxImmediateSize),
    ) ? Number(device.limits.maxImmediateSize) : null;

    device.addEventListener?.('uncapturederror', (event) => {
      event.preventDefault();
      state.device.uncapturedErrors.push({
        name: String(event.error?.constructor?.name ?? 'GPUError'),
        message: String(event.error?.message ?? event.error),
      });
    });
    void device.lost.then((info) => {
      state.device.unexpectedLoss = {
        reason: String(info.reason ?? ''),
        message: String(info.message ?? ''),
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
      pendingErrorScopes.push(Promise.resolve(promise).then((error) => {
        record.error = error === null ? null : {
          message: String(error?.message ?? error),
          name: String(error?.constructor?.name ?? 'GPUError'),
        };
        record.settled = true;
      }, (error) => {
        record.rejection = {
          message: String(error?.message ?? error).slice(0, 4_096),
          name: String(error?.name ?? 'Error').slice(0, 256),
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
      const compilation = {
        calls: 0,
        index: shaderModules.length,
        instrumented: false,
        label: descriptor?.label ?? null,
        status: 'not-requested',
      };
      compilation.instrumented = replaceCallable(
        module,
        'getCompilationInfo',
        (getCompilationInfo) => (...args) => {
          compilation.calls += 1;
          return getCompilationInfo(...args);
        },
      );
      if (compilation.instrumented === false) {
        throw new Error('Could not instrument GPUShaderModule.getCompilationInfo.');
      }
      state.snapshot.compilationInfo.push(compilation);
      shaderModules.push({
        code: String(descriptor?.code ?? ''),
        label: descriptor?.label ?? null,
        module,
        compilation,
        compilationInfoPromise: null,
      });
      return module;
    });
    replaceCallable(device, 'createPipelineLayout', (original) => (descriptor) => {
      const immediateSize = Number(descriptor?.immediateSize ?? 0);
      const layout = original(descriptor);
      layoutSizes.set(layout, immediateSize);
      state.pipelineLayouts.push({
        immediateSize,
        hasOwnImmediateSize: Object.hasOwn(descriptor, 'immediateSize'),
        bindGroupLayoutCount: descriptor?.bindGroupLayouts?.length ?? null,
      });
      return layout;
    });

    const instrumentPipeline = (name) => {
      replaceCallable(device, name, (original) => (descriptor) => {
        const immediateSize = layoutSizes.get(descriptor?.layout) ?? null;
        const result = original(descriptor);
        const record = { method: name, immediateSize, label: descriptor?.label ?? null };
        state.renderPipelines.push(record);
        if (result?.then instanceof Function) {
          void result.then(
            (pipeline) => pipelineSizes.set(pipeline, immediateSize),
            (error) => state.device.pipelinePromiseRejections.push({
              message: String(error?.message ?? error),
              name: String(error?.name ?? 'Error'),
            }),
          );
        } else {
          pipelineSizes.set(result, immediateSize);
        }
        return result;
      });
    };
    instrumentPipeline('createRenderPipeline');
    instrumentPipeline('createRenderPipelineAsync');

    replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
      const commandEncoder = original(descriptor);
      replaceCallable(commandEncoder, 'beginRenderPass', (begin) => (passDescriptor) => {
        const pass = begin(passDescriptor);
        instrumentPass(pass);
        return pass;
      });
      return commandEncoder;
    });
  };

  try {
    const gpu = navigator.gpu;
    state.capabilities.wgslLanguageFeatures = gpu?.wgslLanguageFeatures == null
      ? []
      : Array.from(gpu.wgslLanguageFeatures, String).sort();
    if (gpu === undefined) throw new Error('navigator.gpu is unavailable.');

    const installed = replaceCallable(gpu, 'requestAdapter', (requestAdapter) => async (...args) => {
      state.acquisition.adapterRequests.push({
        argumentsCount: args.length,
        descriptor: serializeDescriptor(args[0]),
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
            descriptor: serializeDescriptor(deviceArgs[0]),
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
      name: String(error?.name ?? 'Error'),
      message: String(error?.message ?? error),
    };
  }

  const awaitStage = async (promise, stage) => {
    let timeoutId;
    state.snapshot.progress = { stage, status: 'pending' };
    const timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        const pendingShaderIndices = state.snapshot.compilationInfo
          .filter((record) => record.status === 'pending')
          .map((record) => record.index);
        const error = new Error(
          `Snapshot stage "${stage}" exceeded ${PAGE_STAGE_TIMEOUT_MS} ms; pending shaders: ${pendingShaderIndices.join(',') || 'none'}.`,
        );
        error.name = 'SnapshotStageTimeoutError';
        reject(error);
      }, PAGE_STAGE_TIMEOUT_MS);
    });
    try {
      const value = await Promise.race([promise, timeout]);
      state.snapshot.progress = { stage, status: 'complete' };
      return value;
    } catch (error) {
      state.snapshot.progress = {
        message: String(error?.message ?? error),
        stage,
        status: 'failed',
      };
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const collectInitialEvidence = async () => {
    if (activeDevice !== null) {
      await awaitStage(
        Promise.resolve().then(() => activeDevice.queue.onSubmittedWorkDone()),
        'queue-settlement',
      );
    }
    await awaitStage(Promise.allSettled(pendingErrorScopes), 'error-scope-settlement');
    state.snapshot.progress = { stage: 'quiet-period', status: 'pending' };
    await new Promise((resolve) => setTimeout(resolve, 50));
    state.snapshot.progress = { stage: 'quiet-period', status: 'complete' };

    const modules = [];
    for (const record of shaderModules) {
      if (record.compilation.calls !== 0 || record.compilationInfoPromise !== null) {
        throw new Error(`Shader ${record.compilation.index} compilation info was requested before the one-shot collector.`);
      }
      record.compilation.status = 'pending';
      try {
        record.compilationInfoPromise = record.module.getCompilationInfo();
        if (record.compilation.calls !== 1) {
          throw new Error(`Shader ${record.compilation.index} compilation info call was not observed exactly once.`);
        }
      } catch (error) {
        record.compilation.status = 'rejected';
        modules.push({
          code: record.code,
          label: record.label,
          messages: [{ type: 'instrumentation-error', message: String(error) }],
        });
        continue;
      }
      try {
        const info = await awaitStage(
          Promise.resolve(record.compilationInfoPromise),
          `shader-compilation-info:${record.compilation.index}`,
        );
        record.compilation.status = 'fulfilled';
        modules.push({
          code: record.code,
          label: record.label,
          messages: Array.from(info.messages, (message) => ({
            type: String(message.type),
            message: String(message.message),
            lineNum: Number(message.lineNum ?? 0),
            linePos: Number(message.linePos ?? 0),
          })),
        });
      } catch (error) {
        record.compilation.status = error?.name === 'SnapshotStageTimeoutError'
          ? 'timed-out'
          : 'rejected';
        if (error?.name === 'SnapshotStageTimeoutError') throw error;
        modules.push({
          code: record.code,
          label: record.label,
          messages: [{ type: 'instrumentation-error', message: String(error) }],
        });
      }
    }
    state.snapshot.progress = { stage: 'complete', status: 'complete' };
    return structuredClone({ ...state, shaderModules: modules });
  };

  let initialCollectionPromise = null;
  window.__drawIndexV3CollectInitialEvidence = () => {
    state.snapshot.collectionCalls += 1;
    if (initialCollectionPromise !== null) {
      throw new Error('Initial evidence collector may only be called once.');
    }
    initialCollectionPromise = collectInitialEvidence();
    return initialCollectionPromise;
  };
  window.__drawIndexV3LateState = () => structuredClone({
    ...state,
    shaderModuleCount: shaderModules.length,
    shaderModuleLabels: shaderModules.map((record) => record.label),
  });
  window.__drawIndexV3OwnsDevice = (device) => activeDevice !== null && device === activeDevice;
}

export function validateEvidence(evidence, protocol, observations) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };

  check(evidence.setupError === null, 'Page instrumentation setup failed.');
  check(evidence.acquisition.adapterRequests.length === 1, 'Expected one adapter request.');
  check(evidence.acquisition.deviceRequests.length === 1, 'Expected one device request.');
  check(
    evidence.acquisition.adapterInfo?.isFallbackAdapter === false,
    'Renderer acquired a fallback adapter or did not expose adapter identity.',
  );
  const adapterRequest = evidence.acquisition.adapterRequests[0];
  check(adapterRequest?.argumentsCount === 1, 'Renderer adapter request shape changed.');
  check(
    adapterRequest?.descriptor?.featureLevel === 'compatibility',
    'Renderer did not use its ordinary compatibility feature level.',
  );
  const deviceRequest = evidence.acquisition.deviceRequests[0];
  check(deviceRequest?.argumentsCount === 1, 'Renderer device request shape changed.');
  check(
    Object.hasOwn(deviceRequest?.descriptor?.requiredLimits ?? {}, 'maxImmediateSize') === false,
    'Renderer promoted maxImmediateSize.',
  );
  check(
    evidence.capabilities.wgslLanguageFeatures.includes('immediate_address_space'),
    'WGSL immediate_address_space is unavailable.',
  );
  check(evidence.capabilities.maxImmediateSize >= 4, 'Device maxImmediateSize is below four.');
  check(evidence.device.callableSetImmediates === true, 'Render pass lacks setImmediates.');
  check(
    evidence.device.pipelinePromiseRejections.length === 0,
    'A render pipeline promise rejected.',
  );
  check(evidence.device.uncapturedErrors.length === 0, 'Uncaptured GPU errors were observed.');
  check(evidence.device.unexpectedLoss === null, 'Unexpected device loss was observed.');
  const scopePushes = evidence.device.errorScopes?.pushes ?? [];
  const scopePops = evidence.device.errorScopes?.pops ?? [];
  check(scopePushes.length >= 3, 'Expected validation scopes around render pipelines.');
  check(scopePops.length === scopePushes.length, 'GPU error scopes are unbalanced.');
  check(
    scopePushes.every((push, index) => (
      push.id === index && push.filter === 'validation'
    )),
    'GPU error-scope pushes differ from the expected validation sequence.',
  );
  check(
    exactArray(
      scopePops.map((pop) => pop.id).sort((a, b) => a - b),
      scopePushes.map((push) => push.id),
    )
      && scopePops.every((pop) => (
        pop.filter === 'validation'
      && pop.settled === true
      && pop.error === null
      && pop.rejection === null
      )),
    'GPU validation scopes did not drain cleanly with matching identities.',
  );

  check(evidence.ready?.backendIsWebGPU === true, 'Example did not use the WebGPU backend.');
  check(evidence.ready?.capability === true, 'Example capability gate did not pass.');
  check(evidence.ready?.ownedDevice === true, 'Renderer did not retain the instrumented device.');
  check(
    evidence.ready?.deviceLimit === evidence.capabilities.maxImmediateSize
      && evidence.ready.deviceLimit >= 4,
    'Renderer device limit differs from the instrumented device.',
  );
  check(evidence.ready?.phase === protocol.example.phase, 'Example phase differs from protocol.');
  check(
    exactArray(evidence.ready?.batchCounts, [8, 8]),
    'Example batches did not emit eight commands each.',
  );
  check(
    evidence.ready?.batchOrders?.length === 2
      && evidence.ready.batchOrders.every((order) => exactArray(order, protocol.example.order)),
    'Actual BatchedMesh command order differs from protocol.',
  );

  const layoutSizes = evidence.pipelineLayouts.map((layout) => layout.immediateSize);
  check(
    exactArray(layoutSizes.slice().sort((a, b) => a - b), [0, 0, 4]),
    'Expected two ordinary and one immediate pipeline layouts.',
  );
  const pipelineSizes = evidence.renderPipelines.map((pipeline) => pipeline.immediateSize);
  check(
    exactArray(pipelineSizes.slice().sort((a, b) => a - b), [0, 0, 4]),
    'Expected two ordinary and one immediate render pipelines.',
  );

  const sceneEvents = evidence.renderPasses[0]?.events ?? [];
  const outputEvents = evidence.renderPasses[1]?.events ?? [];
  const events = evidence.renderPasses.flatMap((pass) => pass.events);
  const immediates = events.filter((event) => event.method === 'setImmediates');
  const draws = events.filter((event) => event.method === 'drawIndexed');
  const otherDraws = events.filter((event) => (
    ['draw', 'drawIndirect', 'drawIndexedIndirect'].includes(event.method)
  ));
  const immediateDraws = draws.filter((draw) => draw.immediateSize === 4);
  const ordinaryDraws = draws.filter((draw) => draw.immediateSize === 0);
  check(
    evidence.renderPasses.length === 2,
    `Expected scene and output render passes, observed ${evidence.renderPasses.length}.`,
  );
  check(
    sceneEvents.filter((event) => event.method === 'setImmediates').length === 8
      && sceneEvents.filter((event) => event.method === 'drawIndexed').length === 16
      && sceneEvents.every((event) => event.method !== 'draw'),
    'The scene pass does not own the exact batch work.',
  );
  const outputWork = outputEvents.filter((event) => (
    ['setImmediates', 'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect']
      .includes(event.method)
  ));
  check(
    outputWork.length === 1
      && outputWork[0] === otherDraws[0]
      && outputWork[0].method === 'draw',
    'The output pass contains unexpected draw work.',
  );
  check(
    otherDraws.length === 1
      && otherDraws[0].method === 'draw'
      && otherDraws[0].immediateSize === 0
      && otherDraws[0].adjacentImmediate === false
      && exactArray(otherDraws[0].arguments, [3, 1, 0, 0]),
    `Expected one ordinary output-transform draw, observed ${otherDraws.length}.`,
  );
  check(immediates.length === 8, `Expected 8 immediate setters, observed ${immediates.length}.`);
  check(immediateDraws.length === 8, `Expected 8 immediate draws, observed ${immediateDraws.length}.`);
  check(ordinaryDraws.length === 8, `Expected 8 ordinary draws, observed ${ordinaryDraws.length}.`);
  check(
    immediates.every((event, index) => (
      event.rangeOffset === 0
      && event.immediateSize === 4
      && event.dataOffset === 0
      && event.dataSize === 1
      && event.sourceType === 'Uint32Array'
      && exactArray(event.values, [index])
    )),
    'Immediate setters do not carry exact ordinals zero through seven.',
  );
  check(
    immediateDraws.every((draw, index) => (
      draw.adjacentImmediate === true
      && exactArray(draw.immediate?.values, [index])
      && exactArray(draw.arguments, [36, 1, 0, 0, index])
    )),
    'An immediate draw lacks its exact command or adjacent setter.',
  );
  check(
    ordinaryDraws.every((draw, index) => (
      draw.adjacentImmediate === false
      && exactArray(draw.arguments, [36, 1, 0, 0, index])
    )),
    'An ordinary batch draw has the wrong command or an immediate setter.',
  );

  const immediateModules = evidence.shaderModules.filter((module) => (
    module.code.includes('requires immediate_address_space;')
  ));
  check(immediateModules.length === 1, 'Expected exactly one immediate shader module.');
  check(
    immediateModules.every((module) => (
      occurrences(module.code, 'requires immediate_address_space;') === 1
      && occurrences(module.code, 'var<immediate> nodeDrawIndex : u32;') === 1
    )),
    'Immediate WGSL requirement or declaration is not exact.',
  );
  check(
    evidence.shaderModules.every((module) => module.messages.length === 0),
    'Shader compilation messages were observed.',
  );
  check(evidence.snapshot?.collectionCalls === 1, 'Initial evidence collector did not run exactly once.');
  check(
    evidence.snapshot?.progress?.stage === 'complete'
      && evidence.snapshot.progress.status === 'complete',
    'Initial evidence collector did not finish exactly.',
  );
  check(
    evidence.snapshot?.compilationInfo?.length === evidence.shaderModules.length
      && evidence.snapshot.compilationInfo.every((record, index) => (
        record.index === index
        && record.calls === 1
        && record.instrumented === true
        && record.status === 'fulfilled'
      )),
    'Shader compilation info was not collected exactly once per module.',
  );
  check(evidence.windowErrors.length === 0, 'Window errors were observed.');
  check(evidence.unhandledRejections.length === 0, 'Unhandled rejections were observed.');
  check(observations.browserDisconnects.length === 0, 'Unexpected browser disconnects were observed.');
  check(observations.pageCrashes.length === 0, 'Page crashes were observed.');
  check(observations.pageErrors.length === 0, 'Playwright page errors were observed.');
  check(observations.consoleErrors.length === 0, 'Console errors were observed.');
  check(observations.consoleWarnings.length === 0, 'Console warnings were observed.');
  check(observations.requestFailures.length === 0, 'Request failures were observed.');
  check(observations.httpErrors.length === 0, 'HTTP errors were observed.');

  return {
    events: {
      immediateSetters: immediates,
      immediateDraws,
      ordinaryDraws,
    },
    reasons,
    verified: reasons.length === 0,
  };
}

export function summarizeEvidence(evidence) {
  return {
    ...evidence,
    shaderModules: evidence.shaderModules.map((module) => ({
      bytes: Buffer.byteLength(module.code),
      immediateDeclarationCount: occurrences(
        module.code,
        'var<immediate> nodeDrawIndex : u32;',
      ),
      immediateRequirementCount: occurrences(
        module.code,
        'requires immediate_address_space;',
      ),
      label: module.label,
      messages: module.messages,
      sha256: sha256(Buffer.from(module.code, 'utf8')),
    })),
  };
}

export function validateLateState(initialEvidence, lateState) {
  const reasons = [];
  const check = (condition, message) => {
    if (!condition) reasons.push(message);
  };
  if (lateState === null || typeof lateState !== 'object' || Array.isArray(lateState)) {
    return { reasons: ['Late synchronous state is not an object.'], verified: false };
  }
  const { shaderModules, ...initialState } = initialEvidence;
  const {
    shaderModuleCount,
    shaderModuleLabels,
    ...lateComparableState
  } = lateState;
  check(
    shaderModuleCount === shaderModules.length,
    'Shader module count changed after the initial checkpoint.',
  );
  check(
    exactArray(shaderModuleLabels, shaderModules.map((module) => module.label)),
    'Shader module identities changed after the initial checkpoint.',
  );
  check(
    isDeepStrictEqual(lateComparableState, initialState),
    'Instrumented page or GPU state changed after the initial checkpoint.',
  );
  return { reasons, verified: reasons.length === 0 };
}

export function createPreCollectionCheckpoint(report) {
  requireCondition(
    report.evidence?.preCollection !== null
      && typeof report.evidence?.preCollection === 'object'
      && Array.isArray(report.evidence.preCollection) === false,
    'Pre-collection state is unavailable.',
  );
  return structuredClone({
    schemaVersion: 1,
    kind: 'three-webgpu-draw-index-v3-pre-collection-checkpoint',
    status: 'pre-collection-not-final',
    claimBoundary: report.claimBoundary,
    protocol: report.protocol,
    priorAttempt: report.priorAttempt,
    research: report.research,
    source: report.source,
    build: report.build,
    browser: report.browser,
    example: report.example,
    observations: report.observations,
    evidence: report.evidence.preCollection,
    pending: [
      'GPU queue and validation-scope settlement',
      'one compilation-info request per shader module',
      'initial semantic validation and checkpoint',
      'screenshot, late-state, postflight, and teardown checks',
    ],
  });
}

export function createInitialCheckpoint(report, initialValidation) {
  requireCondition(
    report.evidence?.initial !== null
      && typeof report.evidence?.initial === 'object'
      && Array.isArray(report.evidence.initial) === false,
    'Initial evidence is unavailable.',
  );
  return structuredClone({
    schemaVersion: 1,
    kind: 'three-webgpu-draw-index-v3-initial-checkpoint',
    status: 'checkpointed-not-final',
    claimBoundary: report.claimBoundary,
    protocol: report.protocol,
    priorAttempt: report.priorAttempt,
    research: report.research,
    source: report.source,
    build: report.build,
    browser: report.browser,
    example: report.example,
    observations: report.observations,
    evidence: report.evidence.initial,
    preCollectionCheckpoint: report.preCollectionCheckpoint,
    initialValidation,
    pending: [
      'screenshot capture and validation',
      'late synchronous page and GPU state comparison',
      'source, build, browser, and research postflight checks',
      'browser, context, and server teardown checks',
    ],
  });
}

function parseArguments() {
  const args = process.argv.slice(2);
  requireCondition(args.length === 4, 'Expected --browser and --expected-research-commit.');
  requireCondition(args[0] === '--browser', 'First argument must be --browser.');
  requireCondition(
    args[2] === '--expected-research-commit',
    'Third argument must be --expected-research-commit.',
  );
  requireCondition(GIT_COMMIT_PATTERN.test(args[3]), 'Expected research commit must be a full Git object ID.');
  return { browserPath: path.resolve(args[1]), expectedResearchCommit: args[3] };
}

async function main() {
  const { browserPath, expectedResearchCommit } = parseArguments();
  const protocolBytes = await readFile(protocolPath);
  const protocol = JSON.parse(protocolBytes.toString('utf8'));
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const patchPath = path.join(repoRoot, ...protocol.patch.path.split('/'));
  const runRoot = path.join(repoRoot, ...protocol.output.directory.split('/'));
  const preCollectionCheckpointPath = path.join(
    runRoot,
    protocol.output.preCollectionCheckpoint,
  );
  const checkpointPath = path.join(runRoot, protocol.output.checkpoint);
  const reportPath = path.join(runRoot, protocol.output.report);
  const screenshotPath = path.join(runRoot, protocol.output.screenshot);

  requireCondition(await exists(runRoot) === false, 'The single-use result directory already exists.');
  requireCondition(await exists(buildRoot) === false, 'The single-use live build directory already exists.');
  const priorAttempt = await validatePriorAttempt(protocol.priorAttempt);
  requireCondition(
    await exists(path.join(candidateRoot, 'node_modules', 'rollup', 'dist', 'es', 'rollup.js')),
    'Candidate dependencies are not installed.',
  );
  const researchBefore = await gitSnapshot(repoRoot);
  requireCondition(
    researchBefore.commit === expectedResearchCommit,
    'Research repository HEAD differs from the approved commit.',
  );
  requireCondition(researchBefore.status === '', 'Research repository is not clean.');

  const candidate = await gitSnapshot(candidateRoot);
  requireCondition(candidate.commit === protocol.upstream.commit, 'Candidate base commit mismatch.');
  requireCondition(candidate.tree === protocol.upstream.tree, 'Candidate base tree mismatch.');
  requireCondition(
    candidate.status.split('\n').every((line) => line === '' || line.startsWith('?? ') === false),
    'Candidate contains ordinary untracked files.',
  );
  const patchBytes = await readFile(patchPath);
  requireCondition(patchBytes.length === protocol.patch.bytes, 'Patch byte count mismatch.');
  requireCondition(sha256(patchBytes) === protocol.patch.sha256, 'Patch hash mismatch.');
  requireCondition(manifest.patch.bytes === protocol.patch.bytes, 'Manifest patch size mismatch.');
  requireCondition(manifest.patch.sha256 === protocol.patch.sha256, 'Manifest patch hash mismatch.');
  requireCondition(manifest.patch.files === EXPECTED_PATCH_FILES, 'Manifest patch file count mismatch.');
  const liveDiff = await candidatePatchBytes();
  requireCondition(Buffer.compare(liveDiff, patchBytes) === 0, 'Candidate diff differs from patch.');

  const browserBefore = await readFile(browserPath);
  const browserFile = await stat(browserPath);
  requireCondition(
    browserPath === path.resolve(protocol.browser.path),
    'Browser path differs from the protocol.',
  );
  requireCondition(browserFile.size === protocol.browser.bytes, 'Browser byte count mismatch.');
  requireCondition(sha256(browserBefore) === protocol.browser.sha256, 'Browser hash mismatch.');

  await mkdir(path.dirname(runRoot), { recursive: true });
  await mkdir(runRoot);

  let browser = null;
  let browserCloseRequested = false;
  let context = null;
  let page = null;
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
    kind: 'three-webgpu-draw-index-v3-owned-device-report',
    status: 'running',
    claimBoundary: protocol.claimBoundary,
    protocol: {
      path: path.relative(repoRoot, protocolPath).replaceAll('\\', '/'),
      bytes: protocolBytes.length,
      sha256: sha256(protocolBytes),
    },
    priorAttempt: { before: priorAttempt, after: null },
    research: { before: researchBefore, expectedCommit: expectedResearchCommit },
    source: {
      candidate,
      patch: { bytes: patchBytes.length, sha256: sha256(patchBytes) },
    },
    build: null,
    browser: null,
    example: null,
    observations,
    evidence: { preCollection: null, initial: null, late: null },
    preCollectionCheckpoint: null,
    checkpoint: null,
    initialValidation: null,
    validation: null,
    screenshot: null,
    cleanup: {
      browser: { attempted: false, closed: false, error: null },
      context: { attempted: false, closed: false, error: null },
      server: { attempted: false, closed: false, error: null },
    },
    failureForensics: { state: null, error: null },
    failure: null,
  };

  try {
    const built = await buildCandidate();
    const buildFiles = built.inventory;
    const builtModules = built.modules;
    requireCondition(
      JSON.stringify(buildFiles) === JSON.stringify(protocol.build.files),
      'Candidate build differs from the frozen module inventory.',
    );
    report.build = { files: buildFiles, servedFromMemory: true };
    const builtWebGPU = builtModules['three.webgpu.js'].toString('utf8');
    requireCondition(
      builtWebGPU.includes('requires immediate_address_space;'),
      'Built WebGPU bundle lacks immediate WGSL.',
    );

    server = await createServer({
      appType: 'custom',
      configFile: false,
      cacheDir: path.join(buildRoot, '.vite-cache'),
      root: repoRoot,
      logLevel: 'error',
      optimizeDeps: { noDiscovery: true },
      server: {
        host: '127.0.0.1',
        port: 0,
        strictPort: false,
        hmr: false,
        headers: isolationHeaders,
        fs: { strict: true, allow: [repoRoot] },
      },
    });
    await server.listen();
    const address = server.httpServer.address();
    requireCondition(address !== null && typeof address !== 'string', 'Vite port unavailable.');
    const baseUrl = `http://127.0.0.1:${address.port}`;

    browser = await chromium.launch({
      executablePath: browserPath,
      headless: true,
      args: [],
    });
    browser.on('disconnected', () => {
      if (browserCloseRequested === false) observations.browserDisconnects.push('unexpected');
    });
    const effectiveArguments = await captureBrowserArguments(browser);
    const forbiddenArguments = findForbiddenArguments(
      effectiveArguments.arguments,
      protocol.browser.forbiddenArgumentPrefixes,
    );
    requireCondition(browser.version() === protocol.browser.productVersion, 'Browser version mismatch.');
    requireCondition(effectiveArguments.available, 'Effective browser arguments unavailable.');
    requireCondition(forbiddenArguments.length === 0, 'Forbidden browser GPU arguments observed.');
    report.browser = {
      executable: browserPath,
      productVersion: browser.version(),
      effectiveArguments,
      forbiddenArguments,
      file: { bytes: browserFile.size, sha256: sha256(browserBefore) },
    };

    context = await browser.newContext({
      viewport: {
        width: protocol.example.viewport[0],
        height: protocol.example.viewport[1],
      },
      deviceScaleFactor: protocol.example.deviceScaleFactor,
      serviceWorkers: 'block',
    });
    const exampleUrlPath = `/.generated/three-draw-index-upstream-candidate/${exampleRelativePath}`;
    const exampleCssUrlPath = '/.generated/three-draw-index-upstream-candidate/examples/example.css';
    const buildUrlRoot = '/.generated/three-draw-index-v3-live-build';
    const originalExample = await readFile(examplePath, 'utf8');
    const originalCss = await readFile(
      path.join(candidateRoot, 'examples', 'example.css'),
      'utf8',
    );
    const exampleHash = sha256(Buffer.from(originalExample, 'utf8'));
    const instrumentedExample = prepareExampleSource(originalExample, buildUrlRoot);

    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== baseUrl) {
        await route.abort('blockedbyclient');
        return;
      }
      if (url.pathname === exampleUrlPath) {
        await route.fulfill({
          body: instrumentedExample,
          contentType: 'text/html; charset=utf-8',
          headers: isolationHeaders,
          status: 200,
        });
        return;
      }
      if (url.pathname.startsWith(`${buildUrlRoot}/`)) {
        const filename = url.pathname.slice(buildUrlRoot.length + 1);
        requireCondition(
          Object.hasOwn(buildFiles, filename),
          `Unexpected built module request: ${filename}`,
        );
        await route.fulfill({
          body: builtModules[filename],
          contentType: 'text/javascript; charset=utf-8',
          headers: isolationHeaders,
          status: 200,
        });
        return;
      }
      if (url.pathname === exampleCssUrlPath) {
        await route.fulfill({
          body: originalCss,
          contentType: 'text/css; charset=utf-8',
          headers: isolationHeaders,
          status: 200,
        });
        return;
      }
      await route.abort('blockedbyclient');
    });

    page = await context.newPage();
    page.on('console', (message) => {
      if (message.type() === 'error') observations.consoleErrors.push(message.text());
      if (message.type() === 'warning') observations.consoleWarnings.push(message.text());
    });
    page.on('crash', () => observations.pageCrashes.push('crash'));
    page.on('pageerror', (error) => observations.pageErrors.push(serializeError(error)));
    page.on('requestfailed', (request) => observations.requestFailures.push({
      url: request.url(),
      errorText: request.failure()?.errorText ?? null,
    }));
    page.on('response', (response) => {
      if (response.status() >= 400) {
        observations.httpErrors.push({ url: response.url(), status: response.status() });
      }
    });
    await page.addInitScript(installPageInstrumentation);
    await page.goto(`${baseUrl}${exampleUrlPath}`, {
      waitUntil: 'domcontentloaded',
      timeout: 60_000,
    });
    await page.waitForFunction(
      () => window.__drawIndexV3State?.ready !== null,
      { timeout: 60_000 },
    );
    const dom = await page.evaluate(() => ({
      description: document.getElementById('description')?.textContent?.trim() ?? null,
      order: document.getElementById('order')?.textContent?.trim() ?? null,
      canvas: {
        width: document.querySelector('canvas')?.width ?? null,
        height: document.querySelector('canvas')?.height ?? null,
      },
    }));
    report.example = {
      css: {
        bytes: Buffer.byteLength(originalCss),
        sha256: sha256(Buffer.from(originalCss, 'utf8')),
      },
      dom,
      source: { bytes: Buffer.byteLength(originalExample), sha256: exampleHash },
    };
    report.observations = observations;
    const preCollectionState = await withDeadline(
      page.evaluate(() => window.__drawIndexV3LateState()),
      SNAPSHOT_TIMEOUT_MS,
      'Pre-collection synchronous state capture',
    );
    report.evidence.preCollection = preCollectionState;
    const preCollectionCheckpointBytes = canonicalJson(createPreCollectionCheckpoint(report));
    await writeFile(
      preCollectionCheckpointPath,
      preCollectionCheckpointBytes,
      { flag: 'wx' },
    );
    await chmod(preCollectionCheckpointPath, 0o444);
    report.preCollectionCheckpoint = {
      path: path.relative(repoRoot, preCollectionCheckpointPath).replaceAll('\\', '/'),
      bytes: preCollectionCheckpointBytes.length,
      sha256: sha256(preCollectionCheckpointBytes),
      status: 'pre-collection-not-final',
    };

    const initialEvidence = await withDeadline(
      page.evaluate(() => window.__drawIndexV3CollectInitialEvidence()),
      INITIAL_COLLECTION_TIMEOUT_MS,
      'Initial evidence collection',
    );
    report.evidence.initial = summarizeEvidence(initialEvidence);
    const initialValidation = validateEvidence(initialEvidence, protocol, observations);
    report.initialValidation = initialValidation;
    const checkpointBytes = canonicalJson(createInitialCheckpoint(report, initialValidation));
    await writeFile(checkpointPath, checkpointBytes, { flag: 'wx' });
    await chmod(checkpointPath, 0o444);
    report.checkpoint = {
      path: path.relative(repoRoot, checkpointPath).replaceAll('\\', '/'),
      bytes: checkpointBytes.length,
      sha256: sha256(checkpointBytes),
      status: 'checkpointed-not-final',
    };
    requireCondition(initialValidation.verified, initialValidation.reasons.join(' '));

    const screenshotBytes = await withDeadline(
      page.screenshot({ fullPage: true }),
      SNAPSHOT_TIMEOUT_MS,
      'Screenshot capture',
    );
    requireCondition(screenshotBytes.length > 10_000, 'Screenshot is unexpectedly small.');
    const screenshotDimensions = pngDimensions(screenshotBytes);
    requireCondition(
      screenshotDimensions.width === protocol.example.viewport[0]
        && screenshotDimensions.height === protocol.example.viewport[1],
      'Screenshot dimensions differ from the protocol viewport.',
    );
    await writeFile(screenshotPath, screenshotBytes, { flag: 'wx' });
    report.screenshot = {
      path: path.relative(repoRoot, screenshotPath).replaceAll('\\', '/'),
      bytes: screenshotBytes.length,
      ...screenshotDimensions,
      sha256: sha256(screenshotBytes),
    };

    const browserAfter = await readFile(browserPath);
    requireCondition(sha256(browserAfter) === protocol.browser.sha256, 'Browser changed during run.');
    const candidateAfter = await gitSnapshot(candidateRoot);
    const liveDiffAfter = await candidatePatchBytes();
    requireCondition(
      JSON.stringify(candidateAfter) === JSON.stringify(candidate),
      'Candidate repository changed during run.',
    );
    requireCondition(
      Buffer.compare(liveDiffAfter, patchBytes) === 0,
      'Candidate diff changed during run.',
    );
    report.source.after = candidateAfter;

    const buildFilesAfter = {};
    for (const filename of EXPECTED_BUILD_FILES) {
      const bytes = await readFile(path.join(buildRoot, filename));
      buildFilesAfter[filename] = { bytes: bytes.length, sha256: sha256(bytes) };
    }
    requireCondition(
      JSON.stringify(buildFilesAfter) === JSON.stringify(buildFiles),
      'Built module files changed after capture.',
    );
    report.build.postflightFiles = buildFilesAfter;

    const preCollectionCheckpointAfter = await readFile(preCollectionCheckpointPath);
    requireCondition(
      preCollectionCheckpointAfter.length === report.preCollectionCheckpoint.bytes
        && sha256(preCollectionCheckpointAfter) === report.preCollectionCheckpoint.sha256,
      'Pre-collection checkpoint changed after capture.',
    );
    const checkpointAfter = await readFile(checkpointPath);
    requireCondition(
      checkpointAfter.length === report.checkpoint.bytes
        && sha256(checkpointAfter) === report.checkpoint.sha256,
      'Initial checkpoint changed after capture.',
    );
    const priorAttemptAfter = await validatePriorAttempt(protocol.priorAttempt);
    requireCondition(
      JSON.stringify(priorAttemptAfter) === JSON.stringify(priorAttempt),
      'Prior v2 artifacts changed during the v3 run.',
    );
    report.priorAttempt.after = priorAttemptAfter;

    const researchAfter = await gitSnapshot(repoRoot);
    requireCondition(
      JSON.stringify(researchAfter) === JSON.stringify(researchBefore),
      'Research repository changed during run.',
    );
    report.research.after = researchAfter;

    await page.waitForTimeout(100);
    const lateState = await withDeadline(
      page.evaluate(() => window.__drawIndexV3LateState()),
      SNAPSHOT_TIMEOUT_MS,
      'Late synchronous state capture',
    );
    report.evidence.late = lateState;
    const evidenceValidation = validateEvidence(initialEvidence, protocol, observations);
    const lateValidation = validateLateState(initialEvidence, lateState);
    const validation = {
      events: evidenceValidation.events,
      lateState: lateValidation,
      reasons: [...evidenceValidation.reasons, ...lateValidation.reasons],
      verified: evidenceValidation.verified && lateValidation.verified,
    };
    report.validation = validation;
    requireCondition(validation.verified, validation.reasons.join(' '));
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed';
    report.failure = serializeError(error);
    if (page !== null && page.isClosed() === false) {
      try {
        report.failureForensics.state = await withDeadline(
          page.evaluate(() => window.__drawIndexV3LateState()),
          FAILURE_FORENSICS_TIMEOUT_MS,
          'Failure forensic synchronous state capture',
        );
      } catch (forensicError) {
        report.failureForensics.error = serializeError(forensicError);
      }
    }
  } finally {
    try {
      report.cleanup.context.attempted = context !== null;
      if (context !== null) {
        await withDeadline(context.close(), TEARDOWN_TIMEOUT_MS, 'Browser context teardown');
      }
      report.cleanup.context.closed = context !== null;
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
      }
      report.cleanup.browser.closed = browser !== null;
    } catch (error) {
      report.cleanup.browser.error = serializeError(error);
      report.status = 'failed';
      report.failure ??= serializeError(error);
    }
    try {
      report.cleanup.server.attempted = server !== null;
      if (server !== null) {
        await withDeadline(server.close(), TEARDOWN_TIMEOUT_MS, 'Vite teardown');
      }
      report.cleanup.server.closed = server !== null;
    } catch (error) {
      report.cleanup.server.error = serializeError(error);
      report.status = 'failed';
      report.failure ??= serializeError(error);
    }
  }

  if (report.status === 'passed') {
    const lateDiagnostics = [];
    for (const key of [
      'browserDisconnects',
      'consoleErrors',
      'consoleWarnings',
      'httpErrors',
      'pageCrashes',
      'pageErrors',
      'requestFailures',
    ]) {
      if ((report.observations?.[key]?.length ?? 0) !== 0) lateDiagnostics.push(key);
    }
    const cleanupComplete = Object.values(report.cleanup).every((entry) => (
      entry.attempted === true && entry.closed === true && entry.error === null
    ));
    if (lateDiagnostics.length !== 0 || cleanupComplete === false) {
      const message = lateDiagnostics.length !== 0
        ? `Late diagnostics were observed: ${lateDiagnostics.join(', ')}.`
        : 'Cleanup did not complete exactly.';
      report.status = 'failed';
      report.failure = serializeError(new Error(message));
      if (report.validation !== null) {
        report.validation.reasons.push(message);
        report.validation.verified = false;
      }
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
    console.error(JSON.stringify({ report: reportPath, status: report.status, failure: report.failure }, null, 2));
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

if (process.argv[1] !== undefined
    && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}

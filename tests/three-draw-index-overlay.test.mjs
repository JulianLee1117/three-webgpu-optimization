import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import {
  buildThreeDrawIndexOverlay,
  DEFAULT_THREE_DRAW_INDEX_SOURCE_ROOT,
  EXPECTED_THREE_DRAW_INDEX_COMMIT,
  EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES,
  EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256,
  EXPECTED_THREE_DRAW_INDEX_REVISION,
  EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT,
  EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256,
  EXPECTED_THREE_DRAW_INDEX_SRC_TREE,
  EXPECTED_THREE_DRAW_INDEX_TREE,
  EXPECTED_THREE_DRAW_INDEX_VERSION,
  inspectThreeDrawIndexSource,
  THREE_DRAW_INDEX_PATCH_PATH,
  THREE_DRAW_INDEX_RUNTIME_ALIASES,
  THREE_DRAW_INDEX_TARGET_PATHS,
} from '../scripts/build-three-draw-index-overlay.mjs';

const execFileAsync = promisify(execFile);
const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'navigator',
);

let temporaryRoot;
let overlayRoot;
let WebGPUBackend;
let WebGPUPipelineUtils;
let GPUPipelineLayoutDescriptor;
let WGSLNodeBuilder;
let GLSLNodeBuilder;
let NodeManager;
let Three;
let TSL;

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function setNavigator(value) {
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    enumerable: true,
    writable: true,
    value,
  });
}

function restoreNavigator() {
  if (originalNavigatorDescriptor === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(
      globalThis,
      'navigator',
      originalNavigatorDescriptor,
    );
  }
}

async function runGit(root, args, { encoding = 'utf8' } = {}) {
  const { stdout } = await execFileAsync(
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
    {
      encoding,
      maxBuffer: 64 * 1024 * 1024,
      windowsHide: true,
    },
  );
  return stdout;
}

async function listFiles(root, relativeDirectory = '') {
  const directory = path.join(root, ...relativeDirectory.split('/').filter(Boolean));
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
  const files = [];
  for (const entry of entries) {
    const relativePath = relativeDirectory === ''
      ? entry.name
      : `${relativeDirectory}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...await listFiles(root, relativePath));
    } else {
      assert.equal(entry.isFile(), true, `unexpected output entry ${relativePath}`);
      files.push(relativePath);
    }
  }
  return files.sort((left, right) => left.localeCompare(right, 'en'));
}

async function outputFingerprint(root) {
  const fingerprint = {};
  for (const relativePath of await listFiles(root)) {
    fingerprint[relativePath] = sha256(
      await readFile(path.join(root, ...relativePath.split('/'))),
    );
  }
  return fingerprint;
}

function sourceSnapshot(source) {
  return {
    commit: source.commit,
    tree: source.tree,
    srcTree: source.srcTree,
    trackedStatus: source.trackedStatus,
    fullStatus: source.fullStatus,
  };
}

function emptyShaderData() {
  return {
    directives: '',
    structs: '',
    uniforms: '',
    varyings: '',
    vars: '',
    codes: '',
    attributes: '',
    flow: '',
  };
}

function occurrences(source, token) {
  return source.split(token).length - 1;
}

function buildPublicDrawIndexShaders(ThreeApi, TSLApi, canaryFragmentGraph = false) {
  const material = new ThreeApi.NodeMaterial();
  material.vertexNode = TSLApi.vec4(TSLApi.float(TSLApi.drawIndex), 0, 0, 1);
  const fragmentDrawIndex = canaryFragmentGraph
    ? TSLApi.float(TSLApi.drawIndex).add(TSLApi.float(1)).div(255)
    : TSLApi.float(TSLApi.drawIndex);
  material.fragmentNode = TSLApi.vec4(0, fragmentDrawIndex, 0, 1);

  const geometry = new ThreeApi.BufferGeometry();
  geometry.setAttribute('position', new ThreeApi.Float32BufferAttribute([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ], 3));
  const mesh = new ThreeApi.Mesh(geometry, material);
  const renderer = {
    backend: {
      _supportsImmediateData: true,
      isWebGPUBackend: true,
      capabilities: { getUniformBufferLimit: () => 65_536 },
    },
    contextNode: TSLApi.context({}),
    library: new ThreeApi.StandardNodeLibrary(),
    getRenderTarget: () => null,
    getMRT: () => null,
    coordinateSystem: ThreeApi.WebGPUCoordinateSystem,
    outputColorSpace: ThreeApi.SRGBColorSpace,
    debug: { diagnostics: { keywords: false } },
    currentSamples: 1,
    depth: false,
    logarithmicDepthBuffer: false,
    reversedDepthBuffer: false,
    lighting: { enabled: false },
    shadowMap: { enabled: false },
    highPrecision: true,
    hasCompatibility: () => false,
  };
  const builder = new ThreeApi.WGSLNodeBuilder(mesh, renderer);
  builder.camera = new ThreeApi.PerspectiveCamera();
  builder.scene = new ThreeApi.Scene();
  builder.build();
  return builder;
}

function makeDevice(maxImmediateSize) {
  return {
    features: new Set(['core-features-and-limits']),
    limits: { maxImmediateSize },
    lost: new Promise(() => {}),
    onuncapturederror: null,
  };
}

async function initBackendWithDevice({ languageFeature, maxImmediateSize }) {
  setNavigator({
    userAgent: 'Node test',
    gpu: {
      wgslLanguageFeatures: {
        has(name) {
          return name === 'immediate_address_space' && languageFeature;
        },
      },
    },
  });
  const device = makeDevice(maxImmediateSize);
  const backend = new WebGPUBackend({ device });
  backend.updateSize = () => {};
  await backend.init({
    xr: { enabled: false },
    onDeviceLost() {},
    onError() {},
  });
  return backend;
}

before(async () => {
  temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'three-draw-index-overlay-test-'));
  overlayRoot = path.join(temporaryRoot, 'overlay');
  await buildThreeDrawIndexOverlay({ outputRoot: overlayRoot });

  const sourceRoot = path.join(overlayRoot, 'src');
  Three = await import(pathToFileURL(path.join(sourceRoot, 'Three.WebGPU.js')));
  TSL = Three.TSL;
  ({ default: WebGPUBackend } = await import(pathToFileURL(
    path.join(sourceRoot, 'renderers', 'webgpu', 'WebGPUBackend.js'),
  )));
  ({ default: WebGPUPipelineUtils } = await import(pathToFileURL(
    path.join(sourceRoot, 'renderers', 'webgpu', 'utils', 'WebGPUPipelineUtils.js'),
  )));
  ({ default: GPUPipelineLayoutDescriptor } = await import(pathToFileURL(
    path.join(
      sourceRoot,
      'renderers',
      'webgpu',
      'descriptors',
      'GPUPipelineLayoutDescriptor.js',
    ),
  )));
  ({ default: WGSLNodeBuilder } = await import(pathToFileURL(
    path.join(sourceRoot, 'renderers', 'webgpu', 'nodes', 'WGSLNodeBuilder.js'),
  )));
  ({ default: GLSLNodeBuilder } = await import(pathToFileURL(
    path.join(
      sourceRoot,
      'renderers',
      'webgl-fallback',
      'nodes',
      'GLSLNodeBuilder.js',
    ),
  )));
  ({ default: NodeManager } = await import(pathToFileURL(
    path.join(sourceRoot, 'renderers', 'common', 'nodes', 'NodeManager.js'),
  )));
}, { timeout: 120_000 });

after(async () => {
  restoreNavigator();
  if (temporaryRoot !== undefined) {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test('source-only overlay is exact, deterministic, and leaves its input untouched', {
  timeout: 120_000,
}, async () => {
  const sourceBefore = await inspectThreeDrawIndexSource();
  const firstFingerprint = await outputFingerprint(overlayRoot);
  const firstManifestBytes = await readFile(
    path.join(overlayRoot, 'THREE_DRAW_INDEX_OVERLAY.json'),
  );

  await buildThreeDrawIndexOverlay({ outputRoot: overlayRoot });
  const secondFingerprint = await outputFingerprint(overlayRoot);
  const sourceAfter = await inspectThreeDrawIndexSource();
  assert.deepEqual(secondFingerprint, firstFingerprint);
  assert.deepEqual(sourceSnapshot(sourceAfter), sourceSnapshot(sourceBefore));
  assert.equal(sourceAfter.commit, EXPECTED_THREE_DRAW_INDEX_COMMIT);
  assert.equal(sourceAfter.tree, EXPECTED_THREE_DRAW_INDEX_TREE);
  assert.equal(sourceAfter.srcTree, EXPECTED_THREE_DRAW_INDEX_SRC_TREE);
  assert.equal(sourceAfter.packageMetadata.version, EXPECTED_THREE_DRAW_INDEX_VERSION);
  assert.equal(sourceAfter.runtimeRevision, EXPECTED_THREE_DRAW_INDEX_REVISION);
  assert.equal(sourceAfter.trackedStatus, '');
  assert.equal(sourceAfter.fullStatus, '');

  assert.deepEqual((await readdir(overlayRoot)).sort(), [
    'src',
    'THREE_DRAW_INDEX_OVERLAY.json',
  ].sort());
  assert.equal((await listFiles(path.join(overlayRoot, 'src'))).length, 754);
  assert.equal(await readFile(
    path.join(overlayRoot, 'THREE_DRAW_INDEX_OVERLAY.json'),
    'utf8',
  ), firstManifestBytes.toString('utf8'));
  await assert.rejects(
    readFile(path.join(overlayRoot, '.git', 'HEAD')),
    /ENOENT/,
  );

  const manifest = JSON.parse(firstManifestBytes);
  assert.equal(manifest.source.payloadGitIndexSha256, EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256);
  assert.equal(manifest.patch.sha256, EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256);
  assert.equal(manifest.patch.bytes, EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES);
  assert.equal(manifest.output.fileCount, EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT);
  assert.deepEqual(manifest.patch.changedPaths, THREE_DRAW_INDEX_TARGET_PATHS);
  assert.deepEqual(manifest.contract.runtimeAliases, THREE_DRAW_INDEX_RUNTIME_ALIASES);
});

test('tracked patch round-trips byte-for-byte as the proposed upstream diff', {
  timeout: 60_000,
}, async () => {
  const roundTripRoot = path.join(temporaryRoot, 'round-trip');
  await mkdir(roundTripRoot, { recursive: true });
  for (const relativePath of THREE_DRAW_INDEX_TARGET_PATHS) {
    const destination = path.join(roundTripRoot, ...relativePath.split('/'));
    await mkdir(path.dirname(destination), { recursive: true });
    await copyFile(
      path.join(overlayRoot, ...relativePath.split('/')),
      destination,
    );
  }

  await runGit(roundTripRoot, ['init', '--quiet']);
  await runGit(roundTripRoot, [
    'apply',
    '--reverse',
    '--whitespace=error-all',
    THREE_DRAW_INDEX_PATCH_PATH,
  ]);

  const source = await inspectThreeDrawIndexSource();
  for (const relativePath of THREE_DRAW_INDEX_TARGET_PATHS) {
    const bytes = await readFile(path.join(roundTripRoot, ...relativePath.split('/')));
    assert.equal(
      sha256(bytes),
      source.manifest.targets[relativePath].beforeSha256,
      `${relativePath} did not reverse to its canonical preimage`,
    );
  }

  await runGit(roundTripRoot, ['add', '--all']);
  await runGit(roundTripRoot, [
    '-c',
    'user.name=Draw Index Test',
    '-c',
    'user.email=draw-index@example.invalid',
    'commit',
    '--quiet',
    '-m',
    'canonical preimage',
  ]);
  await runGit(roundTripRoot, [
    'apply',
    '--whitespace=error-all',
    THREE_DRAW_INDEX_PATCH_PATH,
  ]);
  const diff = await runGit(roundTripRoot, [
    'diff',
    '--full-index',
    '--no-renames',
    '--src-prefix=a/',
    '--dst-prefix=b/',
    '--',
    ...THREE_DRAW_INDEX_TARGET_PATHS,
  ], { encoding: null });
  assert.deepEqual(diff, await readFile(THREE_DRAW_INDEX_PATCH_PATH));
});

test('a wrong source commit fails before replacing an existing output', async () => {
  const wrongRoot = path.join(temporaryRoot, 'wrong-source');
  const preservedOutput = path.join(temporaryRoot, 'preserved-output');
  await mkdir(wrongRoot, { recursive: true });
  await mkdir(preservedOutput, { recursive: true });
  await writeFile(path.join(wrongRoot, 'placeholder.txt'), 'wrong source\n');
  await writeFile(path.join(preservedOutput, 'keep.txt'), 'preserve me\n');
  await runGit(wrongRoot, ['init', '--quiet']);
  await runGit(wrongRoot, ['add', '--all']);
  await runGit(wrongRoot, [
    '-c',
    'user.name=Draw Index Test',
    '-c',
    'user.email=draw-index@example.invalid',
    'commit',
    '--quiet',
    '-m',
    'wrong source',
  ]);

  await assert.rejects(
    buildThreeDrawIndexOverlay({
      sourceRoot: wrongRoot,
      outputRoot: preservedOutput,
    }),
    /source commit mismatch/,
  );
  assert.equal(
    await readFile(path.join(preservedOutput, 'keep.txt'), 'utf8'),
    'preserve me\n',
  );
  assert.deepEqual(await readdir(preservedOutput), ['keep.txt']);
});

test('an existing foreign output is never replaced or deleted', async () => {
  const foreignOutput = path.join(temporaryRoot, 'foreign-output');
  await mkdir(foreignOutput, { recursive: true });
  await writeFile(path.join(foreignOutput, 'keep.txt'), 'do not replace me\n');

  await assert.rejects(
    buildThreeDrawIndexOverlay({ outputRoot: foreignOutput }),
    /lacks its generated marker/,
  );
  assert.equal(
    await readFile(path.join(foreignOutput, 'keep.txt'), 'utf8'),
    'do not replace me\n',
  );
  assert.deepEqual(await readdir(foreignOutput), ['keep.txt']);
});

test('the exact runtime aliases bundle patched WebGPU and TSL source entrypoints', {
  timeout: 60_000,
}, async () => {
  const entryPath = path.join(temporaryRoot, 'alias-entry.mjs');
  await writeFile(
    entryPath,
    [
      "import { WebGPUBackend, WebGPURenderer, WGSLNodeBuilder } from 'three/webgpu';",
      "import { drawIndex } from 'three/tsl';",
      'export { WebGPUBackend, WebGPURenderer, WGSLNodeBuilder, drawIndex };',
      '',
    ].join('\n'),
  );
  const aliases = Object.entries(THREE_DRAW_INDEX_RUNTIME_ALIASES).map(
    ([find, replacement]) => ({
      find: new RegExp(find),
      replacement: path.join(overlayRoot, ...replacement.split('/'))
        .replaceAll('\\', '/'),
    }),
  );
  const { build: viteBuild } = await import('vite');
  const result = await viteBuild({
    configFile: false,
    logLevel: 'silent',
    resolve: { alias: aliases },
    build: {
      minify: false,
      target: 'esnext',
      write: false,
      rollupOptions: {
        input: entryPath,
        preserveEntrySignatures: 'strict',
      },
    },
  });
  assert.equal(Array.isArray(result), false);
  const chunks = result.output.filter((output) => output.type === 'chunk');
  assert.equal(chunks.length, 1);
  assert.deepEqual([...chunks[0].exports].sort(), [
    'WGSLNodeBuilder',
    'WebGPUBackend',
    'WebGPURenderer',
    'drawIndex',
  ]);
  const bundledSource = chunks[0].code;
  assert.match(bundledSource, /WebGPURenderer/);
  assert.match(bundledSource, /requires immediate_address_space;/);
  assert.match(bundledSource, /nodeDrawIndex/);
  assert.match(bundledSource, /setImmediates/);
});

test('public TSL drawIndex builds complete vertex and fragment WGSL', () => {
  const builder = buildPublicDrawIndexShaders(Three, TSL);

  assert.equal(builder.immediateSize, 4);
  assert.equal(
    occurrences(builder.vertexShader, 'requires immediate_address_space;'),
    1,
  );
  assert.equal(
    occurrences(builder.vertexShader, 'var<immediate> nodeDrawIndex : u32;'),
    1,
  );
  assert.match(
    builder.vertexShader,
    /varyings\.nodeVarying\d+ = nodeDrawIndex;/,
  );
  assert.match(
    builder.fragmentShader,
    /@location\( 0 \) @interpolate\(flat, either\) nodeVarying\d+ : u32/,
  );
  assert.match(builder.fragmentShader, /f32\( nodeVarying\d+ \)/);
  assert.doesNotMatch(builder.fragmentShader, /nodeDrawIndex/);
});

test('the canary fragment graph converts drawIndex before adding one', () => {
  const builder = buildPublicDrawIndexShaders(Three, TSL, true);
  assert.match(
    builder.fragmentShader,
    /f32\( nodeVarying\d+ \) \+ 1\.0/,
  );
  assert.equal(
    occurrences(builder.vertexShader, 'var<immediate> nodeDrawIndex : u32;'),
    1,
  );
});

test('the same pinned upstream TSL graph emits an invalid empty drawIndex', async () => {
  const UpstreamThree = await import(pathToFileURL(path.join(
    DEFAULT_THREE_DRAW_INDEX_SOURCE_ROOT,
    'src',
    'Three.WebGPU.js',
  )));
  const consoleEvents = [];
  const previousConsoleFunction = UpstreamThree.getConsoleFunction();
  UpstreamThree.setConsoleFunction((method, ...parameters) => {
    consoleEvents.push({ method, parameters });
  });
  let builder;
  try {
    builder = buildPublicDrawIndexShaders(UpstreamThree, UpstreamThree.TSL);
  } finally {
    UpstreamThree.setConsoleFunction(previousConsoleFunction);
  }

  assert.equal(builder.immediateSize, undefined);
  assert.doesNotMatch(builder.vertexShader, /immediate_address_space/);
  assert.match(builder.vertexShader + builder.fragmentShader, /f32\(\s*\)/);
  assert.equal(consoleEvents.some((event) => event.method === 'error'), true);
});

test('WGSL drawIndex is explicit, structured, unique, and stage-gated', () => {
  const renderer = { backend: { _supportsImmediateData: true } };
  const builder = new WGSLNodeBuilder(
    { material: {}, geometry: {} },
    renderer,
  );
  const unusedSource = builder._getWGSLVertexCode(emptyShaderData());
  assert.equal(builder.immediateSize, 0);
  assert.doesNotMatch(unusedSource, /requires immediate_address_space;/);
  assert.doesNotMatch(unusedSource, /var<immediate>/);

  builder.shaderStage = 'fragment';
  assert.equal(builder.getDrawIndex(), 'nodeDrawIndex');
  builder.shaderStage = 'vertex';
  assert.equal(builder.getDrawIndex(), 'nodeDrawIndex');
  assert.equal(builder.getDrawIndex(), 'nodeDrawIndex');
  assert.equal(builder.immediateSize, 4);
  const source = builder._getWGSLVertexCode(emptyShaderData());
  assert.equal(occurrences(source, 'requires immediate_address_space;'), 1);
  assert.equal(occurrences(source, 'var<immediate> nodeDrawIndex : u32;'), 1);
  assert.doesNotMatch(source, /enable immediate_address_space;/);

  const unsupported = new WGSLNodeBuilder(
    { material: {}, geometry: {} },
    { backend: { _supportsImmediateData: false } },
  );
  unsupported.shaderStage = 'vertex';
  assert.throws(
    () => unsupported.getDrawIndex(),
    /requires WebGPU immediate data support/,
  );
  assert.equal(unsupported.immediateSize, 0);

  unsupported.renderer.backend._supportsImmediateData = true;
  assert.equal(unsupported.isAvailable('drawIndex'), true);
  assert.equal(unsupported.getDrawIndex(), 'nodeDrawIndex');
  unsupported.shaderStage = 'compute';
  assert.throws(
    () => unsupported.getDrawIndex(),
    /not available in compute shaders/,
  );
});

test('render recovery is immediate-free while compute errors propagate', async () => {
  const message = 'THREE.WGSLNodeBuilder: drawIndex requires WebGPU immediate data support.';
  const consoleEvents = [];
  const previousConsoleFunction = Three.getConsoleFunction();
  Three.setConsoleFunction((method, ...parameters) => {
    consoleEvents.push({ method, parameters });
  });

  const fallbackBuilder = () => ({
    vertexShader: 'fallback vertex',
    fragmentShader: 'fallback fragment',
    computeShader: null,
    getAttributesArray: () => [],
    getBindings: () => [],
    updateNodes: [],
    updateBeforeNodes: [],
    updateAfterNodes: [],
    observer: {},
    hardwareClipping: false,
    transforms: [],
    immediateSize: 0,
    build() {},
    async buildAsync() {},
  });
  const exerciseRenderRecovery = async (useAsync) => {
    const manager = new NodeManager({}, {});
    const requestedMaterial = {};
    const builders = [];
    manager.getForRenderCacheKey = () => `unsupported-${useAsync}`;
    manager._createNodeBuilder = (renderObject, material) => {
      assert.equal(renderObject.material, requestedMaterial);
      builders.push(material);
      if (material === requestedMaterial) {
        return {
          build() {
            throw new Error(message);
          },
          async buildAsync() {
            throw new Error(message);
          },
        };
      }
      assert.equal(material.isNodeMaterial, true);
      return fallbackBuilder();
    };
    const renderObject = { material: requestedMaterial };
    const state = await manager.getForRender(renderObject, useAsync);
    assert.equal(builders.length, 2);
    assert.equal(state.immediateSize, 0);
    assert.equal(state.vertexShader, 'fallback vertex');
    assert.doesNotMatch(state.vertexShader, /immediate_address_space/);
    assert.equal(manager.get(renderObject).nodeBuilderState, state);
  };

  try {
    await exerciseRenderRecovery(false);
    await exerciseRenderRecovery(true);
  } finally {
    Three.setConsoleFunction(previousConsoleFunction);
  }

  assert.equal(consoleEvents.length, 2);
  for (const event of consoleEvents) {
    assert.equal(event.method, 'error');
    assert.match(event.parameters.join(' '), /drawIndex requires WebGPU immediate data support/);
  }

  let asyncBuild = false;
  const computeManager = new NodeManager(
    { debug: { onNodeBuilderCreated: null } },
    {
      createNodeBuilder() {
        return {
          build() {
            throw new Error(message);
          },
          async buildAsync() {
            asyncBuild = true;
            throw new Error(message);
          },
        };
      },
    },
  );
  const computeNode = { version: 0 };
  assert.throws(
    () => computeManager.getForCompute(computeNode),
    /drawIndex requires WebGPU immediate data support/,
  );
  await assert.rejects(
    computeManager.getForCompute(computeNode, true),
    /drawIndex requires WebGPU immediate data support/,
  );
  assert.equal(asyncBuild, true);
});

test('BatchedMesh keeps WebGPU instance-index transport without opting into immediates', async () => {
  const batchSource = await readFile(
    path.join(overlayRoot, 'src', 'nodes', 'accessors', 'Batch.js'),
    'utf8',
  );
  assert.match(
    batchSource,
    /builder\.isAvailable\( 'batchingInstanceIndex' \) \|\| builder\.getDrawIndex\(\) === null/,
  );
  assert.doesNotMatch(
    batchSource,
    /const batchingIdNode = builder\.getDrawIndex\(\) === null/,
  );

  const builder = new WGSLNodeBuilder(
    { material: {}, geometry: {} },
    { backend: { _supportsImmediateData: true } },
  );
  assert.equal(builder.isAvailable('batchingInstanceIndex'), true);
  assert.equal(builder.immediateSize, 0);
  const source = builder._getWGSLVertexCode(emptyShaderData());
  assert.doesNotMatch(source, /requires immediate_address_space;/);

  const webglWithoutExtension = new GLSLNodeBuilder(
    { material: {}, geometry: {} },
    { backend: { extensions: new Set() } },
  );
  assert.equal(webglWithoutExtension.isAvailable('batchingInstanceIndex'), false);
  assert.equal(webglWithoutExtension.getDrawIndex(), 'nodeUniformDrawId');

  const webglWithExtension = new GLSLNodeBuilder(
    { material: {}, geometry: {} },
    { backend: { extensions: new Set(['WEBGL_multi_draw']) } },
  );
  assert.equal(webglWithExtension.isAvailable('batchingInstanceIndex'), false);
  assert.equal(webglWithExtension.getDrawIndex(), 'uint( gl_DrawID )');
});

test('immediate size survives NodeBuilderState creation', () => {
  const nodeBuilder = {
    vertexShader: 'vertex',
    fragmentShader: 'fragment',
    computeShader: null,
    getAttributesArray: () => [],
    getBindings: () => [],
    updateNodes: [],
    updateBeforeNodes: [],
    updateAfterNodes: [],
    observer: {},
    hardwareClipping: false,
    transforms: [],
    immediateSize: 4,
  };
  const state = NodeManager.prototype._createNodeBuilderState.call({}, nodeBuilder);
  assert.equal(state.immediateSize, 4);

  nodeBuilder.immediateSize = 0;
  assert.equal(
    NodeManager.prototype._createNodeBuilderState.call({}, nodeBuilder).immediateSize,
    0,
  );
});

test('pipeline layout immediateSize resets after success and synchronous failure', () => {
  const captures = [];
  let throwNext = false;
  const backend = {
    _supportsImmediateData: true,
    device: {
      limits: { maxImmediateSize: 64 },
      createPipelineLayout(descriptor) {
        captures.push({
          bindGroupLayouts: [...descriptor.bindGroupLayouts],
          immediateSize: descriptor.immediateSize,
        });
        if (throwNext) {
          throwNext = false;
          throw new Error('synthetic layout failure');
        }
        return { id: captures.length };
      },
    },
  };
  const utils = new WebGPUPipelineUtils(backend);
  utils._createPipelineLayout(['immediate'], 4);
  utils._createPipelineLayout(['ordinary']);
  throwNext = true;
  assert.throws(
    () => utils._createPipelineLayout(['failure'], 4),
    /synthetic layout failure/,
  );
  utils._createPipelineLayout(['compute']);
  assert.deepEqual(captures.map((capture) => capture.immediateSize), [4, 0, 4, 0]);

  assert.throws(
    () => utils._createPipelineLayout([], 8),
    /Unsupported immediate data size 8/,
  );
  backend._supportsImmediateData = false;
  assert.throws(
    () => utils._createPipelineLayout([], 4),
    /not supported by the active device/,
  );
  assert.equal(captures.length, 4);

  const descriptor = new GPUPipelineLayoutDescriptor();
  assert.equal(descriptor.immediateSize, 0);
  descriptor.immediateSize = 4;
  descriptor.reset();
  assert.equal(descriptor.immediateSize, 0);
});

test('render and compute pipeline lifecycles retain exact immediate metadata', () => {
  const records = {
    layouts: [],
    renderPipelines: [],
    computePipelines: [],
    pushedScopes: 0,
  };
  const backendData = new Map();
  const device = {
    limits: { maxImmediateSize: 64 },
    createPipelineLayout(descriptor) {
      const layout = {
        bindGroupLayouts: [...descriptor.bindGroupLayouts],
        immediateSize: descriptor.immediateSize,
      };
      records.layouts.push(layout);
      return layout;
    },
    pushErrorScope(scope) {
      assert.equal(scope, 'validation');
      records.pushedScopes += 1;
    },
    popErrorScope() {
      records.pushedScopes -= 1;
      return Promise.resolve(null);
    },
    createRenderPipeline(descriptor) {
      records.renderPipelines.push({
        layoutSize: descriptor.layout.immediateSize,
        vertexModule: descriptor.vertex.module,
        fragmentModule: descriptor.fragment.module,
      });
      return { kind: 'render', index: records.renderPipelines.length - 1 };
    },
    createComputePipeline(descriptor) {
      records.computePipelines.push({
        layoutSize: descriptor.layout.immediateSize,
        computeModule: descriptor.compute.module,
      });
      return { kind: 'compute' };
    },
  };
  const backend = {
    _supportsImmediateData: true,
    device,
    attributeUtils: { createShaderVertexBuffers: () => [] },
    utils: {
      getCurrentColorFormat: () => 'rgba8unorm',
      getCurrentDepthStencilFormat: () => 'depth24plus',
    },
    get(key) {
      assert.equal(backendData.has(key), true, 'missing fake backend data');
      return backendData.get(key);
    },
  };
  const pipelineUtils = new WebGPUPipelineUtils(backend);
  pipelineUtils._getColorWriteMask = () => 0xF;
  pipelineUtils._getPrimitiveState = () => ({ topology: 'triangle-list' });
  pipelineUtils._getDepthCompare = () => 'less';
  pipelineUtils._getSampleCount = () => 1;

  const makeRenderObject = (label, immediateSize) => {
    const vertexProgram = { label: `${label}-vertex` };
    const fragmentProgram = { label: `${label}-fragment` };
    const pipeline = { vertexProgram, fragmentProgram };
    backendData.set(vertexProgram, { module: { module: `${label}-vertex-module` } });
    backendData.set(fragmentProgram, { module: { module: `${label}-fragment-module` } });
    backendData.set(pipeline, {});
    return {
      object: {},
      material: {
        alphaToCoverage: false,
        blending: 0,
        colorWrite: true,
        id: label,
        name: label,
        stencilWrite: false,
        transparent: false,
        type: 'NodeMaterial',
      },
      geometry: {},
      pipeline,
      context: {
        depth: false,
        mrt: null,
        stencil: false,
        textures: null,
      },
      getBindings: () => [],
      getNodeBuilderState: () => ({ immediateSize }),
    };
  };

  const immediateRender = makeRenderObject('immediate', 4);
  const ordinaryRender = makeRenderObject('ordinary', 0);
  pipelineUtils.createRenderPipeline(immediateRender, null);
  pipelineUtils.createRenderPipeline(ordinaryRender, null);
  assert.equal(backendData.get(immediateRender.pipeline).immediateSize, 4);
  assert.equal(backendData.get(ordinaryRender.pipeline).immediateSize, 0);

  const computeProgram = { stage: 'compute', name: '' };
  const computePipeline = { computeProgram };
  backendData.set(computeProgram, {
    module: { module: 'compute-module' },
  });
  backendData.set(computePipeline, {});
  pipelineUtils.createComputePipeline(computePipeline, [], null);

  assert.deepEqual(
    records.layouts.map((layout) => layout.immediateSize),
    [4, 0, 0],
  );
  assert.deepEqual(
    records.renderPipelines.map((pipeline) => pipeline.layoutSize),
    [4, 0],
  );
  assert.deepEqual(records.computePipelines, [{
    layoutSize: 0,
    computeModule: 'compute-module',
  }]);
  assert.equal(records.pushedScopes, 0);
});

test('capability requires both the WGSL language feature and four device bytes', async () => {
  for (const [languageFeature, maxImmediateSize, expected] of [
    [false, 64, false],
    [true, undefined, false],
    [true, Number.NaN, false],
    [true, 0, false],
    [true, 3, false],
    [true, 4, true],
    [true, 64, true],
  ]) {
    const backend = await initBackendWithDevice({ languageFeature, maxImmediateSize });
    assert.equal(
      backend._supportsImmediateData,
      expected,
      `language=${languageFeature} limit=${maxImmediateSize}`,
    );
  }
});

test('default device request does not inject or promote maxImmediateSize', async () => {
  let requestedDescriptor = null;
  const device = makeDevice(64);
  setNavigator({
    userAgent: 'Node test',
    gpu: {
      wgslLanguageFeatures: new Set(['immediate_address_space']),
      async requestAdapter() {
        return {
          features: new Set(),
          async requestDevice(descriptor) {
            requestedDescriptor = {
              requiredFeatures: [...descriptor.requiredFeatures],
              requiredLimits: { ...descriptor.requiredLimits },
            };
            return device;
          },
        };
      },
    },
  });
  const backend = new WebGPUBackend();
  backend.updateSize = () => {};
  await backend.init({
    xr: { enabled: false },
    _samples: 4,
    onDeviceLost() {},
    onError() {},
  });
  assert.deepEqual(requestedDescriptor, {
    requiredFeatures: [],
    requiredLimits: {},
  });
  assert.equal(
    Object.hasOwn(requestedDescriptor.requiredLimits, 'maxImmediateSize'),
    false,
  );
  assert.equal(backend._supportsImmediateData, true);
});

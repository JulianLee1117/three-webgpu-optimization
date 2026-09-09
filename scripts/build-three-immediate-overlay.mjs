import { createHash } from 'node:crypto';
import {
  cp,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const EXPECTED_THREE_VERSION = '0.185.1';
export const IMMEDIATE_DRAW_BASE_SYMBOL = 'threeImmediateDrawBase';

export const EXPECTED_TARGET_HASHES = Object.freeze({
  'package.json': '1d5c438acdc1fe5a52fb8ede2e417afd91fec91d22275c40f7b739591b24e4df',
  'src/core/BufferGeometry.js': 'be1b8ad6ce3e502904fd989f5b6690d8fedcbadb74508aec99fc87dfd567e8c7',
  'src/renderers/common/Geometries.js': '38c8195bc93057e220759e6f9ff577be4f1dbaea63af118c1c5d9f02d87881f2',
  'src/renderers/common/RenderObject.js': '8a4fc82b26bb416270e32c2e436238fe0390f53a4c981a3ec61224529001953e',
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js': '1bef5f23edb17cafc2808db0bf9c51f24b162353c3dc8bc14d98a761f9b75b8c',
  'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js': 'be01afed4209eab3538645cea84bc74c01d1c840f5c224fd9b1e287ae1cc491b',
  'src/renderers/webgpu/utils/WebGPUPipelineUtils.js': '3cf042142b37f7b71c57f8d68e3815b8a5fdada5ffb5ce47dc79dca305f06fdb',
  'src/renderers/webgpu/WebGPUBackend.js': 'ffd7f720c3cbbe4c6ec9757dd73b003fa340b1c2e40ccf4be131b6025a649e31',
  'build/three.core.js': '3718df126d69c125362a03340913204470d8c50238605150e57f808840fb7759',
  'build/three.webgpu.js': '50e4013dd3903e8afb09a4829962dbf105488de7bd47f61308f44bd2e66b3340',
});

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_SOURCE_ROOT = path.join(repoRoot, 'node_modules', 'three');
export const DEFAULT_OUTPUT_ROOT = path.join(
  repoRoot,
  '.generated',
  'three-immediate-overlay',
);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function countOccurrences(source, value) {
  if (value.length === 0) return 0;

  let count = 0;
  let offset = 0;
  while (true) {
    const index = source.indexOf(value, offset);
    if (index === -1) return count;
    count += 1;
    offset = index + value.length;
  }
}

function replaceExactlyOnce(source, { id, marker, before, after }) {
  const markerCount = countOccurrences(source, marker);

  if (markerCount === 1) {
    if (countOccurrences(source, after) !== 1) {
      throw new Error(
        `Three immediate overlay transform ${id} is only partially present`,
      );
    }

    return source;
  }

  if (markerCount !== 0) {
    throw new Error(
      `Three immediate overlay transform ${id} has ${markerCount} markers; expected 0 or 1`,
    );
  }

  const anchorCount = countOccurrences(source, before);
  if (anchorCount !== 1) {
    throw new Error(
      `Three immediate overlay transform ${id} found ${anchorCount} source anchors; expected exactly 1`,
    );
  }

  const patched = source.replace(before, after);
  if (countOccurrences(patched, marker) !== 1
      || countOccurrences(patched, after) !== 1) {
    throw new Error(`Three immediate overlay transform ${id} failed its postcondition`);
  }

  return patched;
}

const geometryStorage = {
  id: 'geometry-storage',
  marker: '// three-immediate-overlay: one u32 base per indirect command.',
  before: `\t\tthis.indirectOffset = 0;

\t\t/**
\t\t * This dictionary has as id the name of the attribute to be set and as value`,
  after: `\t\tthis.indirectOffset = 0;

\t\t/**
\t\t * Host-side immediate base values, parallel one-to-one with the indirect offsets.
\t\t *
\t\t * The WebGPU backend writes one value before each indirect draw. The value is
\t\t * visible to a vertex shader as \`threeImmediateDrawBase\`.
\t\t *
\t\t * @type {?Uint32Array}
\t\t * @default null
\t\t */
\t\t// three-immediate-overlay: one u32 base per indirect command.
\t\tthis.indirectImmediateBases = null;

\t\t/**
\t\t * This dictionary has as id the name of the attribute to be set and as value`,
};

const geometrySetter = {
  id: 'geometry-setter',
  marker: '// three-immediate-overlay: retain the validated host-side bases.',
  before: `\t * @param {BufferAttribute} indirect - The attribute holding indirect draw calls.
\t * @param {number|Array<number>} [indirectOffset=0] - The offset, in bytes, into the indirect drawing buffer where the value data begins. If an array is provided, multiple indirect draw calls will be made for each offset.
\t * @return {BufferGeometry} A reference to this instance.
\t */
\tsetIndirect( indirect, indirectOffset = 0 ) {

\t\tthis.indirect = indirect;
\t\tthis.indirectOffset = indirectOffset;

\t\treturn this;

\t}`,
  after: `\t * @param {BufferAttribute} indirect - The attribute holding indirect draw calls.
\t * @param {number|Array<number>} [indirectOffset=0] - The offset, in bytes, into the indirect drawing buffer where the value data begins. If an array is provided, multiple indirect draw calls will be made for each offset.
\t * @param {?Uint32Array} [indirectImmediateBases=null] - One immediate u32 base per indirect offset. Requires a vertex shader that references \`threeImmediateDrawBase\`.
\t * @return {BufferGeometry} A reference to this instance.
\t */
\tsetIndirect( indirect, indirectOffset = 0, indirectImmediateBases = null ) {

\t\tthis.indirect = indirect;
\t\tthis.indirectOffset = indirectOffset;
\t\t// three-immediate-overlay: retain the validated host-side bases.
\t\tthis.indirectImmediateBases = indirectImmediateBases;

\t\treturn this;

\t}`,
};

const geometryGetter = {
  id: 'geometry-getter',
  marker: '// three-immediate-overlay: public geometry-side read seam.',
  before: `\tgetIndirect() {

\t\treturn this.indirect;

\t}

\t/**
\t * Returns the buffer attribute for the given name.`,
  after: `\tgetIndirect() {

\t\treturn this.indirect;

\t}

\t/**
\t * Returns the per-command immediate bases for indirect drawing.
\t *
\t * @return {?Uint32Array} One u32 base per indirect offset, or \`null\`.
\t */
\tgetIndirectImmediateBases() {

\t\t// three-immediate-overlay: public geometry-side read seam.
\t\treturn this.indirectImmediateBases;

\t}

\t/**
\t * Returns the buffer attribute for the given name.`,
};

const geometriesGetter = {
  id: 'geometries-getter',
  marker: '// three-immediate-overlay: geometry manager forwarding seam.',
  before: `\tgetIndirectOffset( renderObject ) {

\t\treturn renderObject.geometry.indirectOffset;

\t}

\t/**
\t * Returns the index of the given render object's geometry.`,
  after: `\tgetIndirectOffset( renderObject ) {

\t\treturn renderObject.geometry.indirectOffset;

\t}

\t/**
\t * Returns the per-command immediate bases for the given render object.
\t *
\t * @param {RenderObject} renderObject - The render object.
\t * @return {?Uint32Array} One u32 base per indirect offset, or \`null\`.
\t */
\tgetIndirectImmediateBases( renderObject ) {

\t\t// three-immediate-overlay: geometry manager forwarding seam.
\t\treturn renderObject.geometry.indirectImmediateBases;

\t}

\t/**
\t * Returns the index of the given render object's geometry.`,
};

const renderObjectGetter = {
  id: 'render-object-getter',
  marker: '// three-immediate-overlay: render-object forwarding seam.',
  before: `\tgetIndirectOffset() {

\t\treturn this._geometries.getIndirectOffset( this );

\t}

\t/**
\t * Returns an array that acts as a key for identifying the render object in a chain map.`,
  after: `\tgetIndirectOffset() {

\t\treturn this._geometries.getIndirectOffset( this );

\t}

\t/**
\t * Returns the per-command immediate bases for indirect drawing.
\t *
\t * @return {?Uint32Array} One u32 base per indirect offset, or \`null\`.
\t */
\tgetIndirectImmediateBases() {

\t\t// three-immediate-overlay: render-object forwarding seam.
\t\treturn this._geometries.getIndirectImmediateBases( this );

\t}

\t/**
\t * Returns an array that acts as a key for identifying the render object in a chain map.`,
};

const wgslImmediateDeclaration = {
  id: 'wgsl-immediate-declaration',
  marker: '// three-immediate-overlay: opt in only when the private symbol is referenced.',
  before: `\t_getWGSLVertexCode( shaderData ) {

\t\treturn \`${ '${ this.getSignature() }' }
// directives
${ '${shaderData.directives}' }

// structs`,
  after: `\t_getWGSLVertexCode( shaderData ) {

\t\t// three-immediate-overlay: opt in only when the private symbol is referenced.
\t\tconst immediateDrawBase = shaderData.flow.includes( 'threeImmediateDrawBase' ) || shaderData.codes.includes( 'threeImmediateDrawBase' )
\t\t\t? 'requires immediate_address_space;\\n\\n// immediate data\\nvar<immediate> threeImmediateDrawBase : u32;\\n'
\t\t\t: '';

\t\treturn \`${ '${ this.getSignature() }' }
// directives
${ '${shaderData.directives}' }
${ '${immediateDrawBase}' }
// structs`,
};

const descriptorStorage = {
  id: 'pipeline-layout-storage',
  marker: '// three-immediate-overlay: default and reset must both be zero.',
  before: `\t\tthis.bindGroupLayouts = null;

\t}

\t/**
\t * Resets the descriptor to its default state.`,
  after: `\t\tthis.bindGroupLayouts = null;

\t\t/**
\t\t * The number of bytes available in the pipeline's immediate address space.
\t\t *
\t\t * @type {number}
\t\t * @default 0
\t\t */
\t\t// three-immediate-overlay: default and reset must both be zero.
\t\tthis.immediateSize = 0;

\t}

\t/**
\t * Resets the descriptor to its default state.`,
};

const descriptorReset = {
  id: 'pipeline-layout-reset',
  marker: '// three-immediate-overlay: prevent the singleton descriptor from leaking opt-in.',
  before: `\t\tthis.label = '';
\t\tthis.bindGroupLayouts = null;

\t}

}`,
  after: `\t\tthis.label = '';
\t\tthis.bindGroupLayouts = null;
\t\t// three-immediate-overlay: prevent the singleton descriptor from leaking opt-in.
\t\tthis.immediateSize = 0;

\t}

}`,
};

const pipelineLayoutOptIn = {
  id: 'pipeline-layout-opt-in',
  marker: '// three-immediate-overlay: derive layout state from this vertex program only.',
  before: `\t\tconst sampleCount = this._getSampleCount( renderObject.context );

\t\t_pipelineLayoutDescriptor.bindGroupLayouts = bindGroupLayouts;

\t\tconst pipelineLayout = device.createPipelineLayout( _pipelineLayoutDescriptor );`,
  after: `\t\tconst sampleCount = this._getSampleCount( renderObject.context );

\t\t_pipelineLayoutDescriptor.bindGroupLayouts = bindGroupLayouts;
\t\t// three-immediate-overlay: derive layout state from this vertex program only.
\t\t_pipelineLayoutDescriptor.immediateSize = vertexProgram.code.includes( 'var<immediate> threeImmediateDrawBase : u32;' ) ? 4 : 0;
\t\tpipelineData.immediateSize = _pipelineLayoutDescriptor.immediateSize;

\t\tconst pipelineLayout = device.createPipelineLayout( _pipelineLayoutDescriptor );`,
};

function backendImmediateTransform({ indexed }) {
  const drawMethod = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
  const marker = indexed
    ? '// three-immediate-overlay: set one indexed-indirect base immediately before its draw.'
    : '// three-immediate-overlay: set one non-indexed-indirect base immediately before its draw.';
  const prefix = `\t\t\t\tconst buffer = this.get( indirect ).buffer;
\t\t\t\tconst indirectOffset = renderObject.getIndirectOffset();
\t\t\t\tconst indirectOffsets = Array.isArray( indirectOffset ) ? indirectOffset : [ indirectOffset ];`;
  const before = `${prefix}

\t\t\t\tfor ( let i = 0; i < indirectOffsets.length; i ++ ) {

\t\t\t\t\tpassEncoderGPU.${drawMethod}( buffer, indirectOffsets[ i ] );

\t\t\t\t}`;
  const after = `${prefix}
\t\t\t\tconst indirectImmediateBases = renderObject.getIndirectImmediateBases();

\t\t\t\tif ( indirectImmediateBases !== null ) {

\t\t\t\t\tif ( indirectImmediateBases instanceof Uint32Array === false ) {

\t\t\t\t\t\tthrow new Error( 'THREE.WebGPUBackend: indirectImmediateBases must be a Uint32Array.' );

\t\t\t\t\t}

\t\t\t\t\tif ( indirectImmediateBases.length !== indirectOffsets.length ) {

\t\t\t\t\t\tthrow new Error( 'THREE.WebGPUBackend: indirectImmediateBases must contain exactly one value per indirect offset.' );

\t\t\t\t\t}

\t\t\t\t\tif ( this.get( renderObject.pipeline ).immediateSize !== 4 ) {

\t\t\t\t\t\tthrow new Error( 'THREE.WebGPUBackend: indirectImmediateBases require a vertex shader reference to threeImmediateDrawBase.' );

\t\t\t\t\t}

\t\t\t\t\tif ( typeof passEncoderGPU.setImmediates !== 'function' ) {

\t\t\t\t\t\tthrow new Error( 'THREE.WebGPUBackend: WebGPU immediate data is unavailable on this encoder.' );

\t\t\t\t\t}

\t\t\t\t}

\t\t\t\tfor ( let i = 0; i < indirectOffsets.length; i ++ ) {

\t\t\t\t\tif ( indirectImmediateBases !== null ) {

\t\t\t\t\t\t${marker}
\t\t\t\t\t\tpassEncoderGPU.setImmediates( 0, indirectImmediateBases, i, 1 );

\t\t\t\t\t}

\t\t\t\t\tpassEncoderGPU.${drawMethod}( buffer, indirectOffsets[ i ] );

\t\t\t\t}`;

  return {
    id: indexed ? 'backend-indexed-indirect' : 'backend-non-indexed-indirect',
    marker,
    before,
    after,
  };
}

const TRANSFORMS = Object.freeze({
  'src/core/BufferGeometry.js': [geometryStorage, geometrySetter, geometryGetter],
  'src/renderers/common/Geometries.js': [geometriesGetter],
  'src/renderers/common/RenderObject.js': [renderObjectGetter],
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js': [wgslImmediateDeclaration],
  'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js': [
    descriptorStorage,
    descriptorReset,
  ],
  'src/renderers/webgpu/utils/WebGPUPipelineUtils.js': [pipelineLayoutOptIn],
  'src/renderers/webgpu/WebGPUBackend.js': [
    backendImmediateTransform({ indexed: true }),
    backendImmediateTransform({ indexed: false }),
  ],
});

const coreBuildTransforms = Object.freeze([
  geometryStorage,
  geometrySetter,
  geometryGetter,
]);
const webgpuBuildTransforms = Object.freeze(
  Object.entries(TRANSFORMS)
    .filter(([relativePath]) => relativePath !== 'src/core/BufferGeometry.js')
    .flatMap(([, transforms]) => transforms),
);

export function transformThreeImmediateTarget(relativePath, source) {
  let transforms = TRANSFORMS[relativePath];
  if (relativePath === 'build/three.core.js') transforms = coreBuildTransforms;
  if (relativePath === 'build/three.webgpu.js') transforms = webgpuBuildTransforms;

  if (transforms === undefined) {
    throw new Error(`No Three immediate overlay transform is registered for ${relativePath}`);
  }

  return transforms.reduce(replaceExactlyOnce, source);
}

async function verifyPinnedSource(sourceRoot) {
  const packagePath = path.join(sourceRoot, 'package.json');
  const packageBytes = await readFile(packagePath);
  let packageMetadata;

  try {
    packageMetadata = JSON.parse(packageBytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Cannot parse Three.js package metadata at ${packagePath}`, {
      cause: error,
    });
  }

  if (packageMetadata.name !== 'three'
      || packageMetadata.version !== EXPECTED_THREE_VERSION) {
    throw new Error(
      `Three immediate overlay requires three@${EXPECTED_THREE_VERSION}; found ${packageMetadata.name ?? '<unknown>'}@${packageMetadata.version ?? '<unknown>'}`,
    );
  }

  const verifiedHashes = {};
  for (const [relativePath, expectedHash] of Object.entries(EXPECTED_TARGET_HASHES)) {
    const bytes = relativePath === 'package.json'
      ? packageBytes
      : await readFile(path.join(sourceRoot, relativePath));
    const actualHash = sha256(bytes);
    if (actualHash !== expectedHash) {
      throw new Error(
        `Three immediate overlay source hash mismatch for ${relativePath}: expected ${expectedHash}, received ${actualHash}`,
      );
    }
    verifiedHashes[relativePath] = actualHash;
  }

  return verifiedHashes;
}

function assertSafeRoots(sourceRoot, outputRoot) {
  const source = path.resolve(sourceRoot);
  const output = path.resolve(outputRoot);
  if (output === path.parse(output).root) {
    throw new Error('Three immediate overlay output cannot be a filesystem root');
  }
  if (source === output
      || output.startsWith(`${source}${path.sep}`)
      || source.startsWith(`${output}${path.sep}`)) {
    throw new Error(
      'Three immediate overlay source and output must be disjoint directories',
    );
  }
}

export async function buildThreeImmediateOverlay({
  sourceRoot = DEFAULT_SOURCE_ROOT,
  outputRoot = DEFAULT_OUTPUT_ROOT,
} = {}) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const resolvedOutputRoot = path.resolve(outputRoot);
  assertSafeRoots(resolvedSourceRoot, resolvedOutputRoot);

  const sourceHashes = await verifyPinnedSource(resolvedSourceRoot);
  const outputParent = path.dirname(resolvedOutputRoot);
  const stagingRoot = path.join(
    outputParent,
    `.three-immediate-overlay-${process.pid}-${createHash('sha256')
      .update(resolvedOutputRoot)
      .digest('hex')
      .slice(0, 12)}.tmp`,
  );

  await mkdir(outputParent, { recursive: true });
  await rm(stagingRoot, { recursive: true, force: true });

  try {
    await cp(resolvedSourceRoot, stagingRoot, {
      recursive: true,
      errorOnExist: true,
      force: false,
    });

    const patchedHashes = {};
    for (const relativePath of Object.keys(TRANSFORMS)) {
      const targetPath = path.join(stagingRoot, relativePath);
      const source = await readFile(targetPath, 'utf8');
      const patched = transformThreeImmediateTarget(relativePath, source);
      await writeFile(targetPath, patched, 'utf8');
      patchedHashes[relativePath] = sha256(Buffer.from(patched));
    }

    for (const relativePath of ['build/three.core.js', 'build/three.webgpu.js']) {
      const buildPath = path.join(stagingRoot, relativePath);
      const buildSource = await readFile(buildPath, 'utf8');
      const patchedBuild = transformThreeImmediateTarget(relativePath, buildSource);
      await writeFile(buildPath, patchedBuild, 'utf8');
      patchedHashes[relativePath] = sha256(Buffer.from(patchedBuild));
    }

    const manifest = {
      schemaVersion: 1,
      overlay: 'three-immediate-address-attribution',
      source: {
        package: 'three',
        version: EXPECTED_THREE_VERSION,
        targetSha256: sourceHashes,
      },
      contract: {
        symbol: IMMEDIATE_DRAW_BASE_SYMBOL,
        wgslRequirement: 'immediate_address_space',
        immediateSizeBytes: 4,
        geometryProperty: 'indirectImmediateBases',
        setter: 'setIndirect(indirect, indirectOffset, indirectImmediateBases)',
      },
      patchedSha256: patchedHashes,
    };
    await writeFile(
      path.join(stagingRoot, 'THREE_IMMEDIATE_OVERLAY.json'),
      `${JSON.stringify(manifest, null, 2)}\n`,
      'utf8',
    );

    await rm(resolvedOutputRoot, { recursive: true, force: true });
    await rename(stagingRoot, resolvedOutputRoot);

    return {
      outputRoot: resolvedOutputRoot,
      manifest,
    };
  } catch (error) {
    await rm(stagingRoot, { recursive: true, force: true });
    throw error;
  }
}

function isDirectExecution() {
  if (process.argv[1] === undefined) return false;
  return fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
}

if (isDirectExecution()) {
  buildThreeImmediateOverlay()
    .then(({ outputRoot }) => {
      process.stdout.write(
        `Generated three@${EXPECTED_THREE_VERSION} immediate-data overlay at ${outputRoot}\n`,
      );
    })
    .catch((error) => {
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    });
}

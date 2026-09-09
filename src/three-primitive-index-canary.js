import * as THREE from '/__three_primitive_index__/three.webgpu.js';
import {
  float,
  instanceIndex,
  primitiveIndex,
  uint,
  uvec4,
  vec4,
} from '/__three_primitive_index__/three.tsl.js';

import {
  PRIMITIVE_INDEX_CANARY_KIND,
  PRIMITIVE_INDEX_INDICES,
  PRIMITIVE_INDEX_INSTANCE_TRANSLATIONS,
  PRIMITIVE_INDEX_ORACLE_SIZE,
  PRIMITIVE_INDEX_POSITIONS,
  PRIMITIVE_INDEX_SENTINEL,
  analyzePrimitiveIndexOracle,
  expectedPrimitiveIndexTuples,
} from './three-primitive-index-canary-contract.js';

const state = {
  kind: PRIMITIVE_INDEX_CANARY_KIND,
  status: 'running',
  error: null,
  geometry: null,
  oracle: null,
  renderer: null,
  validationScope: null,
};
Object.defineProperty(window, '__threePrimitiveIndexCanary', { value: state });

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error'),
    message: String(error?.message ?? error),
    stack: error?.stack == null ? null : String(error.stack),
  };
}

async function sha256(words) {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    words.buffer.slice(words.byteOffset, words.byteOffset + words.byteLength),
  );
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function installStyles() {
  const style = document.createElement('style');
  style.textContent = `
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; overflow: hidden; background: #080d18; color: #f8fafc; }
    main { position: relative; width: 100vw; height: 100vh; isolation: isolate; }
    #canvas-host, #canvas-host canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
    #canvas-host { z-index: -1; }
    #canvas-host::after { content: ''; position: absolute; inset: 0; pointer-events: none;
      background: radial-gradient(circle at 50% 54%, transparent 5%, rgba(8, 13, 24, .34) 77%); }
    header { position: absolute; top: 30px; left: 38px; width: 590px; }
    .eyebrow { color: #76e4f7; font: 700 12px/1.2 ui-monospace, monospace; letter-spacing: .17em; }
    h1 { margin: 9px 0 5px; font-size: 34px; line-height: 1.05; letter-spacing: -.035em; }
    header p { margin: 0; max-width: 560px; color: #b9c7dc; font-size: 15px; line-height: 1.45; }
    code { color: #f3c969; font-family: ui-monospace, monospace; }
    #instance-labels { position: absolute; left: 17%; right: 17%; top: 27%; display: flex;
      justify-content: space-around; color: #e4edf8; font: 700 12px/1 ui-monospace, monospace;
      text-shadow: 0 2px 8px #000; }
    #legend { position: absolute; left: 50%; bottom: 60px; display: flex; gap: 11px;
      transform: translateX(-50%); padding: 10px 14px; border: 1px solid #32445f;
      border-radius: 12px; background: rgba(8, 13, 24, .82); backdrop-filter: blur(10px); }
    .chip { display: flex; align-items: center; gap: 6px; color: #d8e2f0; font: 700 12px/1 ui-monospace, monospace; }
    .swatch { width: 16px; height: 16px; border-radius: 4px; box-shadow: inset 0 0 0 1px #ffffff44; }
    footer { position: absolute; left: 24px; right: 24px; bottom: 18px; display: flex;
      justify-content: space-between; color: #8291a8; font: 600 11px/1.2 ui-monospace, monospace; }
    #status[data-state='passed'] { color: #62e6a7; }
    #status[data-state='failed'] { color: #fb7185; }
  `;
  document.head.appendChild(style);
}

function createIndexedGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(PRIMITIVE_INDEX_POSITIONS, 3),
  );
  geometry.setIndex(new THREE.Uint16BufferAttribute(PRIMITIVE_INDEX_INDICES, 1));
  return geometry;
}

function createInstancedMesh(geometry, material) {
  const mesh = new THREE.InstancedMesh(
    geometry,
    material,
    PRIMITIVE_INDEX_INSTANCE_TRANSLATIONS.length,
  );
  const matrix = new THREE.Matrix4();
  PRIMITIVE_INDEX_INSTANCE_TRANSLATIONS.forEach((translation, index) => {
    matrix.makeTranslation(...translation);
    mesh.setMatrixAt(index, matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}

function createOracleMaterial() {
  const material = new THREE.NodeMaterial();
  material.side = THREE.DoubleSide;
  material.depthTest = false;
  material.depthWrite = false;
  material.blending = THREE.NoBlending;
  material.fragmentNode = uvec4(
    primitiveIndex,
    instanceIndex,
    uint(PRIMITIVE_INDEX_SENTINEL),
    uint(1),
  );
  return material;
}

function createVisualMaterial() {
  const material = new THREE.NodeMaterial();
  material.side = THREE.DoubleSide;
  material.depthTest = false;
  material.depthWrite = false;
  material.blending = THREE.NoBlending;
  const primitive = float(primitiveIndex);
  const instance = float(instanceIndex);
  material.fragmentNode = vec4(
    primitive.add(1).div(5),
    instance.add(1).div(3),
    float(4).sub(primitive).div(5),
    1,
  );
  return material;
}

function fillLegend() {
  const legend = document.getElementById('legend');
  for (let primitive = 0; primitive < 4; primitive += 1) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = `rgb(${(primitive + 1) * 51} 170 ${204 - primitive * 51})`;
    chip.append(swatch, `primitive ${primitive}`);
    legend.append(chip);
  }
}

async function run() {
  installStyles();
  fillLegend();
  const status = document.getElementById('status');
  const renderer = new THREE.WebGPURenderer({ antialias: false });
  renderer.setPixelRatio(1);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.getElementById('canvas-host').appendChild(renderer.domElement);

  await renderer.init();
  const device = renderer.backend.device;
  state.renderer = {
    backendIsWebGPU: renderer.backend.isWebGPUBackend === true,
    devicePrimitiveIndex: device.features.has('primitive-index'),
  };

  const geometry = createIndexedGeometry();
  state.geometry = {
    attributes: Object.keys(geometry.attributes).sort(),
    indexArrayType: geometry.index.array.constructor.name,
    indexCount: geometry.index.count,
    indexValues: Array.from(geometry.index.array),
    instanceCount: PRIMITIVE_INDEX_INSTANCE_TRANSLATIONS.length,
    positionCount: geometry.attributes.position.count,
  };

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-3.2, 3.2, 1.8, -1.8, 0.1, 10);
  camera.position.z = 2;
  const oracleMaterial = createOracleMaterial();
  const visualMaterial = createVisualMaterial();
  const mesh = createInstancedMesh(geometry, oracleMaterial);
  scene.add(mesh);

  const oracleTarget = new THREE.RenderTarget(
    PRIMITIVE_INDEX_ORACLE_SIZE.width,
    PRIMITIVE_INDEX_ORACLE_SIZE.height,
    {
      depthBuffer: false,
      format: THREE.RGBAIntegerFormat,
      magFilter: THREE.NearestFilter,
      minFilter: THREE.NearestFilter,
      samples: 0,
      type: THREE.UnsignedIntType,
    },
  );
  oracleTarget.texture.name = 'primitive-index-exact-u32-oracle';

  device.pushErrorScope('validation');
  let validationError = null;
  try {
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(oracleTarget);
    renderer.render(scene, camera);
    const words = await renderer.readRenderTargetPixelsAsync(
      oracleTarget,
      0,
      0,
      PRIMITIVE_INDEX_ORACLE_SIZE.width,
      PRIMITIVE_INDEX_ORACLE_SIZE.height,
    );
    const analysis = analyzePrimitiveIndexOracle(words);
    state.oracle = {
      ...analysis,
      byteLength: words.byteLength,
      expectedTuples: expectedPrimitiveIndexTuples(),
      readbackSha256: await sha256(words),
      typedArray: words.constructor.name,
    };

    mesh.material = visualMaterial;
    renderer.setRenderTarget(null);
    renderer.setClearColor(0x080d18, 1);
    renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
  } finally {
    validationError = await device.popErrorScope();
    state.validationScope = validationError === null ? null : {
      name: validationError.constructor.name,
      message: validationError.message,
    };
  }

  if (validationError !== null) throw validationError;
  if (state.oracle.exact !== true) {
    throw new Error('The rgba32uint primitiveIndex oracle did not match all eight exact tuples.');
  }

  state.status = 'passed';
  status.dataset.state = 'passed';
  status.textContent = 'PASS: exact u32 tuples 0-3 repeated across instances 0 and 1';
}

run().catch((error) => {
  state.status = 'failed';
  state.error = serializeError(error);
  const status = document.getElementById('status');
  status.dataset.state = 'failed';
  status.textContent = `FAIL: ${state.error.message}`;
  console.error(error);
});

import {
  Color,
  DepthTexture,
  DirectionalLight,
  FloatType,
  HemisphereLight,
  Mesh,
  MeshBasicNodeMaterial,
  RenderTarget,
  Scene,
} from 'three/webgpu';
import {
  Fn,
  attribute,
  float,
  instanceIndex,
  positionGeometry,
  storage,
  uint,
  vec4,
} from 'three/tsl';
import { VIEWPORT } from '../config.js';

function freezeStaticTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

function exactBytes(value) {
  if (!ArrayBuffer.isView(value)) {
    throw new TypeError('Render-target readback must be a typed array.');
  }
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function sha256Bytes(value) {
  const bytes = exactBytes(value);
  return toHex(await crypto.subtle.digest('SHA-256', bytes));
}

async function readbackRecord(value, format, expectedByteLength) {
  if (value.byteLength !== expectedByteLength) {
    throw new Error(
      `${format} readback has ${value.byteLength} bytes; expected ${expectedByteLength}.`,
    );
  }
  return {
    format,
    arrayType: value.constructor.name,
    byteLength: value.byteLength,
    sha256: await sha256Bytes(value),
  };
}

function materialRecord(material) {
  return {
    type: material.type,
    color: material.color?.getHex() ?? null,
    roughness: material.roughness ?? null,
    metalness: material.metalness ?? null,
    opacity: material.opacity,
    transparent: material.transparent,
    blending: material.blending,
    side: material.side,
    depthTest: material.depthTest,
    depthWrite: material.depthWrite,
    alphaTest: material.alphaTest,
    toneMapped: material.toneMapped,
    hasStoragePositionNode: material.positionNode !== null,
  };
}

function createParityScene(geometry, material, { background, lit }) {
  const parityScene = new Scene();
  parityScene.background = new Color(background);
  if (lit) {
    parityScene.add(new HemisphereLight(0x9bc5ff, 0x182038, 2.2));
    const keyLight = new DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(20, 35, 60);
    parityScene.add(keyLight);
  }
  const mesh = new Mesh(geometry, material);
  freezeStaticTransform(mesh);
  mesh.frustumCulled = false;
  parityScene.add(mesh);
  return parityScene;
}

function validateObjectIdReadback(value, objectCount, expectedIds) {
  const bytes = exactBytes(value);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error('Object-ID readback must contain whole RGBA8 pixels.');
  }
  if (!(expectedIds instanceof Uint32Array)) {
    throw new TypeError('Render parity expectedIds must be a Uint32Array.');
  }
  const expected = new Uint8Array(objectCount);
  for (const objectId of expectedIds) {
    if (objectId >= objectCount) {
      throw new RangeError(`Expected object ID ${objectId} is out of range.`);
    }
    expected[objectId] = 1;
  }

  let backgroundPixels = 0;
  let coveredPixels = 0;
  let outOfRangePixels = 0;
  let nonVisiblePixels = 0;
  for (let offset = 0; offset < bytes.byteLength; offset += 4) {
    const encoded = bytes[offset]
      | (bytes[offset + 1] << 8)
      | (bytes[offset + 2] << 16);
    if (encoded === 0) {
      backgroundPixels += 1;
      continue;
    }
    coveredPixels += 1;
    const objectId = encoded - 1;
    if (objectId >= objectCount) outOfRangePixels += 1;
    else if (expected[objectId] !== 1) nonVisiblePixels += 1;
  }
  return {
    pass: coveredPixels > 0 && outOfRangePixels === 0 && nonVisiblePixels === 0,
    encoding: 'rgb24-object-id-plus-one-zero-background',
    backgroundPixels,
    coveredPixels,
    outOfRangePixels,
    nonVisiblePixels,
  };
}

function createObjectIdMaterial({ matrixAttribute, visibleIdsAttribute, objectCount }) {
  const matrixRead = storage(matrixAttribute, 'mat4', objectCount).toReadOnly();
  const visibleRead = storage(visibleIdsAttribute, 'uint', objectCount).toReadOnly();
  const sliceIndex = attribute('bucketBase', 'uint').add(instanceIndex);
  const objectId = visibleRead.element(sliceIndex);
  const fragmentObjectId = objectId.toVarying('v_renderParityObjectId');

  const material = new MeshBasicNodeMaterial();
  material.toneMapped = false;
  material.positionNode = Fn(() => {
    const objectMatrix = matrixRead.element(objectId).toVar('renderParityObjectMatrix');
    return objectMatrix.mul(vec4(positionGeometry, 1)).xyz;
  })();
  material.colorNode = Fn(() => {
    // Add one so object zero remains distinguishable from the clear background.
    const encoded = fragmentObjectId.add(uint(1)).toVar('encodedRenderParityObjectId');
    return vec4(
      float(encoded.bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
      1,
    );
  })();
  return material;
}

function requireParityStrategy(strategy) {
  const resources = strategy?.parityResources;
  if (!resources
    || !resources.matrixAttribute
    || !resources.visibleIdsAttribute
    || !Number.isInteger(resources.objectCount)
    || resources.objectCount <= 0
    || resources.objectCount > 0x00ff_ffff) {
    throw new Error(`Strategy ${strategy?.id ?? 'unknown'} does not expose parity resources.`);
  }
  if (strategy.geometries?.length !== 1 || strategy.materials?.length !== 1) {
    throw new Error('Render parity requires exactly one merged geometry and material.');
  }
  return resources;
}

async function captureOnce({
  renderer,
  camera,
  colorTarget,
  objectIdTarget,
  colorScene,
  objectIdScene,
}) {
  const previousTarget = renderer.getRenderTarget();
  const previousCubeFace = renderer.getActiveCubeFace();
  const previousMipmapLevel = renderer.getActiveMipmapLevel();
  try {
    renderer.setRenderTarget(colorTarget);
    renderer.render(colorScene, camera);
    renderer.setRenderTarget(objectIdTarget);
    renderer.render(objectIdScene, camera);
  } finally {
    renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
  }

  const [color, depth, objectId] = await Promise.all([
    renderer.readRenderTargetPixelsAsync(
      colorTarget,
      0,
      0,
      VIEWPORT.width,
      VIEWPORT.height,
    ),
    renderer.backend.copyTextureToBuffer(
      colorTarget.depthTexture,
      0,
      0,
      VIEWPORT.width,
      VIEWPORT.height,
      0,
    ),
    renderer.readRenderTargetPixelsAsync(
      objectIdTarget,
      0,
      0,
      VIEWPORT.width,
      VIEWPORT.height,
    ),
  ]);

  const [colorRecord, depthRecord, objectIdRecord] = await Promise.all([
    readbackRecord(color, 'rgba8unorm', VIEWPORT.width * VIEWPORT.height * 4),
    readbackRecord(depth, 'depth32float', VIEWPORT.width * VIEWPORT.height * 4),
    readbackRecord(
      objectId,
      'rgba8unorm-object-id-plus-one',
      VIEWPORT.width * VIEWPORT.height * 4,
    ),
  ]);
  return {
    records: { color: colorRecord, depth: depthRecord, objectId: objectIdRecord },
    objectIdPixels: objectId,
  };
}

export async function captureExactRenderParity({
  renderer,
  camera,
  strategy,
  expectedIds,
}) {
  if (typeof renderer?.backend?.copyTextureToBuffer !== 'function') {
    throw new Error('Pinned WebGPU depth readback is unavailable.');
  }
  if ((VIEWPORT.width * 4) % 256 !== 0) {
    throw new Error('Exact parity requires an unpadded 256-byte-aligned readback row.');
  }
  const resources = requireParityStrategy(strategy);
  const geometry = strategy.geometries[0];
  const sourceMaterial = strategy.materials[0];
  const objectIdMaterial = createObjectIdMaterial(resources);
  const colorScene = createParityScene(geometry, sourceMaterial, {
    background: 0x030711,
    lit: true,
  });
  const objectIdScene = createParityScene(geometry, objectIdMaterial, {
    background: 0x000000,
    lit: false,
  });
  const colorTarget = new RenderTarget(VIEWPORT.width, VIEWPORT.height, {
    depthBuffer: true,
    depthTexture: new DepthTexture(VIEWPORT.width, VIEWPORT.height, FloatType),
    samples: 0,
  });
  const objectIdTarget = new RenderTarget(VIEWPORT.width, VIEWPORT.height, {
    depthBuffer: true,
    samples: 0,
  });

  try {
    const firstCapture = await captureOnce({
      renderer,
      camera,
      colorTarget,
      objectIdTarget,
      colorScene,
      objectIdScene,
    });
    const secondCapture = await captureOnce({
      renderer,
      camera,
      colorTarget,
      objectIdTarget,
      colorScene,
      objectIdScene,
    });
    const first = firstCapture.records;
    const second = secondCapture.records;
    const stable = ['color', 'depth', 'objectId'].every(
      (name) => first[name].sha256 === second[name].sha256
        && first[name].byteLength === second[name].byteLength
        && first[name].arrayType === second[name].arrayType,
    );
    const objectIdValidation = validateObjectIdReadback(
      secondCapture.objectIdPixels,
      resources.objectCount,
      expectedIds,
    );
    return {
      schemaVersion: 1,
      kind: 'fixed-camera-offscreen-exact-render-parity',
      pass: stable && objectIdValidation.pass,
      width: VIEWPORT.width,
      height: VIEWPORT.height,
      captures: 2,
      material: materialRecord(sourceMaterial),
      color: second.color,
      depth: second.depth,
      objectId: second.objectId,
      objectIdValidation,
      reversedDepthBuffer: renderer.reversedDepthBuffer === true,
      stability: {
        pass: stable,
        firstCapture: first,
        first: {
          colorSha256: first.color.sha256,
          depthSha256: first.depth.sha256,
          objectIdSha256: first.objectId.sha256,
        },
      },
    };
  } finally {
    colorTarget.dispose();
    objectIdTarget.dispose();
    objectIdMaterial.dispose();
  }
}

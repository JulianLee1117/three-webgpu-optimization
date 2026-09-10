import * as THREE from 'three/webgpu';
import {
  Fn, float, vec2, vec3, vec4, ivec2, uint, uniform, storage,
  instanceIndex, positionLocal, uv, textureLoad, floor, min, dot, sqrt, exp, varying,
} from 'three/tsl';

const IOR = 1.49;
const HALF_EXTENT = 2.1;
const SPLAT_WIDTH = 0.22;
const PROBE_RESOLUTION = 8;

/**
 * Forward-only optical renderer. No target image or target labels enter this
 * module: the photon texture depends only on glass heights and receiver distance.
 * Initialize the supplied WebGPURenderer before construction. Render this pass
 * before the receiver that samples `texture`.
 */
export function createPhotonOptics(renderer, { grid = 25, rayResolution = 128, mapResolution = 384 } = {}) {
  const requireWebGPU = () => {
    if (renderer?.isWebGPURenderer !== true || renderer.backend?.isWebGPUBackend !== true || !renderer.backend.device) {
      throw new Error('Photon optics requires an initialized WebGPU backend. Call await renderer.init(); WebGL fallback is unsupported.');
    }
  };
  requireWebGPU();
  if (!Number.isInteger(grid) || grid < 3 || grid > 49 ||
      !Number.isInteger(rayResolution) || rayResolution < 8 || rayResolution > 192 ||
      !Number.isInteger(mapResolution) || mapResolution < 64 || mapResolution > 768) {
    throw new RangeError('Optics bounds: grid 3–49, ray resolution 8–192, map resolution 64–768.');
  }

  const heights = new Float32Array(grid * grid);
  const heightTexture = new THREE.DataTexture(heights, grid, grid, THREE.RedFormat, THREE.FloatType);
  heightTexture.name = 'Optical surface heights';
  heightTexture.minFilter = heightTexture.magFilter = THREE.NearestFilter;
  heightTexture.generateMipmaps = false;
  heightTexture.flipY = false;
  heightTexture.colorSpace = THREE.NoColorSpace;
  heightTexture.needsUpdate = true;
  const distance = uniform(10);
  const target = new THREE.RenderTarget(mapResolution, mapResolution, {
    type: THREE.HalfFloatType, format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    depthBuffer: false, stencilBuffer: false, generateMipmaps: false, samples: 0,
  });
  target.texture.name = 'Additive refracted photon irradiance';
  target.texture.colorSpace = THREE.LinearSRGBColorSpace;

  // Exact bilinear height/derivatives and Snell expression from solver.forward.
  // One function is shared by the raster positions and compute readback probe.
  const refractedRay = Fn(([xy]) => {
    const gridPoint = xy.add(1).mul((grid - 1) / 2).toVar();
    const cell = min(floor(gridPoint), vec2(grid - 2)).toVar();
    const fraction = gridPoint.sub(cell).toVar();
    const tx = fraction.x, ty = fraction.y;
    const oneX = float(1).sub(tx), oneY = float(1).sub(ty);
    const h00 = textureLoad(heightTexture, ivec2(cell), 0).r.toVar();
    const h10 = textureLoad(heightTexture, ivec2(cell.add(vec2(1, 0))), 0).r.toVar();
    const h01 = textureLoad(heightTexture, ivec2(cell.add(vec2(0, 1))), 0).r.toVar();
    const h11 = textureLoad(heightTexture, ivec2(cell.add(1)), 0).r.toVar();
    const h = h00.mul(oneX).mul(oneY).add(h10.mul(tx).mul(oneY))
      .add(h01.mul(oneX).mul(ty)).add(h11.mul(tx).mul(ty)).toVar();
    const px = h00.negate().mul(oneY).add(h10.mul(oneY))
      .sub(h01.mul(ty)).add(h11.mul(ty)).mul((grid - 1) / 2).toVar();
    const py = h00.negate().mul(oneX).sub(h10.mul(tx))
      .add(h01.mul(oneX)).add(h11.mul(tx)).mul((grid - 1) / 2).toVar();
    const slopeSquared = px.mul(px).add(py.mul(py)).toVar();
    const q = sqrt(float(1).sub(slopeSquared.mul(IOR * IOR - 1))).toVar();
    const beta = float(IOR).sub(q).div(slopeSquared.mul(IOR).add(q)).toVar();
    const cosI = float(1).div(sqrt(slopeSquared.add(1))).toVar();
    const cosT = q.mul(cosI).toVar();
    const rs = cosI.mul(IOR).sub(cosT).div(cosI.mul(IOR).add(cosT)).toVar();
    const rp = cosI.sub(cosT.mul(IOR)).div(cosI.add(cosT.mul(IOR))).toVar();
    const exitReflection = rs.mul(rs).add(rp.mul(rp)).mul(0.5);
    const transmission = float(1).sub(exitReflection).mul(1 - ((IOR - 1) / (IOR + 1)) ** 2);
    return vec3(xy.add(vec2(px, py).mul(distance.sub(h).mul(beta))), transmission);
  });

  const regularRay = (index, resolution) => vec2(
    float(index.mod(uint(resolution))).add(0.5).mul(2 / resolution).sub(1),
    float(index.div(uint(resolution))).add(0.5).mul(2 / resolution).sub(1),
  );
  const sourceQuad = new THREE.PlaneGeometry(1, 1);
  const geometry = new THREE.InstancedBufferGeometry().copy(sourceQuad);
  sourceQuad.dispose();
  geometry.instanceCount = rayResolution * rayResolution;
  const material = new THREE.MeshBasicNodeMaterial({
    transparent: true, blending: THREE.AdditiveBlending,
    depthWrite: false, depthTest: false, toneMapped: false, fog: false,
  });
  material.name = 'Normalized Gaussian photon splat';
  const photonRay = refractedRay(regularRay(instanceIndex, rayResolution));
  material.positionNode = vec3(photonRay.xy.add(positionLocal.xy.mul(SPLAT_WIDTH)), 0);
  const centeredUV = uv().sub(0.5);
  // Integrating exp(-48*r²/R²) in receiver coordinates gives pi*R²/48.
  // Each photon carries aperture area / ray count = 4/N units of flux.
  const fluxDensity = 4 / (rayResolution * rayResolution) * 48 / (Math.PI * SPLAT_WIDTH * SPLAT_WIDTH);
  const irradiance = exp(dot(centeredUV, centeredUV).mul(-48)).mul(fluxDensity)
    .mul(varying(photonRay.z, 'vPhotonTransmission'));
  material.fragmentNode = vec4(vec3(irradiance), 1);
  const photons = new THREE.Mesh(geometry, material);
  photons.name = 'Refracted parallel photon field';
  photons.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(photons);
  const camera = new THREE.OrthographicCamera(-HALF_EXTENT, HALF_EXTENT, HALF_EXTENT, -HALF_EXTENT, 0.1, 2);
  camera.position.z = 1;
  camera.updateMatrixWorld();

  const probeCount = PROBE_RESOLUTION ** 2;
  const probeAttribute = new THREE.StorageBufferAttribute(probeCount, 2);
  const probeStorage = storage(probeAttribute, 'vec2', probeCount);
  const probeNode = Fn(() => {
    probeStorage.element(instanceIndex).assign(refractedRay(regularRay(instanceIndex, PROBE_RESOLUTION)).xy);
  })().compute(probeCount);
  // Three r186 BufferAttribute has no dispose(). Geometry ownership lets the
  // renderer release this storage allocation along with its other attributes.
  geometry.setAttribute('photonProbeStorage', probeAttribute);

  let disposed = false, hasRendered = false, probing = false, readingIrradiance = false;
  let maxSlope = 0;
  const alive = () => { if (disposed) throw new Error('Photon optics has been disposed.'); requireWebGPU(); };

  function update(lens, receiverDistance = lens?.distance) {
    alive();
    if (lens?.grid !== grid || !(lens.height instanceof Float32Array || lens.height instanceof Float64Array) ||
        lens.height.length !== heights.length || !lens.height.every(Number.isFinite) ||
        !Number.isFinite(lens.ior) || Math.abs(lens.ior - IOR) > 1e-12 ||
        !Number.isFinite(lens.extent ?? 1) || Math.abs((lens.extent ?? 1) - 1) > 1e-12 ||
        (lens.baseZ !== undefined && !Number.isFinite(lens.baseZ)) ||
        !Number.isFinite(receiverDistance) || receiverDistance <= 0 || receiverDistance > 100) {
      throw new RangeError('Expected a finite matching lens with extent 1, ior 1.49 and receiver distance in (0,100].');
    }
    const candidate = Float32Array.from(lens.height);
    let slopeSquared = 0;
    for (const h of candidate) {
      if (!Number.isFinite(h) || h >= receiverDistance) throw new Error('Optical surface crossed the receiver.');
      if (lens.baseZ !== undefined && h <= lens.baseZ) throw new Error('Optical surface crossed the flat entrance face.');
    }
    for (let y = 0; y < grid - 1; y++) for (let x = 0; x < grid - 1; x++) {
      const p = y * grid + x, h00 = candidate[p], h10 = candidate[p + 1], h01 = candidate[p + grid], h11 = candidate[p + grid + 1];
      // Bilinear squared slope is convex in each axis: its maximum lies at a
      // cell corner. Validate the uploaded float32 surface, not a CPU surrogate.
      for (const dx of [h10 - h00, h11 - h01]) for (const dy of [h01 - h00, h11 - h10]) {
        slopeSquared = Math.max(slopeSquared, (dx * dx + dy * dy) * ((grid - 1) / 2) ** 2);
      }
    }
    if ((IOR * IOR - 1) * slopeSquared >= 1) throw new Error('Optical surface produces total internal reflection.');
    let changed = false;
    for (let i = 0; i < heights.length; i++) if (heights[i] !== candidate[i]) { changed = true; break; }
    if (changed) { heights.set(candidate); heightTexture.needsUpdate = true; }
    distance.value = receiverDistance;
    maxSlope = Math.sqrt(slopeSquared);
    return { grid, ior: IOR, distance: receiverDistance, maxSlope };
  }

  function render() {
    alive();
    const oldTarget = renderer.getRenderTarget();
    const oldFace = renderer.getActiveCubeFace(), oldLevel = renderer.getActiveMipmapLevel();
    const oldClear = renderer.getClearColor(new THREE.Color()), oldAlpha = renderer.getClearAlpha();
    const oldAutoClear = renderer.autoClear;
    const oldXR = renderer.xr?.enabled;
    try {
      if (renderer.xr) renderer.xr.enabled = false;
      renderer.autoClear = false;
      renderer.setRenderTarget(target);
      renderer.setClearColor(0, 0);
      renderer.clear(true, false, false);
      renderer.render(scene, camera);
      hasRendered = true;
    } finally {
      renderer.setRenderTarget(oldTarget, oldFace, oldLevel);
      renderer.setClearColor(oldClear, oldAlpha);
      renderer.autoClear = oldAutoClear;
      if (renderer.xr) renderer.xr.enabled = oldXR;
    }
    return target.texture;
  }

  async function probe(lens, receiverDistance = lens?.distance) {
    alive();
    if (probing || readingIrradiance) throw new Error('Only one photon readback may run at a time.');
    if (lens) update(lens, receiverDistance);
    probing = true;
    try {
      // Establish geometry ownership even if probe() is the first GPU call.
      if (!hasRendered) render();
      const metadata = {
        resolution: PROBE_RESOLUTION, count: probeCount, offsetX: 0.5, offsetY: 0.5,
        grid, ior: IOR, distance: distance.value, height: heights.slice(), maxSlope, fresnel: true,
      };
      renderer.compute(probeNode);
      const hits = new Float32Array(await renderer.getArrayBufferAsync(probeAttribute));
      if (hits.length !== probeCount * 2 || !hits.every(Number.isFinite)) throw new Error('Photon probe returned invalid GPU ray positions.');
      return { ...metadata, hits };
    } finally { probing = false; }
  }

  /**
   * Read the actual most recently rendered floating-point map, without display
   * exposure, tint, normalization or a vertical flip. WebGPU copies start at the
   * top-left texel; r186 WebGPUTextureUtils.copyTextureToBuffer preserves this
   * order and may return 256-byte row padding, removed below.
   *
   * For pixel (column,row), analytical receiver coordinates are:
   * x = -halfExtent + (column+.5)*2*halfExtent/width
   * y = +halfExtent - (row+.5)*2*halfExtent/height.
   * Integrate one RGB channel times (2*halfExtent)^2/(width*height) for flux.
   * Alpha counts overlapping splats and is not an energy channel.
   */
  async function readIrradiance() {
    alive();
    if (probing || readingIrradiance) throw new Error('Only one photon readback may run at a time.');
    readingIrradiance = true;
    try {
      if (!hasRendered) render();
      const raw = await renderer.readRenderTargetPixelsAsync(target, 0, 0, mapResolution, mapResolution);
      if (!(raw instanceof Uint16Array || raw instanceof Float32Array)) throw new Error('Unexpected irradiance readback component type.');
      const rowElements = mapResolution * 4;
      const stride = raw.length === rowElements * mapResolution
        ? rowElements : Math.ceil(rowElements * raw.BYTES_PER_ELEMENT / 256) * 256 / raw.BYTES_PER_ELEMENT;
      if (raw.length !== (mapResolution - 1) * stride + rowElements) throw new Error('Unexpected irradiance readback row layout.');
      const pixels = new Float32Array(mapResolution * rowElements);
      const half = raw instanceof Uint16Array;
      for (let row = 0; row < mapResolution; row++) {
        const source = row * stride, destination = row * rowElements;
        for (let component = 0; component < rowElements; component++) {
          const value = half ? THREE.DataUtils.fromHalfFloat(raw[source + component]) : raw[source + component];
          if (!Number.isFinite(value)) throw new Error('Irradiance map contains a non-finite value.');
          pixels[destination + component] = value;
        }
      }
      return { width: mapResolution, height: mapResolution, halfExtent: HALF_EXTENT, pixels };
    } finally { readingIrradiance = false; }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    probeNode.dispose();
    geometry.dispose();
    material.dispose();
    heightTexture.dispose();
    target.dispose();
    scene.clear();
  }

  const metadata = Object.freeze({
    grid, rayResolution, mapResolution, halfExtent: HALF_EXTENT, splatWidth: SPLAT_WIDTH,
    sourceFlux: 4, fresnel: true, ior: IOR,
  });
  return { texture: target.texture, heightTexture, metadata, update, render, probe, readIrradiance, dispose };
}

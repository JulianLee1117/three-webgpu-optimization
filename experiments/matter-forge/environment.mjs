import * as THREE from 'three';

/**
 * Procedural set dressing for Matter Forge. Supplied collider meshes match their
 * boxes exactly. Open anchor braces show constraint extents, without filling them.
 * The distant architecture and water are scenery, not simulation geometry.
 * No bridge, simulated material, load motion, or rover crossing is fabricated here.
 * update(seconds) only scrolls the distant water texture; dispose() owns this group.
 */
export function buildEnvironment(scene, { N = 24, colliders = [], anchors = [] } = {}) {
  if (!scene?.isScene) throw Error('buildEnvironment requires a Three Scene');
  if (!Number.isFinite(N) || N < 8 || N > 128) throw Error('Environment N must be in [8, 128]');
  if (!Array.isArray(colliders) || colliders.length > 64 || !Array.isArray(anchors) || anchors.length > 32) throw Error('Environment box count exceeds its bounded scope');
  const checked = (box, kind, index) => {
    if (!box || !Array.isArray(box.min) || !Array.isArray(box.max) || box.min.length !== 3 || box.max.length !== 3 ||
      box.min.some((v, i) => !Number.isFinite(v) || !Number.isFinite(box.max[i]) || Math.abs(v) > N * 4 || Math.abs(box.max[i]) > N * 4 || box.max[i] <= v)) throw Error(`Invalid ${kind} box ${index}`);
    return { name: String(box.name ?? `${kind}-${index}`), min: [...box.min], max: [...box.max] };
  };
  const boxes = colliders.map((b, i) => checked(b, 'collider', i));
  const clamps = anchors.map((b, i) => checked(b, 'anchor', i));
  const scale = N / 24;
  const group = new THREE.Group(); group.name = 'Matter Forge industrial spillway';
  const geometries = new Set(), materials = new Set(), textures = new Set();
  const geometrical = g => (geometries.add(g), g);
  const material = m => (materials.add(m), m);
  const ownedTexture = t => (textures.add(t), t);
  const boxGeometry = geometrical(new THREE.BoxGeometry(1, 1, 1));
  const cylinderGeometry = geometrical(new THREE.CylinderGeometry(.5, .5, 1, 8, 1));
  const coneGeometry = geometrical(new THREE.ConeGeometry(1, 1, 7, 1));
  let randomState = 61741;
  const random = () => { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; };

  function surfaceTexture(kind) {
    const size = 256;
    if (typeof document === 'undefined') return null; // Geometry can be inspected without a browser/GPU.
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = size;
    const context = canvas.getContext('2d'); if (!context) return null;
    const pixels = context.createImageData(size, size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const grain = (random() - .5) * (kind === 'water' ? 11 : 30);
      const streak = kind === 'water' ? 15 * Math.sin(y * .28 + Math.sin(x * .03) * 2) : 6 * Math.sin(x * .12) + 4 * Math.cos(y * .017);
      const base = kind === 'water' ? [89, 123, 127] : [150, 149, 137];
      for (let c = 0; c < 3; c++) pixels.data[i + c] = Math.max(0, Math.min(255, base[c] + grain + streak));
      pixels.data[i + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    if (kind !== 'water') {
      context.lineWidth = 2;
      for (let y = 64; y < size; y += 64) {
        context.strokeStyle = 'rgba(35,43,44,.23)'; context.beginPath(); context.moveTo(0, y); context.lineTo(size, y); context.stroke();
        context.fillStyle = 'rgba(225,228,214,.10)'; context.fillRect(0, y + 2, size, 1);
      }
      for (let i = 0; i < 20; i++) {
        const x = random() * size, y = random() * size;
        context.strokeStyle = 'rgba(37,45,43,.12)'; context.lineWidth = 1 + random() * 3;
        context.beginPath(); context.moveTo(x, y); context.lineTo(x + 2, y + 15 + random() * 50); context.stroke();
      }
    }
    const texture = ownedTexture(new THREE.CanvasTexture(canvas));
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(kind === 'water' ? 16 : 2, kind === 'water' ? 16 : 2);
    texture.anisotropy = 4;
    return texture;
  }

  const concreteTexture = surfaceTexture('concrete');
  const waterTexture = surfaceTexture('water');
  const concrete = material(new THREE.MeshStandardMaterial({ color: 0x768387, map: concreteTexture, roughness: .87, metalness: .04 }));
  const darkConcrete = material(new THREE.MeshStandardMaterial({ color: 0x364a50, map: concreteTexture, roughness: .94, metalness: .02 }));
  const paleConcrete = material(new THREE.MeshStandardMaterial({ color: 0xa0a499, map: concreteTexture, roughness: .78, metalness: .04 }));
  const iron = material(new THREE.MeshStandardMaterial({ color: 0x27363c, roughness: .43, metalness: .75 }));
  const rust = material(new THREE.MeshStandardMaterial({ color: 0x80503a, roughness: .69, metalness: .52 }));
  const amber = material(new THREE.MeshStandardMaterial({ color: 0xb7792c, roughness: .57, metalness: .38 }));
  const stripe = material(new THREE.MeshStandardMaterial({ color: 0xe3b458, roughness: .76, metalness: .1 }));
  const black = material(new THREE.MeshStandardMaterial({ color: 0x111b20, roughness: .62, metalness: .3 }));
  const glow = material(new THREE.MeshBasicMaterial({ color: 0xffd28a, toneMapped: false }));
  const mountainMaterial = material(new THREE.MeshStandardMaterial({ color: 0x283c42, roughness: 1, flatShading: true }));
  const batches = new Map();
  const temp = new THREE.Object3D(), up = new THREE.Vector3(0, 1, 0), direction = new THREE.Vector3();
  function instance(geometry, mat, position, dimensions, quaternion) {
    let byGeometry = batches.get(geometry); if (!byGeometry) batches.set(geometry, byGeometry = new Map());
    let matrices = byGeometry.get(mat); if (!matrices) byGeometry.set(mat, matrices = []);
    temp.position.set(...position); temp.scale.set(...dimensions); temp.quaternion.identity();
    if (quaternion) temp.quaternion.copy(quaternion);
    temp.updateMatrix(); matrices.push(temp.matrix.clone());
  }
  const block = (mat, center, dimensions) => instance(boxGeometry, mat, center, dimensions);
  function beam(mat, a, b, radius = .045 * scale) {
    direction.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]); const length = direction.length();
    if (length < 1e-8) return;
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction.multiplyScalar(1 / length));
    instance(cylinderGeometry, mat, a.map((v, i) => (v + b[i]) / 2), [2 * radius, length, 2 * radius], quaternion);
  }
  const midpoint = b => b.min.map((v, i) => (v + b.max[i]) / 2);
  const dimensions = b => b.min.map((v, i) => b.max[i] - v);

  // Exact physical boxes are separate named meshes for inspectability.
  for (const box of boxes) {
    const metallic = /metal|steel|iron|grille|gate|plate/i.test(box.name);
    const mesh = new THREE.Mesh(boxGeometry, metallic ? iron : concrete);
    mesh.name = `Exact collider: ${box.name}`; mesh.position.set(...midpoint(box)); mesh.scale.set(...dimensions(box));
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.collider = { name: box.name, min: [...box.min], max: [...box.max] };
    group.add(mesh);
    if (metallic) continue;
    const [x0, y0, z0] = box.min, [x1, y1, z1] = box.max;
    const width = x1 - x0, height = y1 - y0, depth = z1 - z0;
    // Scenery footing is confined below the supplied bank's lower face. The
    // separately named collision box above remains exactly inspectable.
    if (width >= 2 * scale && depth >= 2 * scale && y0 > -6 * scale) {
      block(darkConcrete, [(x0 + x1) / 2, (y0 - 6 * scale) / 2, (z0 + z1) / 2], [width, y0 + 6 * scale, depth]);
    }
    // Thin facade details sit on the supplied box surfaces; no bridge surface is added.
    if (width >= 2 * scale && depth >= 2 * scale) {
      for (const z of [z0, z1]) {
        block(paleConcrete, [(x0 + x1) / 2, y1 - .18 * scale, z], [width, .24 * scale, .075 * scale]);
        const front = z === z1 ? 1 : -1;
        for (let x = x0 + .5 * scale; x < x1 - .1 * scale; x += 1.5 * scale) {
          // All railing posts stay at the z perimeter. x-facing bridge access stays open.
          beam(iron, [x, y1, z], [x, y1 + .9 * scale, z], .045 * scale);
          block(iron, [x, y1 + .025 * scale, z], [.18 * scale, .05 * scale, .18 * scale]);
        }
        beam(rust, [x0 + .2 * scale, y1 + .9 * scale, z], [x1 - .2 * scale, y1 + .9 * scale, z]);
        beam(iron, [x0 + .2 * scale, y1 + .43 * scale, z], [x1 - .2 * scale, y1 + .43 * scale, z], .026 * scale);
        if (height > 2 * scale) for (let x = x0 + .75 * scale; x < x1 - .25 * scale; x += 1.75 * scale) {
          block(darkConcrete, [x, y0 + height * .4, z + front * .025 * scale], [.22 * scale, height * .68, .05 * scale]);
          for (const yy of [y0 + .6 * scale, y1 - .75 * scale]) block(iron, [x, yy, z + front * .04 * scale], [.08 * scale, .08 * scale, .035 * scale]);
        }
      }
      for (let x = x0 + .15 * scale; x < x1 - .15 * scale; x += .46 * scale) {
        block(stripe, [x, y1 + .005 * scale, z1 - .15 * scale], [.24 * scale, .01 * scale, .16 * scale]);
      }
    }
  }

  for (const box of clamps) {
    const [x0, y0, z0] = box.min, [x1, y1, z1] = box.max;
    // Open 12-edge frame shows the complete clamped volume without an opaque fill.
    for (const y of [y0, y1]) for (const z of [z0, z1]) beam(amber, [x0, y, z], [x1, y, z], .036 * scale);
    for (const x of [x0, x1]) for (const z of [z0, z1]) beam(iron, [x, y0, z], [x, y1, z], .045 * scale);
    for (const x of [x0, x1]) for (const y of [y0, y1]) beam(amber, [x, y, z0], [x, y, z1], .036 * scale);
    for (const x of [x0, x1]) for (const z of [z0, z1]) {
      block(iron, [x, y0, z], [.21 * scale, .10 * scale, .21 * scale]);
      block(glow, [x, y1, z], [.075 * scale, .075 * scale, .075 * scale]);
    }
  }

  // Architecture stays behind the simulation (negative z) or outside its x edges.
  const backZ = -5 * scale, wallY = 4 * scale;
  block(darkConcrete, [N * .5, wallY, backZ], [N * 2.5, 18 * scale, 2.6 * scale]);
  block(paleConcrete, [N * .5, 13.2 * scale, backZ], [N * 2.55, .65 * scale, 3.1 * scale]);
  for (let i = -3; i <= 7; i++) {
    const x = i * 5 * scale;
    block(concrete, [x, 4.1 * scale, backZ + 1.25 * scale], [1.05 * scale, 18.5 * scale, 1.1 * scale]);
    block(iron, [x + 2.5 * scale, 5.7 * scale, backZ + 1.35 * scale], [2.5 * scale, 6.5 * scale, .12 * scale]);
    for (let j = 0; j < 5; j++) beam(rust, [x + (1.5 + j * .48) * scale, 2.5 * scale, backZ + 1.48 * scale], [x + (1.5 + j * .48) * scale, 8.9 * scale, backZ + 1.48 * scale], .045 * scale);
  }
  // Upper maintenance walkway and guardrail, fully outside the playable volume.
  const walkwayZ = backZ + 2.4 * scale;
  block(iron, [N * .5, 13.7 * scale, walkwayZ], [N * 2.5, .18 * scale, 1.4 * scale]);
  for (let i = -8; i <= 20; i++) {
    const x = i * 2 * scale;
    beam(iron, [x, 13.7 * scale, walkwayZ + .6 * scale], [x, 14.8 * scale, walkwayZ + .6 * scale]);
  }
  beam(rust, [-16 * scale, 14.8 * scale, walkwayZ + .6 * scale], [40 * scale, 14.8 * scale, walkwayZ + .6 * scale]);
  beam(iron, [-16 * scale, 14.25 * scale, walkwayZ + .6 * scale], [40 * scale, 14.25 * scale, walkwayZ + .6 * scale], .026 * scale);

  // Machinery, drain pipes, cabinets and towers in the background.
  for (const x of [-7 * scale, 29 * scale]) {
    block(concrete, [x, 8 * scale, -8 * scale], [4 * scale, 26 * scale, 5 * scale]);
    block(iron, [x, 21.4 * scale, -8 * scale], [5 * scale, .4 * scale, 6 * scale]);
    for (const xx of [x - 1.5 * scale, x + 1.5 * scale]) {
      beam(iron, [xx, 21.6 * scale, -6 * scale], [xx, 25 * scale, -6 * scale], .07 * scale);
      beam(iron, [xx, 21.6 * scale, -10 * scale], [xx, 25 * scale, -10 * scale], .07 * scale);
    }
    block(iron, [x, 25 * scale, -8 * scale], [4.2 * scale, .2 * scale, 4.4 * scale]);
    block(amber, [x, 14.8 * scale, -6.3 * scale], [1.4 * scale, 1.8 * scale, .7 * scale]);
    beam(rust, [x + 1.2 * scale, 15 * scale, -5 * scale], [x + 1.2 * scale, -4 * scale, -5 * scale], .18 * scale);
  }
  // Distant banks and mountains establish depth without adding a fake playable floor.
  for (const x of [-14 * scale, 38 * scale]) block(darkConcrete, [x, -3 * scale, N * .35], [14 * scale, 9 * scale, 45 * scale]);
  for (let i = 0; i < 15; i++) {
    const x = (i - 7) * 9 * scale, height = (12 + random() * 17) * scale;
    const quaternion = new THREE.Quaternion().setFromAxisAngle(up, random() * Math.PI);
    instance(coneGeometry, mountainMaterial, [x, height / 2 - 7 * scale, -35 * scale - random() * 18 * scale], [(10 + random() * 9) * scale, height, (10 + random() * 9) * scale], quaternion);
  }

  const waterMaterial = material(new THREE.MeshStandardMaterial({ color: 0x173039, map: waterTexture, roughness: .24, metalness: .56 }));
  const water = new THREE.Mesh(geometrical(new THREE.PlaneGeometry(N * 6, N * 6)), waterMaterial);
  water.name = 'Distant water scenery (not simulated)'; water.rotation.x = -Math.PI / 2;
  water.position.set(N / 2, -5.5 * scale, N / 2); water.receiveShadow = true; group.add(water);
  function lamp(x, y, z, pointLight = false) {
    beam(iron, [x, y - 2.6 * scale, z], [x, y, z], .08 * scale);
    beam(rust, [x, y, z], [x + .6 * scale, y, z], .07 * scale);
    block(iron, [x + .62 * scale, y - .09 * scale, z], [.6 * scale, .24 * scale, .48 * scale]);
    block(glow, [x + .62 * scale, y - .22 * scale, z], [.42 * scale, .018 * scale, .30 * scale]);
    if (pointLight) {
      const light = new THREE.PointLight(0xffbf72, 48 * scale * scale, 12 * scale, 2);
      light.position.set(x + .62 * scale, y - .4 * scale, z + .15 * scale); group.add(light);
    }
  }
  for (const x of [-9 * scale, -1 * scale, 7 * scale, 15 * scale, 23 * scale, 31 * scale]) lamp(x, 15.3 * scale, -2.4 * scale);
  // Exactly two local lights, placed at bank outer corners rather than over the bridge.
  const banks = boxes.filter(b => !/metal|steel|iron|grille|gate|plate/i.test(b.name) && b.max[0] - b.min[0] >= 2 * scale && b.max[2] - b.min[2] >= 2 * scale).slice(0, 2);
  banks.forEach((b, i) => lamp(i === 0 ? b.min[0] + .3 * scale : b.max[0] - .8 * scale, b.max[1] + 2.3 * scale, b.min[2] + .12 * scale, true));

  // Batch repeated geometry/material combinations into a small number of draws.
  let instanceCount = 0, triangles = 0;
  for (const [geometry, byMaterial] of batches) for (const [mat, matrices] of byMaterial) {
    const mesh = new THREE.InstancedMesh(geometry, mat, matrices.length);
    mesh.name = 'Batched industrial scenery'; matrices.forEach((matrix, i) => mesh.setMatrixAt(i, matrix));
    mesh.instanceMatrix.needsUpdate = true; mesh.computeBoundingSphere();
    mesh.castShadow = false; mesh.receiveShadow = true; group.add(mesh);
    instanceCount += matrices.length; triangles += (geometry.index?.count ?? geometry.attributes.position.count) / 3 * matrices.length;
  }
  const skyGeometry = geometrical(new THREE.SphereGeometry(N * 8, 24, 12));
  const skyPositions = skyGeometry.attributes.position, colors = new Float32Array(skyPositions.count * 3);
  const skyTop = new THREE.Color(0x111e31), skyHorizon = new THREE.Color(0x6b737b), color = new THREE.Color();
  for (let i = 0; i < skyPositions.count; i++) {
    const t = Math.max(0, Math.min(1, (skyPositions.getY(i) / (N * 8) + .12) * 1.6));
    color.copy(skyHorizon).lerp(skyTop, t); color.toArray(colors, i * 3);
  }
  skyGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const sky = new THREE.Mesh(skyGeometry, material(new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, depthWrite: false, fog: false })));
  sky.name = 'Dusk sky'; sky.position.set(N / 2, 0, N / 2); sky.renderOrder = -1000; group.add(sky);
  const hemisphere = new THREE.HemisphereLight(0xc6dce8, 0x343c39, 1.6); group.add(hemisphere);
  const sun = new THREE.DirectionalLight(0xffd5b0, 2.5); sun.position.set(-.7 * N, 1.6 * N, .8 * N);
  sun.target.position.set(N * .5, N * .32, N * .5); group.add(sun, sun.target);
  sun.castShadow = true; sun.shadow.mapSize.set(512, 512); sun.shadow.camera.near = .5 * scale; sun.shadow.camera.far = N * 5;
  sun.shadow.camera.left = sun.shadow.camera.bottom = -N * .9; sun.shadow.camera.right = sun.shadow.camera.top = N * .9;
  sun.shadow.bias = -.0004; sun.shadow.normalBias = .035 * scale;
  const previousFog = scene.fog, fog = new THREE.FogExp2(0x4b5e68, .012 / scale);
  scene.fog = fog; scene.add(group);
  group.userData.environment = { exactColliderCount: boxes.length, openAnchorCount: clamps.length, instanceCount, approximateSceneryTriangles: triangles, physicsSuppliedByCaller: true, maximumShadowMap: [512, 512], movingGameplayObjects: 0 };
  let disposed = false;
  return {
    group,
    update(seconds) {
      if (disposed || !Number.isFinite(seconds)) return;
      if (waterTexture) { waterTexture.offset.x = (seconds * .004) % 1; waterTexture.offset.y = (seconds * .011) % 1; }
    },
    dispose() {
      if (disposed) return; disposed = true;
      group.removeFromParent(); if (scene.fog === fog) scene.fog = previousFog;
      sun.shadow.dispose(); geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); textures.forEach(t => t.dispose());
      // InstancedMesh owns GPU-side instance resources separately from shared geometry.
      group.traverse(object => { if (object.isInstancedMesh) object.dispose(); });
    }
  };
}

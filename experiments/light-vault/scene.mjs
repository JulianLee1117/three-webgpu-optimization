import * as THREE from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import {RectAreaLightTexturesLib} from 'three/addons/lights/RectAreaLightTexturesLib.js';

/**
 * Artwork for the Glass Vault. Display distances are schematic: lateral sizes
 * are magnified independently of the optical solver's millimetre coordinates.
 * This module produces no optical field. Main owns the phase and irradiance
 * materials, propagation, capture conditions, and the physical distance scale.
 */
export function createVaultScene(scene) {
  const ltc = RectAreaLightTexturesLib.LTC_HALF_1
    ? RectAreaLightTexturesLib : RectAreaLightTexturesLib.init();
  THREE.RectAreaLightNode?.setLTC(ltc);
  const root = new THREE.Group();
  root.name = 'The Glass Vault';
  scene.add(root);
  const geometries = new Set(), materials = new Set(), textures = new Set();
  const lights = [];
  let disposed = false;
  const ownG = value => (geometries.add(value), value);
  const ownM = value => (materials.add(value), value);
  const material = (color, extra = {}) => ownM(new THREE.MeshStandardMaterial({
    color, roughness: .5, ...extra,
  }));
  const obsidian = material(0x162326, {metalness: .35, roughness: .35});
  const interior = material(0x0c2527, {roughness: .87});
  const brass = material(0xb79b63, {metalness: .76, roughness: .45});
  const engraving = material(0xc6b58e, {metalness: .64, roughness: .51});
  const rosette = material(0xe1d3ac, {metalness: .55, roughness: .47});
  const agedGold = material(0x685334, {metalness: .67, roughness: .59});
  const darkMetal = material(0x19272d, {metalness: .58, roughness: .43});
  const velvet = material(0x142427, {roughness: .97});
  const goldLight = material(0xffda91, {
    emissive: 0xffbc57, emissiveIntensity: .6, roughness: .44, metalness: .12,
  });
  const gemMaterial = ownM(new THREE.MeshPhysicalMaterial({
    color: 0x43b6b0, metalness: .09, roughness: .18,
    clearcoat: 1, clearcoatRoughness: .12, flatShading: true,
    emissive: 0x164b45, emissiveIntensity: .35,
  }));
  const rounded = ownG(new RoundedBoxGeometry(1, 1, 1, 3, .045));
  const box = ownG(new THREE.BoxGeometry(1, 1, 1));
  const cylinder = ownG(new THREE.CylinderGeometry(1, 1, 1, 48));
  const ring = ownG(new THREE.TorusGeometry(1, .014, 8, 96));
  const mesh = (geometry, mat, position, scale = [1, 1, 1], parent = root) => {
    const object = new THREE.Mesh(geometry, mat);
    object.position.set(...position); object.scale.set(...scale);
    object.castShadow = object.receiveShadow = true;
    parent.add(object); return object;
  };
  const slab = (mat, position, scale, parent = root) => mesh(rounded, mat, position, scale, parent);
  const xRing = (radius, position, mat = brass, parent = root) => {
    const object = mesh(ring, mat, position, [radius, radius, radius], parent);
    object.rotation.y = Math.PI / 2; return object;
  };
  function lamp(light, position) {
    light.position.set(...position); lights.push(light); root.add(light); return light;
  }

  // One continuous velvet stage; the gap from seal to receiver is unobstructed.
  const floor = mesh(ownG(new THREE.PlaneGeometry(160, 160)), velvet, [0, -.065, 0]);
  floor.rotation.x = -Math.PI / 2; floor.castShadow = false;
  slab(darkMetal, [-3.3, .04, 0], [2.08, .15, 3.08]);
  slab(agedGold, [-3.3, .118, 0], [1.99, .018, 2.99]);

  // A genuinely hollow case, with a velvet interior revealed by the door.
  const body = new THREE.Group(); body.name = 'Hollow obsidian case'; root.add(body);
  slab(obsidian, [-4.08, 1.52, 0], [.18, 2.75, 2.77], body);
  slab(obsidian, [-3.30, 2.86, 0], [1.75, .17, 2.77], body);
  slab(obsidian, [-3.30, .22, 0], [1.75, .17, 2.77], body);
  for (const z of [-1.31, 1.31]) slab(obsidian, [-3.30, 1.54, z], [1.75, 2.67, .16], body);
  slab(interior, [-3.98, 1.53, 0], [.014, 2.46, 2.46], body);
  slab(interior, [-3.30, .315, 0], [1.40, .018, 2.43], body);
  for (const x of [-4.13, -2.51]) {
    for (const z of [-1.40, 1.40]) slab(brass, [x, 1.53, z], [.035, 2.73, .035], body);
    for (const y of [.17, 2.90]) slab(brass, [x, y, 0], [.037, .035, 2.80], body);
  }
  // A few broad, engraved inlays give material richness without prop clutter.
  for (const z of [-1.402, 1.402]) {
    slab(engraving, [-3.32, 1.56, z], [1.33, 2.35, .012], body);
    slab(obsidian, [-3.32, 1.56, z + Math.sign(z) * .009], [1.13, 2.08, .012], body);
  }
  for (const x of [-3.92, -2.69]) for (const z of [-1.16, 1.16]) {
    mesh(cylinder, brass, [x, .105, z], [.095, .15, .095]);
  }

  // The optical seal is static while solving. Opening only occurs after capture.
  const doorPivot = new THREE.Group();
  doorPivot.name = 'Vault hinge'; doorPivot.position.set(-2.50, 1.50, -1.30);
  root.add(doorPivot);
  slab(obsidian, [0, 0, 1.30], [.11, 2.65, 2.62], doorPivot);
  for (const z of [.015, 2.585]) slab(brass, [.065, 0, z], [.022, 2.59, .033], doorPivot);
  for (const y of [-1.28, 1.28]) slab(brass, [.065, y, 1.30], [.022, .032, 2.59], doorPivot);
  const ornament = mesh(ownG(new THREE.RingGeometry(.735, 1.045, 96)), rosette,
    [.073, .15, 1.30], [1, 1, 1], doorPivot);
  ornament.rotation.y = Math.PI / 2;
  xRing(1.069, [.078, .15, 1.30], brass, doorPivot);
  xRing(.724, [.085, .15, 1.30], brass, doorPivot);
  // The source plate is a separate, fixed instrument. The door animation must
  // never rotate the optical coordinate system or change the propagated field.
  const sourceMount = new THREE.Group(); sourceMount.name = 'Stationary optical seal';
  sourceMount.position.set(-2.4, 1.65, 0); root.add(sourceMount);
  const sealMaterial = ownM(new THREE.MeshBasicMaterial({color: 0x304744, side: THREE.DoubleSide}));
  const phaseMesh = mesh(ownG(new THREE.CircleGeometry(.70, 96)), sealMaterial,
    [0, 0, 0], [1, 1, 1], sourceMount);
  phaseMesh.name = 'Actual optical phase plate'; phaseMesh.rotation.y = Math.PI / 2;
  phaseMesh.castShadow = phaseMesh.receiveShadow = false;
  xRing(.714, [.004, 0, 0], brass, sourceMount);
  for (const z of [-.78, .78]) {
    slab(darkMetal, [-.015, -.83, z], [.045, 1.31, .040], sourceMount);
    slab(brass, [-.015, -.34, z * .91], [.048, .16, .15], sourceMount);
  }
  slab(darkMetal, [-.02, -1.50, 0], [.21, .045, 1.82], sourceMount);
  for (const y of [-.96, .98]) {
    mesh(cylinder, brass, [-2.48, 1.50 + y, -1.345], [.074, .31, .074]);
    mesh(cylinder, darkMetal, [-2.48, 1.50 + y, -1.345], [.080, .024, .080]);
  }
  const fastenerGeometry = ownG(new THREE.CylinderGeometry(.026, .026, .016, 12));
  for (const y of [-1.17, 1.17]) for (const z of [.15, 2.45]) {
    const screw = mesh(fastenerGeometry, brass, [.070, y, z], [1, 1, 1], doorPivot);
    screw.rotation.z = Math.PI / 2;
  }
  // Fine repeated notches are a single mesh, outside the active phase aperture.
  const ticks = new THREE.InstancedMesh(box, agedGold, 48);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < 48; i++) {
    const angle = i * Math.PI / 24, radius = 1.09;
    dummy.position.set(.074, .15 + Math.cos(angle) * radius, 1.30 + Math.sin(angle) * radius);
    dummy.rotation.set(angle, 0, 0); dummy.scale.set(.012, i % 4 === 0 ? .048 : .024, .009);
    dummy.updateMatrix(); ticks.setMatrixAt(i, dummy.matrix);
  }
  ticks.instanceMatrix.needsUpdate = true; doorPivot.add(ticks);

  // A restrained jewel is a reward behind the real hinged door.
  mesh(cylinder, agedGold, [-3.32, .51, 0], [.40, .20, .40]);
  mesh(cylinder, brass, [-3.32, .623, 0], [.36, .027, .36]);
  const jewel = mesh(ownG(new THREE.IcosahedronGeometry(.38, 0)), gemMaterial, [-3.32, 1.12, 0]);
  jewel.rotation.set(.3, .2, .1);
  const innerGlow = lamp(new THREE.PointLight(0x72f3d5, .4, 3, 2), [-3.3, 1.62, .2]);
  const crown = mesh(ownG(new THREE.TorusGeometry(.41, .012, 8, 64)), brass, [-3.32, 1.12, 0]);
  crown.rotation.set(.5, .2, Math.PI / 4);

  const receiverGroup = new THREE.Group(); receiverGroup.name = 'Draggable receiver card';
  receiverGroup.position.set(.9, 1.65, 0); root.add(receiverGroup);
  const receiverMaterial = ownM(new THREE.MeshBasicMaterial({color: 0x152127, side: THREE.DoubleSide}));
  const receiverMesh = mesh(ownG(new THREE.PlaneGeometry(2.3, 2.3)), receiverMaterial,
    [0, 0, 0], [1, 1, 1], receiverGroup);
  receiverMesh.name = 'Computed wave intensity receiver'; receiverMesh.rotation.y = -Math.PI / 2;
  receiverMesh.castShadow = receiverMesh.receiveShadow = false;
  for (const z of [-1.18, 1.18]) slab(brass, [0, 0, z], [.038, 2.40, .048], receiverGroup);
  for (const y of [-1.18, 1.18]) slab(brass, [0, y, 0], [.038, .048, 2.40], receiverGroup);
  for (const z of [-1.15, 1.15]) for (const y of [-1.15, 1.15]) {
    slab(agedGold, [.013, y, z], [.050, .16, .16], receiverGroup);
  }
  // An obvious grip connects dragging to the physical card, without a slider rail.
  slab(darkMetal, [0, -1.34, 0], [.095, .25, .52], receiverGroup);
  slab(engraving, [.053, -1.34, 0], [.014, .18, .40], receiverGroup);
  slab(brass, [0, -1.495, 0], [.14, .065, .68], receiverGroup);

  lamp(new THREE.HemisphereLight(0xbdd7e0, 0x10181b, .52), [0, 7, 0]);
  const key = lamp(new THREE.SpotLight(0xffe4b8, 150, 24, .66, .85, 2), [1.5, 7.5, 4.5]);
  key.target.position.set(-2.8, 1.25, 0); root.add(key.target);
  key.castShadow = true; key.shadow.mapSize.set(1024, 1024);
  key.shadow.bias = -.00015; key.shadow.normalBias = .025;
  key.shadow.camera.near = .5; key.shadow.camera.far = 24;
  const fill = lamp(new THREE.RectAreaLight(0x9bd4df, 3.1, 5, 6), [-4.9, 4.5, -3.5]);
  fill.lookAt(-3, 1.5, 0);
  const warm = lamp(new THREE.RectAreaLight(0xffd195, 2.1, 3, 5), [3, 4, 5.5]);
  warm.lookAt(-2.4, 1.5, 0);

  // All quadrants are actual base-color artwork. They are not measured PBR maps.
  const atlasUrl = new URL('./assets/material-atlas.png', import.meta.url).href;
  const readyPromise = typeof document === 'undefined' ? Promise.resolve(false) : new Promise(resolve => {
    new THREE.TextureLoader().load(atlasUrl, atlas => {
      if (disposed) { atlas.dispose(); resolve(false); return; }
      textures.add(atlas); atlas.colorSpace = THREE.SRGBColorSpace;
      function tile(column, row) {
        const texture = atlas.clone();
        // Half a texel inset prevents neighbouring artwork bleeding at edges.
        const inset = .5 / atlas.image.width;
        texture.repeat.set(.5 - 2 * inset, .5 - 2 * inset);
        texture.offset.set(column * .5 + inset, (1 - row) * .5 + inset);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = 4; texture.needsUpdate = true;
        textures.add(texture); return texture;
      }
      obsidian.map = tile(0, 0); obsidian.color.set(0xffffff);
      engraving.map = tile(1, 0); engraving.color.set(0xcabd9e);
      velvet.map = tile(0, 1); velvet.color.set(0xa6a6a6);
      interior.map = velvet.map; interior.color.set(0x99a8a8);
      rosette.map = tile(1, 1); rosette.color.set(0xffffff);
      for (const mat of [obsidian, engraving, velvet, interior, rosette]) mat.needsUpdate = true;
      resolve(true);
    }, undefined, () => resolve(false));
  });

  function update({receiverX, opening = 0} = {}) {
    if (Number.isFinite(receiverX)) receiverGroup.position.x = receiverX;
    const amount = THREE.MathUtils.clamp(Number.isFinite(opening) ? opening : 0, 0, 1);
    doorPivot.rotation.y = amount * 1.36;
    innerGlow.intensity = .4 + amount * 1.1;
  }
  function dispose() {
    if (disposed) return;
    disposed = true; scene.remove(root);
    for (const geometry of geometries) geometry.dispose();
    for (const mat of materials) mat.dispose();
    for (const texture of textures) texture.dispose();
    ticks.dispose();
    for (const light of lights) light.shadow?.map?.dispose();
    root.clear();
  }
  return {root, receiverGroup, receiverMesh, phaseMesh, doorPivot, lights,
    readyPromise, atlasUrl, update, dispose};
}

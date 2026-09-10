import * as THREE from 'three';

/** Add six silk fins to the town's existing deformable surface field. */
export function addFins(town) {
  if (!town?.root || !Array.isArray(town.surfaces)) throw new TypeError('Expected a town with a root and deformable surfaces.');
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide, roughness: .72, metalness: 0,
  });
  const meshes = [], entries = [], cream = new THREE.Color(0xf4dcac), red = new THREE.Color(0xc67959), rib = new THREE.Color(0xd1b487);
  const spans = 12, chords = 10;
  for (const center of [-2.3, 0, 2.3]) for (const side of [-1, 1]) {
    const positions = [], colors = [], indices = [];
    for (let row = 0; row <= spans; row++) {
      const u = row / spans;
      const halfChord = .44 * Math.sqrt(1 - .975 * u);
      for (let col = 0; col <= chords; col++) {
        const v = col / chords * 2 - 1;
        const sweep = -.88 * u * u;
        const x = center + sweep + v * halfChord;
        const y = -.155 - .33 * Math.sin(u * Math.PI * .77) + .09 * v * v * u;
        const z = side * (1.55 + 1.30 * u + .065 * (1 - v * v) * u);
        positions.push(x, y, z);
        const band = row === 3 || row === 4 || row === 9;
        const c = (band ? red : cream).clone();
        // Chordwise shading makes the longitudinal sewn ribs visible without
        // separate line draw calls or a texture dependency.
        if (col % 3 === 0) c.lerp(rib, .26);
        c.multiplyScalar(.94 + .06 * Math.cos(v * Math.PI));
        colors.push(c.r, c.g, c.b);
      }
    }
    for (let row = 0; row < spans; row++) for (let col = 0; col < chords; col++) {
      const a = row * (chords + 1) + col, b = a + 1, c = a + chords + 1, d = c + 1;
      if (side > 0) indices.push(a, c, b, b, c, d);
      else indices.push(a, b, c, b, d, c);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices); geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `Silk fin ${center} ${side > 0 ? 'starboard' : 'port'}`;
    mesh.castShadow = mesh.receiveShadow = true; mesh.frustumCulled = false;
    const entry = {mesh, restPositions: new Float32Array(positions)};
    town.root.add(mesh); town.surfaces.push(entry); meshes.push(mesh); entries.push(entry);
  }
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    for (const mesh of meshes) { mesh.removeFromParent(); mesh.geometry.dispose(); }
    for (const entry of entries) { const i = town.surfaces.indexOf(entry); if (i >= 0) town.surfaces.splice(i, 1); }
    material.dispose();
  };
  // The town removes itself from the scene when disposed, so the later scene
  // traversal cannot discover these added resources. Own their cleanup here.
  const previousDispose = town.dispose?.bind(town);
  town.dispose = () => { dispose(); previousDispose?.(); };
  return {meshes, dispose};
}

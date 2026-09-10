import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// Everything is either a deformable surface in town coordinates or a small,
// rigid attachment in its own local frame. The simulation owns their motion.
export function createTown() {
  const root = new THREE.Group();
  root.name = 'Porto Piccolo';
  const attachments = [], surfaces = [], residents = [];
  const geometries = new Set(), materials = new Set(), textures = new Set();
  const material = (color, options = {}) => {
    const value = new THREE.MeshStandardMaterial({ color, roughness: .82, ...options });
    materials.add(value);
    return value;
  };
  const texture = (paint, size = 256) => {
    if (typeof document === 'undefined') return null;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    paint(canvas.getContext('2d'), size);
    const result = new THREE.CanvasTexture(canvas);
    result.colorSpace = THREE.SRGBColorSpace;
    result.wrapS = result.wrapT = THREE.RepeatWrapping;
    result.anisotropy = 4;
    textures.add(result);
    return result;
  };
  const tileTexture = texture((ctx, n) => {
    ctx.fillStyle = '#bd6647'; ctx.fillRect(0, 0, n, n);
    for (let y = 0; y < 8; y++) for (let x = -1; x < 12; x++) {
      const px = x * n / 11 + (y % 2) * n / 22, py = y * n / 8;
      ctx.fillStyle = ['#c67450', '#b75a3d', '#d2825a', '#bd6246'][(x + y * 3 + 20) % 4];
      ctx.fillRect(px + 1, py + 1, n / 11 - 2, n / 8 - 2);
      ctx.strokeStyle = '#e69868'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px + 3, py + 3); ctx.lineTo(px + 3, py + n / 8 - 2); ctx.stroke();
      ctx.strokeStyle = '#783e32'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, py + n / 8 - 1); ctx.lineTo(px + n / 11, py + n / 8 - 1); ctx.stroke();
    }
  });
  const pavingTexture = texture((ctx, n) => {
    ctx.fillStyle = '#aa9f89'; ctx.fillRect(0, 0, n, n);
    for (let y = 0; y < 12; y++) for (let x = -1; x < 8; x++) {
      const px = x * n / 7 + (y % 2) * n / 14, py = y * n / 12;
      ctx.fillStyle = ['#d2c4a9', '#c1b39a', '#d6c9af', '#bcae95'][(x + y + 20) % 4];
      ctx.fillRect(px + 1, py + 1, n / 7 - 2, n / 12 - 2);
    }
  });
  if (pavingTexture) pavingTexture.repeat.set(5, 2.5);
  const plasterTexture = texture((ctx, n) => {
    ctx.fillStyle = '#f8f3df'; ctx.fillRect(0, 0, n, n);
    let seed = 19;
    for (let i = 0; i < 900; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const x = seed % n; seed = (seed * 1664525 + 1013904223) >>> 0;
      const y = seed % n;
      ctx.fillStyle = i % 3 ? 'rgba(130,109,84,.045)' : 'rgba(255,255,255,.18)';
      ctx.fillRect(x, y, 2 + i % 4, 1 + i % 3);
    }
  });
  const M = {
    walls: [0xefe3c5, 0xe5b783, 0xd9cfb0, 0xe8c9a2, 0xd7dccc].map(c => material(c, { map: plasterTexture })),
    roof: material(0xffffff, { map: tileTexture }), roofEdge: material(0xab5038),
    teal: material(0x387774), blue: material(0x4d7080), ivory: material(0xf4e4c1),
    window: material(0xf7be69, { emissive: 0xeab766, emissiveIntensity: .34, roughness: .35 }),
    glass: material(0x314f53, { roughness: .25, metalness: .16 }),
    door: material(0x675241), wood: material(0x947752), darkWood: material(0x534f45),
    metal: material(0x34494b, { roughness: .48, metalness: .5 }),
    stone: material(0xc0b395), paving: material(0xffffff, { map: pavingTexture }),
    foundation: material(0x938d7c), path: material(0xd5c5a4),
    green: material(0x527451), darkGreen: material(0x315c50), lightGreen: material(0x78925f),
    flower: material(0xd68568), lemon: material(0xe3b651), purple: material(0x9873a0),
    redCloth: material(0xd58465), canvas: material(0xf5dfb5),
    lantern: material(0xffd58b, { emissive: 0xffb752, emissiveIntensity: 1.15 }),
    waterline: material(0x637a77), shirt: material(0x64839b), skin: material(0xe9bc8d),
    trousers: material(0x3e5662), hair: material(0x4b433a), hat: material(0xe0c498),
  };
  const v = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1), matrix = new THREE.Matrix4();
  const coloredMaterials = new Map();
  function batchMaterial(mat, chunks) {
    if (mat.map || mat.emissive.getHex() !== 0) return mat;
    const finish = mat.metalness >= .3 ? 'metal' : mat.roughness < .4 ? 'glass' : 'matte';
    const key = `${finish}:${mat.side}`;
    if (!coloredMaterials.has(key)) {
      coloredMaterials.set(key, material(0xffffff, {
        vertexColors: true, side: mat.side,
        roughness: finish === 'glass' ? .25 : finish === 'metal' ? .48 : .82,
        metalness: finish === 'metal' ? .5 : finish === 'glass' ? .16 : 0,
      }));
    }
    for (const geometry of chunks) {
      const count = geometry.attributes.position.count, values = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) mat.color.toArray(values, i * 3);
      geometry.setAttribute('color', new THREE.BufferAttribute(values, 3));
    }
    return coloredMaterials.get(key);
  }
  function batch() {
    const parts = new Map();
    return {
      add(geo, mat, pos = [0, 0, 0], rot = [0, 0, 0]) {
        if (geo.index) { const source = geo; geo = source.toNonIndexed(); source.dispose(); }
        q.setFromEuler(new THREE.Euler(...rot)); v.fromArray(pos); matrix.compose(v, q, s);
        geo.applyMatrix4(matrix);
        if (!parts.has(mat)) parts.set(mat, []);
        parts.get(mat).push(geo);
      },
      box(w, h, d, mat, pos = [0, 0, 0], rot = [0, 0, 0]) { this.add(new THREE.BoxGeometry(w, h, d), mat, pos, rot); },
      cylinder(top, bottom, h, mat, pos = [0, 0, 0], rot = [0, 0, 0], segments = 10) {
        this.add(new THREE.CylinderGeometry(top, bottom, h, segments), mat, pos, rot);
      },
      sphere(radius, mat, pos = [0, 0, 0], scale = [1, 1, 1]) {
        const geo = new THREE.SphereGeometry(radius, 8, 6); geo.scale(...scale); this.add(geo, mat, pos);
      },
      finish(name) {
        const group = new THREE.Group(); group.name = name;
        const grouped = new Map();
        for (const [sourceMaterial, chunks] of parts) {
          const mat = batchMaterial(sourceMaterial, chunks);
          if (!grouped.has(mat)) grouped.set(mat, []);
          grouped.get(mat).push(...chunks);
        }
        for (const [mat, chunks] of grouped) {
          const geometry = mergeGeometries(chunks, false);
          chunks.forEach(g => g.dispose());
          if (!geometry) throw new Error(`Could not merge ${name}`);
          geometries.add(geometry);
          const mesh = new THREE.Mesh(geometry, mat);
          mesh.castShadow = mesh.receiveShadow = true;
          group.add(mesh);
        }
        return group;
      },
    };
  }
  function attach(object, anchor, baseYaw = 0, kind = 'prop', extra = {}) {
    const entry = { object, anchor: [...anchor], baseYaw, kind, ...extra };
    object.position.fromArray(anchor); object.rotation.y = baseYaw;
    object.userData.kind = kind; object.userData.anchor = entry.anchor;
    root.add(object); attachments.push(entry);
    return entry;
  }
  function surface(geometry, mat, name) {
    geometries.add(geometry);
    const mesh = new THREE.Mesh(geometry, mat); mesh.name = name;
    mesh.castShadow = mesh.receiveShadow = true;
    root.add(mesh);
    surfaces.push({ mesh, restPositions: new Float32Array(geometry.attributes.position.array) });
    return mesh;
  }
  function groundPlane(width, depth, y, segmentsX = 48, segmentsZ = 20) {
    const geometry = new THREE.PlaneGeometry(width, depth, segmentsX, segmentsZ);
    geometry.rotateX(-Math.PI / 2); geometry.translate(0, y, 0);
    return geometry;
  }
  // A gently pointed quay gives the whole town a boat-like silhouette without
  // making the houses themselves look like toys stuck on a rectangular board.
  const quay = groundPlane(8, 3.6, .035);
  const quayPos = quay.attributes.position;
  for (let i = 0; i < quayPos.count; i++) {
    const x = quayPos.getX(i), z = quayPos.getZ(i);
    const taper = 1 - .2 * Math.pow(Math.abs(x) / 4, 8);
    quayPos.setZ(i, z * taper);
  }
  quay.computeVertexNormals(); surface(quay, M.paving, 'Paved quay');
  const underside = new THREE.BoxGeometry(8, .25, 3.6, 48, 2, 20);
  const underPos = underside.attributes.position;
  for (let i = 0; i < underPos.count; i++) {
    const x = underPos.getX(i), y = underPos.getY(i), z = underPos.getZ(i);
    const taper = 1 - .2 * Math.pow(Math.abs(x) / 4, 8);
    underPos.setXYZ(i, x, y - .11, z * taper * (y < 0 ? .97 : 1));
  }
  underside.computeVertexNormals(); surface(underside, M.foundation, 'Stone foundation');
  surface(groundPlane(7.9, .62, .049, 48, 2), M.path, 'Central promenade');
  for (const z of [-.35, .35]) {
    const strip = groundPlane(7.9, .038, .052, 48, 1); strip.translate(0, 0, z);
    surface(strip, M.ivory, 'Promenade border');
  }
  for (const z of [-1.67, 1.67]) {
    const strip = groundPlane(7.3, .09, .059, 44, 1); strip.translate(0, 0, z);
    surface(strip, M.stone, 'Quayside coping');
  }

  function roofGeometry(w, d, y, rise) {
    const geometry = new THREE.BufferGeometry();
    const positions = [], uv = [];
    const tri = (a, b, c, ua, ub, uc) => { positions.push(...a, ...b, ...c); uv.push(...ua, ...ub, ...uc); };
    const x = w / 2, z = d / 2;
    tri([-x,y,z],[x,y,z],[x,y+rise,0],[0,0],[1,0],[1,1]);
    tri([-x,y,z],[x,y+rise,0],[-x,y+rise,0],[0,0],[1,1],[0,1]);
    tri([x,y,-z],[-x,y,-z],[-x,y+rise,0],[0,0],[1,0],[1,1]);
    tri([x,y,-z],[-x,y+rise,0],[x,y+rise,0],[0,0],[1,1],[0,1]);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geometry.computeVertexNormals();
    return geometry;
  }
  function signTexture(label, background) {
    return texture((ctx, n) => {
      ctx.fillStyle = background; ctx.fillRect(0, 0, n, n);
      ctx.strokeStyle = '#e9d6aa'; ctx.lineWidth = 5; ctx.strokeRect(6, 58, n - 12, 140);
      ctx.fillStyle = '#faeccd'; ctx.font = `600 ${label.length > 6 ? 28 : 36}px Georgia`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, n / 2, n / 2);
    });
  }
  function house(index, w, d, h, shop = false) {
    const b = batch(), wall = M.walls[index % M.walls.length], trim = index % 3 ? M.teal : M.blue;
    const front = d / 2;
    b.box(w, h, d, wall, [0, h / 2, 0]);
    b.box(w + .025, .09, d + .03, M.stone, [0, .045, 0]);
    b.box(w + .055, .035, d + .055, M.ivory, [0, h - .02, 0]);
    b.box(w + .035, .035, d + .02, M.ivory, [0, h * .5, 0]);
    // Small ivory quoins at the outside corners catch directional sunlight.
    for (const x of [-w / 2, w / 2]) for (const z of [-d / 2, d / 2]) {
      for (let row = 0; row < Math.floor(h / .115); row++) {
        b.box(.055, .04, .055, M.ivory, [x, .08 + row * .115, z]);
      }
    }
    const rise = .23 + .04 * (index % 3);
    b.add(roofGeometry(w + .15, d + .17, h, rise), M.roof);
    // The gable is solid plaster. Triangular ends retain proper silhouettes.
    for (const side of [-1, 1]) {
      const g = new THREE.BufferGeometry();
      const vertices = side < 0
        ? [-w/2,h,-d/2, -w/2,h,d/2, -w/2,h+rise,0]
        : [w/2,h,d/2, w/2,h,-d/2, w/2,h+rise,0];
      g.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      g.setAttribute('uv', new THREE.Float32BufferAttribute([0,0,1,0,.5,1], 2)); g.computeVertexNormals(); b.add(g, wall);
    }
    b.cylinder(.027, .027, w + .16, M.roofEdge, [0, h + rise + .008, 0], [0, 0, Math.PI / 2], 8);
    for (const z of [-front - .06, front + .06]) {
      b.box(w + .13, .035, .035, M.roofEdge, [0, h + .003, z]);
      b.box(w + .12, .023, .024, M.metal, [0, h - .035, z + Math.sign(z) * .008]);
    }
    const chimneyX = w * .29, chimneyZ = -d * .19;
    b.box(.115, .25, .13, M.walls[(index + 1) % M.walls.length], [chimneyX, h + rise + .035, chimneyZ]);
    b.box(.15, .035, .17, M.roofEdge, [chimneyX, h + rise + .17, chimneyZ]);
    b.box(.08, .006, .095, M.darkWood, [chimneyX, h + rise + .19, chimneyZ]);
    const window = (x, y, z, back = false, lit = true) => {
      const sign = back ? -1 : 1, width = .16, height = .21;
      b.box(width + .04, height + .04, .023, M.ivory, [x, y, z]);
      b.box(width, height, .027, lit ? M.window : M.glass, [x, y, z + sign * .012]);
      b.box(.012, height, .033, trim, [x, y, z + sign * .024]);
      b.box(width, .012, .033, trim, [x, y, z + sign * .024]);
      b.box(width + .07, .027, .065, M.ivory, [x, y - height / 2 - .025, z + sign * .013]);
      for (const sx of [-1, 1]) {
        b.box(.055, height + .02, .025, trim, [x + sx * .125, y, z + sign * .012], [0, sx * sign * -.16, 0]);
        for (let slat = -2; slat <= 2; slat++) b.box(.042, .006, .028, M.darkGreen, [x + sx * .125, y + slat * .035, z + sign * .028]);
      }
    };
    const upperY = Math.max(.44, h - .23);
    for (const x of [-w * .24, w * .24]) {
      window(x, upperY, front + .012, false, true);
      window(x, upperY, -front - .012, true, index % 2 === 0);
      if (h > .96) window(x, .32, -front - .012, true, index % 3 === 0);
    }
    // Side windows matter from the oblique camera, not only the street view.
    for (const side of [-1, 1]) {
      b.box(.026, .21, .18, M.ivory, [side * (w / 2 + .011), upperY, .02]);
      b.box(.03, .17, .14, M.glass, [side * (w / 2 + .024), upperY, .02]);
      b.box(.034, .012, .14, trim, [side * (w / 2 + .029), upperY, .02]);
    }
    const doorX = shop ? -w * .3 : 0;
    b.box(.19, .32, .032, M.ivory, [doorX, .17, front + .012]);
    b.box(.15, .285, .039, trim, [doorX, .158, front + .029]);
    b.box(.09, .11, .043, M.glass, [doorX, .23, front + .033]);
    b.sphere(.011, M.lemon, [doorX + .046, .14, front + .06]);
    b.box(.26, .032, .12, M.stone, [doorX, .018, front + .056]);
    if (shop) {
      b.box(w * .43, .235, .035, M.ivory, [w * .14, .19, front + .018]);
      b.box(w * .39, .195, .04, M.glass, [w * .14, .19, front + .036]);
      for (const offset of [-.10, .02, .13]) b.box(.014, .20, .047, trim, [w * .14 + offset, .19, front + .045]);
      const aw = w + .045;
      for (let stripe = 0; stripe < 7; stripe++) {
        const x = -aw / 2 + aw / 7 * (stripe + .5);
        b.box(aw / 7, .024, .24, stripe % 2 ? M.canvas : M.redCloth, [x, .43, front + .13], [.18, 0, 0]);
        b.box(aw / 7, .063, .015, stripe % 2 ? M.canvas : M.redCloth, [x, .37, front + .244]);
      }
      for (const x of [-aw / 2, aw / 2]) b.box(.014, .18, .014, M.metal, [x, .30, front + .23]);
      b.box(.28, .085, .15, M.wood, [w * .19, .07, front + .21]);
      for (let fruit = 0; fruit < 7; fruit++) b.sphere(.021, fruit % 2 ? M.lemon : M.flower, [w * .19 - .09 + fruit * .029, .123, front + .2]);
    } else if (index % 3 !== 0) {
      // A balcony with real thin uprights and plants, rather than a painted box.
      b.box(w * .7, .035, .20, M.stone, [0, upperY - .14, front + .07]);
      b.box(w * .7, .014, .014, M.metal, [0, upperY + .005, front + .17]);
      for (let rail = 0; rail < 9; rail++) b.box(.008, .15, .008, M.metal, [-w * .34 + w * .68 * rail / 8, upperY - .07, front + .17]);
      for (const x of [-w * .34, w * .34]) b.box(.009, .014, .18, M.metal, [x, upperY + .005, front + .08]);
      b.box(w * .38, .045, .085, M.roofEdge, [0, upperY - .075, front + .16]);
      for (let flower = 0; flower < 7; flower++) {
        const x = (flower - 3) * .043;
        b.sphere(.038, M.green, [x, upperY - .035, front + .16], [1, .6, 1]);
        b.sphere(.015, flower % 2 ? M.flower : M.purple, [x, upperY - .005, front + .17]);
      }
    }
    // Rain pipe, doorstep planter, and a few flowers give every house a scale.
    b.cylinder(.011, .011, h - .06, M.metal, [w / 2 - .045, h / 2 - .015, front + .036], [0, 0, 0], 6);
    b.cylinder(.055, .043, .09, M.roofEdge, [-w / 2 - .055, .05, front - .01], [0, 0, 0], 8);
    b.sphere(.073, M.green, [-w / 2 - .055, .13, front - .01], [1, .9, 1]);
    b.sphere(.026, M.flower, [-w / 2 - .035, .18, front + .012]);
    const group = b.finish(shop ? 'Harbour shop' : 'Seaside home');
    if (shop) {
      const labels = ['PORTO', 'PANE', 'FIORE', 'CAFFE', 'MARE'];
      const map = signTexture(labels[index % labels.length], index % 2 ? '#3a6e69' : '#785245');
      if (map) {
        const mat = material(0xffffff, { map, roughness: .9 });
        const geo = new THREE.PlaneGeometry(w * .58, .14); geometries.add(geo);
        const sign = new THREE.Mesh(geo, mat); sign.position.set(0, .535, front + .021); group.add(sign);
      }
    }
    return group;
  }

  const blocks = [
    [-3.05, -.95, .81, .80, .91], [-1.93, -.94, .92, .81, 1.06],
    [-.70, -1.00, .87, .84, .78], [.46, -.97, .89, .80, 1.13],
    [1.68, -1.02, .91, .79, .87], [2.86, -.92, .76, .73, .98],
    [-3.03, .97, .79, .73, .73], [-1.91, 1.02, .89, .80, 1.02],
    [-.68, .96, .90, .79, .88], [.55, 1.03, .91, .82, .76],
    [1.79, .96, .90, .77, 1.08], [2.93, .97, .73, .72, .78],
  ];
  blocks.forEach(([x, z, w, d, h], index) => {
    const yaw = z < 0 ? 0 : Math.PI;
    attach(house(index, w, d, h, index % 3 === 1), [x, .07, z], yaw, 'building', { footprint: [w, d] });
  });

  function lamp() {
    const b = batch();
    b.cylinder(.035, .055, .055, M.metal, [0, .027, 0]);
    b.cylinder(.012, .019, .50, M.metal, [0, .29, 0], [0, 0, 0], 8);
    b.box(.014, .014, .115, M.metal, [0, .548, .046]);
    b.cylinder(.049, .037, .02, M.metal, [0, .533, .095], [0, 0, 0], 8);
    b.cylinder(.029, .024, .075, M.lantern, [0, .487, .095], [0, 0, 0], 6);
    b.cylinder(.036, .025, .013, M.metal, [0, .446, .095], [0, 0, 0], 8);
    for (const x of [-.028, .028]) b.box(.006, .079, .006, M.metal, [x, .488, .095]);
    return b.finish('Warm harbour lantern');
  }
  for (const x of [-3.65, -2.52, -1.31, -.08, 1.15, 2.37, 3.57]) {
    for (const z of [-.43, .43]) attach(lamp(), [x, .065, z], z < 0 ? 0 : Math.PI, 'lamp');
  }
  function tree(index) {
    const b = batch();
    b.cylinder(.085, .072, .12, M.stone, [0, .06, 0]);
    b.cylinder(.018, .024, .26, M.wood, [0, .19, 0], [0, 0, -.1], 7);
    b.sphere(.14, M.green, [0, .40, 0], [.84, 1.25, .85]);
    b.sphere(.095, M.lightGreen, [-.045, .44, .025], [.8, 1.0, .8]);
    for (let i = 0; i < 5; i++) b.sphere(.018, M.lemon, [Math.sin(i * 2.4) * .10, .37 + (i % 3) * .055, Math.cos(i * 2.4) * .09]);
    return b.finish(index % 2 ? 'Potted lemon tree' : 'Quayside tree');
  }
  for (const [index, x] of [-2.50, -1.28, -.07, 1.16, 2.40].entries()) {
    for (const z of [-1.48, 1.48]) attach(tree(index), [x, .067, z], 0, 'tree');
  }
  function bench() {
    const b = batch();
    for (let i = 0; i < 3; i++) b.box(.26, .018, .026, M.wood, [0, .13, (i - 1) * .031]);
    for (let i = 0; i < 3; i++) b.box(.26, .024, .016, M.wood, [0, .17 + i * .03, -.055]);
    for (const x of [-.09, .09]) {
      b.box(.016, .13, .016, M.metal, [x, .065, -.036]);
      b.box(.016, .13, .016, M.metal, [x, .065, .036]);
      b.box(.013, .21, .013, M.metal, [x, .14, -.055], [-.1, 0, 0]);
    }
    return b.finish('Wooden promenade bench');
  }
  for (const x of [-2.52, -.06, 2.40]) for (const z of [-1.49, 1.49]) attach(bench(), [x + .14, .06, z], z < 0 ? Math.PI : 0, 'bench');

  // Individual rail sections stay rigid as their supporting waterfront bends.
  function quayRail() {
    const b = batch();
    for (const x of [-.18, .18]) {
      b.cylinder(.016, .022, .24, M.metal, [x, .12, 0], [0, 0, 0], 6);
      b.sphere(.024, M.metal, [x, .248, 0]);
    }
    b.box(.38, .015, .015, M.metal, [0, .21, 0]);
    b.box(.38, .01, .01, M.metal, [0, .10, 0]);
    return b.finish('Quayside iron railing');
  }
  for (let i = 0; i < 17; i++) for (const z of [-1.7, 1.7]) {
    const x = -3.4 + i * .425;
    attach(quayRail(), [x, .06, z * (1 - .2 * Math.pow(Math.abs(x) / 4, 8))], 0, 'railing');
  }
  function lighthouse() {
    const b = batch();
    b.cylinder(.19, .24, .08, M.stone, [0, .04, 0], [0, 0, 0], 12);
    b.cylinder(.11, .17, .74, M.ivory, [0, .44, 0], [0, 0, 0], 12);
    b.cylinder(.145, .155, .07, M.redCloth, [0, .29, 0], [0, 0, 0], 12);
    b.cylinder(.12, .13, .07, M.redCloth, [0, .61, 0], [0, 0, 0], 12);
    b.cylinder(.20, .18, .045, M.stone, [0, .83, 0], [0, 0, 0], 12);
    b.cylinder(.10, .10, .20, M.window, [0, .95, 0], [0, 0, 0], 8);
    for (let i = 0; i < 8; i++) {
      const a = i * Math.PI / 4;
      b.box(.012, .20, .012, M.metal, [Math.sin(a) * .105, .95, Math.cos(a) * .105]);
      b.cylinder(.006, .006, .10, M.metal, [Math.sin(a) * .18, .895, Math.cos(a) * .18], [0, 0, 0], 5);
    }
    b.cylinder(0, .16, .15, M.teal, [0, 1.115, 0], [0, 0, 0], 8);
    b.cylinder(.005, .005, .09, M.metal, [0, 1.21, 0], [0, 0, 0], 5);
    b.box(.075, .16, .022, M.teal, [0, .14, .164]);
    return b.finish('Little harbour lighthouse');
  }
  attach(lighthouse(), [3.67, .06, .0], 0, 'landmark');
  function fountain() {
    const b = batch();
    b.cylinder(.20, .23, .06, M.stone, [0, .03, 0], [0, 0, 0], 16);
    b.cylinder(.18, .18, .025, M.teal, [0, .073, 0], [0, 0, 0], 16);
    b.cylinder(.035, .054, .24, M.stone, [0, .17, 0], [0, 0, 0], 10);
    b.cylinder(.11, .045, .04, M.ivory, [0, .295, 0], [0, 0, 0], 12);
    b.sphere(.035, M.ivory, [0, .335, 0]);
    return b.finish('Town fountain');
  }
  attach(fountain(), [-3.67, .06, 0], 0, 'landmark');
  function bunting() {
    const b = batch();
    for (const z of [-.36, .36]) {
      b.cylinder(.009, .014, .62, M.wood, [0, .31, z], [0, 0, 0], 6);
      b.sphere(.018, M.lemon, [0, .64, z]);
    }
    b.cylinder(.004, .004, .72, M.darkWood, [0, .58, 0], [Math.PI / 2, 0, 0], 5);
    for (let i = 0; i < 7; i++) {
      const z = -.30 + i * .10, sag = .04 * Math.sin(i / 6 * Math.PI);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute([0,.585-sag,z-.035, 0,.585-sag,z+.035, .015,.49-sag,z], 3));
      geo.setAttribute('uv', new THREE.Float32BufferAttribute([0,1,1,1,.5,0], 2)); geo.computeVertexNormals();
      const mat = [M.redCloth, M.canvas, M.teal, M.lemon][i % 4]; mat.side = THREE.DoubleSide;
      b.add(geo, mat);
    }
    return b.finish('Festival pennants');
  }
  for (const x of [-2.53, -.08, 2.40]) attach(bunting(), [x, .07, 0], 0, 'bunting');

  function person(index) {
    const b = batch(), shirt = [M.teal, M.redCloth, M.shirt, M.purple, M.lemon][index % 5];
    b.add(new THREE.CapsuleGeometry(.035, .075, 3, 7), shirt, [0, .13, 0]);
    for (const side of [-1, 1]) {
      b.cylinder(.012, .015, .066, M.trousers, [side * .018, .040, 0], [0, 0, side * .045], 6);
      b.sphere(.017, M.darkWood, [side * .018, .010, .010], [.8, .5, 1.3]);
      b.cylinder(.012, .014, .071, shirt, [side * .041, .128, .003], [.10, 0, side * .17], 6);
      b.sphere(.011, M.skin, [side * .048, .092, .007]);
    }
    b.sphere(.038, M.skin, [0, .212, 0], [.86, 1, .88]);
    b.sphere(.036, M.hair, [0, .226, -.007], [.9, .66, .9]);
    b.sphere(.007, M.skin, [0, .212, .034], [.75, 1, 1]);
    if (index % 3 !== 1) {
      b.cylinder(.047, .048, .009, M.hat, [0, .243, 0], [0, 0, -.10], 10);
      b.cylinder(.028, .031, .029, M.hat, [0, .260, 0], [0, 0, -.10], 10);
      b.cylinder(.030, .031, .007, M.darkWood, [0, .250, 0], [0, 0, -.10], 10);
    }
    if (index % 4 === 1) {
      b.sphere(.024, M.darkWood, [0, .233, -.035], [1, 1, .8]);
      b.box(.034, .041, .020, M.lemon, [-.057, .075, 0], [0, 0, -.12]);
    }
    return b.finish('Harbour resident');
  }
  for (let i = 0; i < 8; i++) {
    const entry = attach(person(i), [-2.9 + i * .80, .074, i % 2 ? .14 : -.12], i % 2 ? Math.PI / 2 : -Math.PI / 2, 'resident', {
      name: ['Luca', 'Mara', 'Nico', 'Alba', 'Pip', 'Rosa', 'Emi', 'Leo'][i],
      phase: i * 1.73, walkSpeed: .07 + i % 3 * .01,
    });
    residents.push(entry);
  }
  root.updateMatrixWorld(true);
  return {
    root, attachments, surfaces, residents,
    dispose() {
      geometries.forEach(g => g.dispose());
      materials.forEach(m => m.dispose());
      textures.forEach(t => t.dispose());
      root.removeFromParent();
    },
  };
}

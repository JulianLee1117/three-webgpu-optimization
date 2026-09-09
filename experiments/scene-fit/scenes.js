// Fixed scene families for image-based fitting. Targets are declared here for
// rendering the reference only; fitting code should receive bounds and an oracle.
// All dimensions are world-space lengths. Cameras are supplied by the caller.

function createScene(THREE) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101827);
  const geometries = [];
  const materials = [];
  const meshes = [];
  const lights = [];
  let disposed = false;

  function geometry(value) {
    geometries.push(value);
    return value;
  }

  function material(color, roughness = 0.78, metalness = 0.06) {
    const value = new THREE.MeshStandardMaterial({ color, roughness, metalness });
    materials.push(value);
    return value;
  }

  function mesh(shape, surface) {
    const value = new THREE.Mesh(shape, surface);
    value.castShadow = false;
    value.receiveShadow = false;
    meshes.push(value);
    scene.add(value);
    return value;
  }

  const box = geometry(new THREE.BoxGeometry(1, 1, 1));
  const cylinder = geometry(new THREE.CylinderGeometry(1, 1, 1, 12));
  const sphere = geometry(new THREE.SphereGeometry(1, 8, 6));
  const floor = mesh(geometry(new THREE.PlaneGeometry(4, 4)), material(0x263247));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.015;

  const ambient = new THREE.HemisphereLight(0xd3e7ff, 0x657052, 2.1);
  const key = new THREE.DirectionalLight(0xffecd0, 3.0);
  key.position.set(3, 6, 4);
  key.castShadow = false;
  lights.push(ambient, key);
  scene.add(ambient, key);

  function complete(update, initialValues) {
    update(initialValues);
    return {
      scene,
      update,
      dispose() {
        if (disposed) return;
        disposed = true;
        for (const value of geometries) value.dispose();
        for (const value of materials) value.dispose();
        for (const value of lights) value.dispose?.();
        scene.clear();
        meshes.length = 0;
      },
    };
  }

  return { scene, box, cylinder, sphere, geometry, material, mesh, complete };
}

function buildRover(THREE) {
  const kit = createScene(THREE);
  const bodyPaint = kit.material(0xeaa83e);
  const cabinPaint = kit.material(0xe9edf0);
  const tirePaint = kit.material(0x172230, 0.92, 0);
  const hubPaint = kit.material(0x6998a4, 0.5, 0.25);
  const glassPaint = kit.material(0x246479, 0.34, 0.1);
  const lampPaint = kit.material(0xfff0b5);

  const body = kit.mesh(kit.box, bodyPaint);
  const cabin = kit.mesh(kit.box, cabinPaint);
  const windscreen = kit.mesh(kit.box, glassPaint);
  const frontBumper = kit.mesh(kit.box, tirePaint);
  const backBumper = kit.mesh(kit.box, tirePaint);
  const leftLamp = kit.mesh(kit.box, lampPaint);
  const rightLamp = kit.mesh(kit.box, lampPaint);
  const wheels = [];
  const hubs = [];
  for (let i = 0; i < 4; i++) {
    const wheel = kit.mesh(kit.cylinder, tirePaint);
    const hub = kit.mesh(kit.cylinder, hubPaint);
    wheel.rotation.z = Math.PI / 2;
    hub.rotation.z = Math.PI / 2;
    wheels.push(wheel);
    hubs.push(hub);
  }

  function update(values) {
    const width = values[0];
    const length = values[1];
    const bodyHeight = values[2];
    const radius = values[3];
    const tireWidth = values[4];
    const axleSpacing = values[5];
    const cabinWidth = values[6];
    const cabinHeight = values[7];
    const bodyY = radius + bodyHeight * 0.5;
    const cabinY = radius + bodyHeight + cabinHeight * 0.5;
    const cabinLength = length * 0.43;
    const cabinZ = -length * 0.12;

    body.position.set(0, bodyY, 0);
    body.scale.set(width, bodyHeight, length);
    cabin.position.set(0, cabinY, cabinZ);
    cabin.scale.set(cabinWidth, cabinHeight, cabinLength);
    windscreen.position.set(0, cabinY + cabinHeight * 0.04, cabinZ + cabinLength * 0.5 + 0.008);
    windscreen.scale.set(cabinWidth * 0.78, cabinHeight * 0.60, 0.02);
    frontBumper.position.set(0, bodyY - bodyHeight * 0.18, length * 0.5 + 0.055);
    frontBumper.scale.set(width * 0.9, 0.11, 0.13);
    backBumper.position.set(0, bodyY - bodyHeight * 0.18, -length * 0.5 - 0.055);
    backBumper.scale.set(width * 0.9, 0.11, 0.13);
    leftLamp.position.set(-width * 0.31, bodyY + bodyHeight * 0.18, length * 0.5 + 0.012);
    leftLamp.scale.set(0.15, 0.10, 0.035);
    rightLamp.position.set(width * 0.31, bodyY + bodyHeight * 0.18, length * 0.5 + 0.012);
    rightLamp.scale.set(0.15, 0.10, 0.035);

    for (let i = 0; i < 4; i++) {
      const side = i < 2 ? -1 : 1;
      const end = i % 2 === 0 ? -1 : 1;
      const wheelX = side * (width * 0.5 + 0.035 + tireWidth * 0.5);
      const wheelZ = end * axleSpacing * 0.5;
      wheels[i].position.set(wheelX, radius, wheelZ);
      wheels[i].scale.set(radius, tireWidth, radius);
      hubs[i].position.set(wheelX, radius, wheelZ);
      hubs[i].scale.set(radius * 0.43, tireWidth + 0.025, radius * 0.43);
    }
  }

  return kit.complete(update, [1.10, 1.70, 0.40, 0.28, 0.22, 1.25, 0.70, 0.52]);
}

function buildRobot(THREE) {
  const kit = createScene(THREE);
  const shellPaint = kit.material(0x55a8a4);
  const headPaint = kit.material(0xf2c967);
  const jointPaint = kit.material(0x34465f);
  const darkPaint = kit.material(0x162439);
  const accentPaint = kit.material(0xf27558);

  const torso = kit.mesh(kit.box, shellPaint);
  const neck = kit.mesh(kit.cylinder, jointPaint);
  const head = kit.mesh(kit.box, headPaint);
  const panel = kit.mesh(kit.box, jointPaint);
  const mouth = kit.mesh(kit.box, darkPaint);
  const antenna = kit.mesh(kit.cylinder, jointPaint);
  const antennaTip = kit.mesh(kit.sphere, accentPaint);
  const legs = [];
  const feet = [];
  const arms = [];
  const hands = [];
  const eyes = [];
  for (let i = 0; i < 2; i++) {
    legs.push(kit.mesh(kit.box, jointPaint));
    feet.push(kit.mesh(kit.box, accentPaint));
    arms.push(kit.mesh(kit.box, shellPaint));
    hands.push(kit.mesh(kit.sphere, headPaint));
    eyes.push(kit.mesh(kit.box, darkPaint));
  }

  function update(values) {
    const shoulders = values[0];
    const torsoHeight = values[1];
    const headWidth = values[2];
    const headHeight = values[3];
    const legLength = values[4];
    const legSpacing = values[5];
    const armLength = values[6];
    const antennaHeight = values[7];
    const torsoBottom = 0.10 + legLength;
    const torsoTop = torsoBottom + torsoHeight;
    const headBottom = torsoTop + 0.08;
    const headY = headBottom + headHeight * 0.5;
    const armTop = torsoTop + 0.02;

    torso.position.set(0, torsoBottom + torsoHeight * 0.5, 0);
    torso.scale.set(shoulders, torsoHeight, 0.42);
    neck.position.set(0, torsoTop + 0.04, 0);
    neck.scale.set(0.105, 0.09, 0.105);
    head.position.set(0, headY, 0);
    head.scale.set(headWidth, headHeight, 0.43);
    panel.position.set(0, torsoBottom + torsoHeight * 0.51, 0.219);
    panel.scale.set(shoulders * 0.40, torsoHeight * 0.46, 0.026);
    mouth.position.set(0, headY - headHeight * 0.22, 0.222);
    mouth.scale.set(headWidth * 0.32, 0.027, 0.025);
    antenna.position.set(0, headBottom + headHeight + antennaHeight * 0.5, 0);
    antenna.scale.set(0.025, antennaHeight, 0.025);
    antennaTip.position.set(0, headBottom + headHeight + antennaHeight, 0);
    antennaTip.scale.set(0.065, 0.065, 0.065);

    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;
      legs[i].position.set(side * legSpacing * 0.5, 0.10 + legLength * 0.5, 0);
      legs[i].scale.set(0.17, legLength, 0.21);
      feet[i].position.set(side * legSpacing * 0.5, 0.05, 0.07);
      feet[i].scale.set(0.26, 0.10, 0.37);
      arms[i].position.set(side * (shoulders * 0.5 + 0.15), armTop - armLength * 0.5, 0);
      arms[i].scale.set(0.19, armLength, 0.23);
      hands[i].position.set(side * (shoulders * 0.5 + 0.15), armTop - armLength - 0.04, 0);
      hands[i].scale.set(0.12, 0.12, 0.12);
      eyes[i].position.set(side * headWidth * 0.23, headY + headHeight * 0.10, 0.225);
      eyes[i].scale.set(headWidth * 0.15, headHeight * 0.23, 0.035);
    }
  }

  return kit.complete(update, [0.90, 0.48, 0.60, 0.33, 0.42, 0.46, 0.46, 0.14]);
}

function createRoofGeometry(THREE) {
  // Closed triangular prism: unit width/depth, with base y=0 and ridge y=1.
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -0.5, 0, 0.5, 0.5, 0, 0.5, 0, 1, 0.5,
    0.5, 0, -0.5, -0.5, 0, -0.5, 0, 1, -0.5,
    -0.5, 0, -0.5, -0.5, 0, 0.5, 0, 1, 0.5,
    -0.5, 0, -0.5, 0, 1, 0.5, 0, 1, -0.5,
    0.5, 0, 0.5, 0.5, 0, -0.5, 0, 1, -0.5,
    0.5, 0, 0.5, 0, 1, -0.5, 0, 1, 0.5,
    -0.5, 0, -0.5, 0.5, 0, -0.5, 0.5, 0, 0.5,
    -0.5, 0, -0.5, 0.5, 0, 0.5, -0.5, 0, 0.5,
  ], 3));
  geometry.computeVertexNormals();
  return geometry;
}

function buildPavilion(THREE) {
  const kit = createScene(THREE);
  const stonePaint = kit.material(0xe5d4ac, 0.92, 0);
  const trimPaint = kit.material(0x779e9d);
  const roofPaint = kit.material(0xb75042, 0.86, 0);
  const platform = kit.mesh(kit.box, stonePaint);
  const roof = kit.mesh(kit.geometry(createRoofGeometry(THREE)), roofPaint);
  const pillars = [];
  const bases = [];
  const capitals = [];
  const beams = [];
  for (let i = 0; i < 4; i++) {
    pillars.push(kit.mesh(kit.box, stonePaint));
    bases.push(kit.mesh(kit.box, trimPaint));
    capitals.push(kit.mesh(kit.box, trimPaint));
    beams.push(kit.mesh(kit.box, trimPaint));
  }

  function update(values) {
    const openingWidth = values[0];
    const pillarHeight = values[1];
    const pillarWidth = values[2];
    const depth = values[3];
    const roofHeight = values[4];
    const overhang = values[5];
    const foundationMargin = values[6];
    const foundationHeight = values[7];
    const width = openingWidth + 2 * pillarWidth;
    const pillarX = (openingWidth + pillarWidth) * 0.5;
    const pillarZ = (depth - pillarWidth) * 0.5;
    const beamY = foundationHeight + pillarHeight + 0.06;
    const roofY = foundationHeight + pillarHeight + 0.12;

    platform.position.set(0, foundationHeight * 0.5, 0);
    platform.scale.set(width + 2 * foundationMargin, foundationHeight, depth + 2 * foundationMargin);
    roof.position.set(0, roofY, 0);
    roof.scale.set(width + 2 * overhang, roofHeight, depth + 2 * overhang);

    for (let i = 0; i < 4; i++) {
      const side = i < 2 ? -1 : 1;
      const end = i % 2 === 0 ? -1 : 1;
      pillars[i].position.set(side * pillarX, foundationHeight + pillarHeight * 0.5, end * pillarZ);
      pillars[i].scale.set(pillarWidth, pillarHeight, pillarWidth);
      bases[i].position.set(side * pillarX, foundationHeight + 0.035, end * pillarZ);
      bases[i].scale.set(pillarWidth * 1.25, 0.07, pillarWidth * 1.25);
      capitals[i].position.set(side * pillarX, foundationHeight + pillarHeight - 0.04, end * pillarZ);
      capitals[i].scale.set(pillarWidth * 1.25, 0.08, pillarWidth * 1.25);
    }
    beams[0].position.set(0, beamY, -pillarZ);
    beams[0].scale.set(width, 0.12, pillarWidth);
    beams[1].position.set(0, beamY, pillarZ);
    beams[1].scale.set(width, 0.12, pillarWidth);
    beams[2].position.set(-pillarX, beamY, 0);
    beams[2].scale.set(pillarWidth, 0.12, depth);
    beams[3].position.set(pillarX, beamY, 0);
    beams[3].scale.set(pillarWidth, 0.12, depth);
  }

  return kit.complete(update, [1.50, 0.95, 0.23, 1.05, 0.34, 0.15, 0.15, 0.16]);
}

export const SCENE_SPECS = [
  {
    id: 'rover',
    title: 'Toy rover',
    parameters: [
      { name: 'Body width', min: 0.78, max: 1.55 },
      { name: 'Body length', min: 1.30, max: 2.30 },
      { name: 'Body height', min: 0.24, max: 0.60 },
      { name: 'Wheel radius', min: 0.18, max: 0.37 },
      { name: 'Wheel thickness', min: 0.13, max: 0.31 },
      { name: 'Axle spacing', min: 0.82, max: 1.60 },
      { name: 'Cabin width', min: 0.43, max: 0.96 },
      { name: 'Cabin height', min: 0.30, max: 0.76 },
    ],
    target: [1.32, 1.92, 0.34, 0.32, 0.20, 1.42, 0.79, 0.63],
    build: buildRover,
  },
  {
    id: 'robot',
    title: 'Geometric robot',
    parameters: [
      { name: 'Shoulder width', min: 0.64, max: 1.18 },
      { name: 'Torso height', min: 0.34, max: 0.62 },
      { name: 'Head width', min: 0.43, max: 0.79 },
      { name: 'Head height', min: 0.24, max: 0.42 },
      { name: 'Leg length', min: 0.26, max: 0.55 },
      { name: 'Leg spacing', min: 0.28, max: 0.64 },
      { name: 'Arm length', min: 0.30, max: 0.55 },
      { name: 'Antenna height', min: 0.06, max: 0.22 },
    ],
    target: [1.01, 0.55, 0.68, 0.37, 0.46, 0.52, 0.49, 0.17],
    build: buildRobot,
  },
  {
    id: 'pavilion',
    title: 'Garden pavilion',
    parameters: [
      { name: 'Opening width', min: 1.00, max: 2.00 },
      { name: 'Pillar height', min: 0.72, max: 1.15 },
      { name: 'Pillar width', min: 0.14, max: 0.32 },
      { name: 'Pavilion depth', min: 0.68, max: 1.45 },
      { name: 'Roof height', min: 0.18, max: 0.50 },
      { name: 'Roof overhang', min: 0.05, max: 0.26 },
      { name: 'Foundation margin', min: 0.05, max: 0.26 },
      { name: 'Foundation height', min: 0.08, max: 0.24 },
    ],
    target: [1.64, 1.04, 0.20, 1.18, 0.41, 0.20, 0.13, 0.14],
    build: buildPavilion,
  },
];

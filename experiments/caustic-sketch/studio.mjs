import * as THREE from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import {RectAreaLightTexturesLib} from 'three/addons/lights/RectAreaLightTexturesLib.js';

function walnutTexture(){
  const n=256,data=new Uint8Array(n*n*4);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const u=x/n,v=y/n;
    const wave=Math.sin(v*81+Math.sin(u*6)*2.1+Math.sin(u*17)*.28);
    const fine=Math.sin(v*179+Math.sin(u*11)*3)*.06;
    const grain=.52+.12*wave+fine;
    const i=(y*n+x)*4;data[i]=51+grain*24;data[i+1]=34+grain*18;data[i+2]=23+grain*14;data[i+3]=255;
  }
  const texture=new THREE.DataTexture(data,n,n,THREE.RGBAFormat);texture.colorSpace=THREE.SRGBColorSpace;
  texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(1.8,2.8);
  texture.magFilter=THREE.LinearFilter;texture.minFilter=THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps=true;texture.needsUpdate=true;return texture;
}

/** A compact optics bench. The analytical lens, photons, and irradiance are owned by main. */
export function createStudio(scene){
  // WebGPURenderer uses the node light library, not WebGL UniformsLib.
  const ltc=RectAreaLightTexturesLib.LTC_HALF_1?RectAreaLightTexturesLib:RectAreaLightTexturesLib.init();
  THREE.RectAreaLightNode?.setLTC(ltc);
  const root=new THREE.Group();root.name='Sunprint optics studio';scene.add(root);
  const geometries=new Set(),materials=new Set(),textures=new Set(),lights=[],instanceBatches=[];
  const ownG=g=>(geometries.add(g),g),ownM=m=>(materials.add(m),m);
  const mat=(color,extras={})=>ownM(new THREE.MeshStandardMaterial({color,roughness:.6,...extras}));
  const brass=mat(0xae9464,{metalness:.72,roughness:.46});
  const agedBrass=mat(0x705f40,{metalness:.62,roughness:.56});
  const blackMetal=mat(0x14272c,{metalness:.48,roughness:.49});
  const teal=mat(0x1d4145,{metalness:.22,roughness:.57});
  const rubber=mat(0x101719,{roughness:.88});
  const stone=mat(0x27353a,{roughness:.68,metalness:.025});
  const luminous=mat(0xffe4b6,{emissive:0xffd493,emissiveIntensity:.55});
  const tickMaterial=mat(0xd6c09a,{metalness:.26,roughness:.61});
  const grain=walnutTexture();textures.add(grain);
  const walnut=mat(0xffffff,{map:grain,roughness:.71,metalness:0});
  const rounded=ownG(new RoundedBoxGeometry(1,1,1,2,.035));
  const box=ownG(new THREE.BoxGeometry(1,1,1));
  const cylinder=ownG(new THREE.CylinderGeometry(1,1,1,24));
  const torus=ownG(new THREE.TorusGeometry(1,.022,6,48));
  function mesh(g,m,p,s=[1,1,1],parent=root){const o=new THREE.Mesh(g,m);o.position.set(...p);o.scale.set(...s);o.castShadow=o.receiveShadow=true;parent.add(o);return o;}
  const slab=(m,p,s,parent=root)=>mesh(rounded,m,p,s,parent);
  const axisY=new THREE.Vector3(0,1,0);
  function rod(a,b,r,m){const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),d=bv.clone().sub(av);const o=mesh(cylinder,m,av.clone().add(bv).multiplyScalar(.5).toArray(),[r,d.length(),r]);o.quaternion.setFromUnitVectors(axisY,d.normalize());return o;}
  function light(l){lights.push(l);root.add(l);return l;}

  // A single continuous base creates contact and a clear visual axis.
  slab(walnut,[0,-.57,-3.3],[11,.14,19.4]);
  slab(blackMetal,[0,-.655,-3.3],[11.08,.035,19.48]);
  for(const x of [-4.1,-2.7,-1.3,.1,1.5,2.9,4.3])mesh(box,rubber,[x,-.496,-3.3],[.009,.004,19.3]);
  const bench=slab(stone,[0,-.43,-3.0],[3.35,.12,13.25]);
  bench.name='Optical rail base';
  for(const x of [-1.48,1.48]){
    slab(agedBrass,[x,-.362,-3.0],[.045,.021,13.1]);
    for(let z=-9;z<=3;z+=.5)mesh(box,tickMaterial,[x+(x<0?.08:-.08),-.349,z],[.07,.004,.012]);
  }
  for(const x of [-1.27,1.27])for(let z=-9;z<=3;z+=1){
    const screw=mesh(cylinder,blackMetal,[x,-.36,z],[.023,.006,.023]);
    mesh(box,agedBrass,[x,-.355,z],[.028,.004,.006]);
  }

  const lensMount=new THREE.Group();lensMount.name='Freeform glass holder';root.add(lensMount);
  for(const x of [-1.14,1.14]){
    slab(rubber,[x,-.34,.06],[.42,.055,.87],lensMount);
    slab(blackMetal,[x,-.29,.06],[.34,.06,.75],lensMount);
    mesh(cylinder,brass,[x,.06,.06],[.037,.69,.037],lensMount);
    mesh(cylinder,blackMetal,[x,-.20,.06],[.061,.16,.061],lensMount);
    // Two slender rails hold the glass instead of hiding it in an opaque box.
    slab(brass,[x,1.60,.578],[.049,2.40,.043],lensMount);
    slab(blackMetal,[x,1.60,-.37],[.043,2.40,.043],lensMount);
  }
  for(const y of [.42,2.78]){
    slab(brass,[0,y,.578],[2.33,.049,.043],lensMount);
    slab(blackMetal,[0,y,-.37],[2.33,.043,.043],lensMount);
  }
  for(const x of [-1.14,1.14])for(const y of [.42,2.78]){
    slab(agedBrass,[x,y,.104],[.044,.044,.948],lensMount);
    const knob=mesh(cylinder,blackMetal,[x,y,.631],[.053,.064,.053],lensMount);knob.rotation.x=Math.PI/2;
    const cap=mesh(cylinder,brass,[x,y,.670],[.036,.012,.036],lensMount);cap.rotation.x=Math.PI/2;
    const slot=mesh(box,rubber,[x,y,.680],[.039,.007,.006],lensMount);slot.rotation.z=Math.PI*.23;
    // Small contact pads remain wholly outside the optical aperture.
    slab(rubber,[x,y<1?.52:2.68,.24],[.115,.08,.20],lensMount);
  }
  // Small ticks on the outside of the holder are spatial cues, not evidence.
  for(let i=0;i<=10;i++){
    const x=-1+i*.2;mesh(box,tickMaterial,[x,2.78,.602],[.007,i%5===0?.030:.016,.003],lensMount);
    const y=.6+i*.2;mesh(box,tickMaterial,[-1.14,y,.602],[i%5===0?.030:.016,.007,.003],lensMount);
  }
  slab(agedBrass,[0,.11,.36],[.62,.14,.013],lensMount);
  slab(blackMetal,[0,.11,.37],[.56,.1,.007],lensMount);

  // The source frame remains outside the two-unit aperture. Main supplies
  // parallel analytical rays; this housing does not imply point-light optics.
  const source=new THREE.Group();source.name='Collimated source aperture';root.add(source);
  for(const x of [-1.13,1.13]){
    slab(blackMetal,[x,1.6,4.10],[.09,2.35,.16],source);
    slab(luminous,[x,1.6,4.005],[.025,2.19,.012],source);
    slab(blackMetal,[x,.055,4.1],[.055,.76,.10],source);
    slab(rubber,[x,-.335,4.1],[.35,.06,.59],source);
  }
  for(const y of [.44,2.76]){
    slab(blackMetal,[0,y,4.1],[2.35,.095,.16],source);
    slab(luminous,[0,y,4.005],[2.19,.025,.012],source);
  }
  for(const x of [-.86,-.43,0,.43,.86]){
    slab(agedBrass,[x,2.80,4.1],[.20,.015,.11],source);
  }
  // Power cable follows the bench edge and never crosses the optical path.
  const cable=new THREE.CatmullRomCurve3([new THREE.Vector3(1.15,-.26,4.12),new THREE.Vector3(1.70,-.40,3.85),new THREE.Vector3(1.79,-.43,2.25),new THREE.Vector3(2.5,-.49,1.2)]);
  mesh(ownG(new THREE.TubeGeometry(cable,32,.025,6,false)),rubber,[0,0,0]);
  slab(blackMetal,[2.48,-.39,1.06],[.64,.20,.43]);
  mesh(cylinder,brass,[2.60,-.26,1.06],[.062,.056,.062]);
  mesh(cylinder,luminous,[2.28,-.282,1.10],[.018,.007,.018]);

  const receiverAssembly=new THREE.Group();receiverAssembly.name='Movable photon receiver';root.add(receiverAssembly);
  const receiverStart=root.children.length;
  const receiverBacking=slab(blackMetal,[0,1.68,-10.09],[4.43,4.34,.15]);
  receiverBacking.name='Photon receiver backing';
  const receiverMaterial=ownM(new THREE.MeshBasicMaterial({color:0x0e171b}));
  const receiverMesh=mesh(ownG(new THREE.PlaneGeometry(4.2,4.2)),receiverMaterial,[0,1.6,-10]);
  receiverMesh.name='Photon irradiance receiver';receiverMesh.castShadow=false;receiverMesh.receiveShadow=false;
  for(const x of [-2.15,2.15]){
    slab(brass,[x,1.685,-9.985],[.073,4.35,.082]);
    slab(teal,[x,1.685,-10.095],[.18,4.38,.13]);
  }
  slab(brass,[0,3.81,-9.985],[4.36,.073,.082]);
  slab(brass,[0,-.455,-9.985],[4.36,.055,.082]);
  for(const x of [-1.94,1.94])for(const y of [-.29,3.57]){
    const screw=mesh(cylinder,agedBrass,[x,y,-9.985],[.025,.02,.025]);screw.rotation.x=Math.PI/2;
  }
  for(const x of [-1.75,1.75]){
    slab(rubber,[x,-.44,-10.04],[.63,.08,.9]);
    rod([x,-.39,-10.37],[x,1.2,-10.14],.027,blackMetal);
  }
  // The shallow top hood shades the screen; it never overlaps its image plane.
  slab(blackMetal,[0,3.91,-10.035],[4.52,.06,.62]);
  slab(agedBrass,[0,3.877,-9.71],[4.41,.014,.029]);
  // Identity parent preserves all default coordinates. Moving this group by
  // z=10-distance keeps the receiver, backing, frame, feet, and hood together.
  for(const object of root.children.slice(receiverStart))receiverAssembly.add(object);

  light(new THREE.HemisphereLight(0xaac8d2,0x172e31,.56));
  const key=light(new THREE.SpotLight(0xffd4a3,34,18,.63,.95,2));
  key.position.set(-3.8,6.4,4.7);key.target.position.set(0,1.3,0);root.add(key.target);key.castShadow=true;
  key.shadow.mapSize.set(1024,1024);key.shadow.bias=-.00025;key.shadow.normalBias=.023;
  const edge=light(new THREE.RectAreaLight(0x9dd6dc,1.8,2.3,3.6));edge.position.set(3.8,3,-1.0);edge.lookAt(0,1.6,0);
  const softbox=light(new THREE.RectAreaLight(0xffe1b7,1.65,3.0,1.7));softbox.position.set(-1.8,4.0,1.5);softbox.lookAt(0,1.2,0);
  const screenAccent=light(new THREE.RectAreaLight(0xb9d6d3,1.2,4.5,.2));screenAccent.position.set(0,4.15,-8.8);screenAccent.lookAt(0,1.7,-10);
  function batchRepeated(parent){
    for(const child of [...parent.children])if(child.isGroup)batchRepeated(child);
    const sets=new Map();
    for(const child of parent.children){
      if(!child.isMesh||child.name||Array.isArray(child.material))continue;
      const key=`${child.geometry.uuid}:${child.material.uuid}:${child.castShadow}:${child.receiveShadow}`;
      if(!sets.has(key))sets.set(key,[]);sets.get(key).push(child);
    }
    for(const repeated of sets.values())if(repeated.length>=3){
      const first=repeated[0],batch=new THREE.InstancedMesh(first.geometry,first.material,repeated.length);
      batch.name='Repeated apparatus details';batch.castShadow=first.castShadow;batch.receiveShadow=first.receiveShadow;
      repeated.forEach((part,i)=>{part.updateMatrix();batch.setMatrixAt(i,part.matrix);parent.remove(part);});
      batch.instanceMatrix.needsUpdate=true;batch.computeBoundingSphere();parent.add(batch);instanceBatches.push(batch);
    }
  }
  batchRepeated(root);
  let disposed=false;
  return {root,lensMount,receiverAssembly,receiverMesh,screen:receiverMesh,source,lights,
    update(_time){/* Art is stationary; main owns all measured optical behavior. */},
    dispose(){if(disposed)return;disposed=true;root.removeFromParent();for(const l of lights)l.shadow?.dispose();for(const b of instanceBatches)b.dispose();for(const g of geometries)g.dispose();for(const m of materials)m.dispose();for(const t of textures)t.dispose();}
  };
}

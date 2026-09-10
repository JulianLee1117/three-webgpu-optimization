import * as THREE from 'three';

// Art direction is procedural. These visual surfaces are not a fluid simulation.
const SEA_Y = -.6;
const TAU = Math.PI * 2;
function random(seed = 8) { return () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; }; }

function roofGeometry() {
  const g = new THREE.BufferGeometry();
  const a = [-.5,0,-.5, .5,0,-.5, 0,.42,-.5, -.5,0,.5, 0,.42,.5, .5,0,.5,
    -.5,0,-.5,0,.42,-.5,0,.42,.5, -.5,0,-.5,0,.42,.5,-.5,0,.5,
    .5,0,-.5,.5,0,.5,0,.42,.5, .5,0,-.5,0,.42,.5,0,.42,-.5];
  g.setAttribute('position',new THREE.Float32BufferAttribute(a,3)); g.computeVertexNormals(); return g;
}

function stratumGeometry(radius, height, seed, baseColor) {
  const rng=random(seed), rings=14, segments=17, points=[], colors=[], indices=[];
  const phases=Array.from({length:segments},()=>.82+rng()*.28);
  const color=new THREE.Color(baseColor);
  for(let j=0;j<=rings;j++) {
    const y=height*j/rings, taper=(1-.77*Math.pow(j/rings,1.35))*(j%2?.94:1.025);
    for(let i=0;i<segments;i++) {
      const a=i/segments*TAU, r=radius*taper*phases[i]*(.94+.12*rng());
      points.push(Math.cos(a)*r,y+(j===0?0:(rng()-.5)*height*.012),Math.sin(a)*r);
      const c=color.clone().multiplyScalar(.65+.16*rng()+(j%2)*.24);
      if (j>rings*.72 && i%4<2) c.lerp(new THREE.Color(0x5b7450),.45);
      colors.push(c.r,c.g,c.b);
    }
  }
  for(let j=0;j<rings;j++) for(let i=0;i<segments;i++) {
    const a=j*segments+i,b=j*segments+(i+1)%segments,c=(j+1)*segments+i,d=(j+1)*segments+(i+1)%segments;
    indices.push(a,c,b,b,c,d);
  }
  points.push(0,height,0); colors.push(color.r,color.g,color.b);
  for(let i=0;i<segments;i++) indices.push(rings*segments+i,points.length/3-1,rings*segments+(i+1)%segments);
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
  geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  geometry.setIndex(indices); geometry.computeVertexNormals(); return geometry;
}

const oceanVertex = `
uniform float uTime;
varying vec3 vWorld;
varying vec3 vNormalWorld;
float heightAt(vec2 p) {
  return .047*sin(p.x*1.15+p.y*.43+uTime*.8)+.027*sin(p.y*1.94-p.x*.65-uTime*.92)
       +.017*sin(p.x*3.2+p.y*1.7+uTime*1.3);
}
void main(){
  vec4 w=modelMatrix*vec4(position,1.0);
  float h=heightAt(w.xz);
  w.y+=h;
  float d=.065;
  vNormalWorld=normalize(vec3((h-heightAt(w.xz+vec2(d,0.0)))/d,1.0,(h-heightAt(w.xz+vec2(0.0,d)))/d));
  vWorld=w.xyz;
  gl_Position=projectionMatrix*viewMatrix*w;
}`;
const oceanFragment = `
uniform float uTime;
uniform vec3 uSun;
varying vec3 vWorld;
varying vec3 vNormalWorld;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1)),f.x),f.y);}
void main(){
  vec2 p=vWorld.xz;
  vec3 eye=normalize(cameraPosition-vWorld);
  float small=sin(p.x*9.3+p.y*4.2+uTime*1.5)*cos(p.y*12.0-uTime*1.6);
  // Evaluate the normal per fragment. Interpolating a fine wave from the large
  // ocean grid creates a visible triangular sparkle pattern at distance.
  float a=cos(p.x*1.15+p.y*.43+uTime*.8), b=cos(p.y*1.94-p.x*.65-uTime*.92), c=cos(p.x*3.2+p.y*1.7+uTime*1.3);
  float dx=.047*1.15*a-.027*.65*b+.017*3.2*c;
  float dz=.047*.43*a+.027*1.94*b+.017*1.7*c;
  vec3 n=normalize(vec3(-dx+small*.012,1.0,-dz+sin(p.x*12.3-p.y*7.1+uTime)*.01));
  float fres=pow(1.0-max(0.0,dot(eye,n)),3.5);
  float drift=noise(p*.35+vec2(uTime*.018,0.0));
  vec3 water=mix(vec3(.008,.065,.08),vec3(.018,.16,.19),drift);
  water=mix(water,vec3(.39,.25,.35),fres*.42);
  vec3 halfway=normalize(eye+uSun);
  float glint=pow(max(dot(n,halfway),0.0),180.0);
  float silk=pow(max(0.0,sin(p.x*4.1+p.y*1.9+sin(p.y*.8-uTime*.3))),30.0);
  water+=vec3(.12,.22,.18)*silk*.055;
  water+=vec3(1.0,.57,.30)*glint*.34;
  float fog=1.0-exp(-.003*length(cameraPosition-vWorld));
  water=mix(water,vec3(.28,.27,.38),fog*.55);
  gl_FragColor=vec4(water,1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

/** A small navigable archipelago. Bounds are supplied separately for game logic. */
export function createWorld(scene) {
  const root=new THREE.Group(); root.name='Sunset archipelago'; scene.add(root);
  const geometries=new Set(),materials=new Set(), animated=[], rng=random(723);
  const ownG=g=>(geometries.add(g),g), ownM=m=>(materials.add(m),m);
  const mat=(color, extra={})=>ownM(new THREE.MeshStandardMaterial({color,roughness:.86,...extra}));
  const rock=mat(0xadbeb6,{vertexColors:true,flatShading:true});
  const stone=mat(0x828c85), stoneDark=mat(0x495e5b), trim=mat(0xb7ab91), terracotta=mat(0xb85337);
  const plaster=mat(0xeee1c2), wood=mat(0x654c37), brass=mat(0xa68043,{metalness:.42,roughness:.46});
  const leaf=mat(0x3d6e4e), leafDark=mat(0x254e45), bloom=mat(0xe091ad), glow=mat(0xffd696,{emissive:0xffa545,emissiveIntensity:1.15});
  const teal=mat(0x2c8a93), flag=mat(0xb9473e,{side:THREE.DoubleSide});
  const cube=ownG(new THREE.BoxGeometry(1,1,1)), sphere=ownG(new THREE.IcosahedronGeometry(1,1));
  const cylinder=ownG(new THREE.CylinderGeometry(1,1,1,9)), cone=ownG(new THREE.ConeGeometry(1,1,9)), roof=ownG(roofGeometry());
  const batches=new Map();
  function prop(geometry,material,position,scale=[1,1,1],rotation=0) {
    let byMaterial=batches.get(geometry); if(!byMaterial) batches.set(geometry,byMaterial=new Map());
    let list=byMaterial.get(material); if(!list)byMaterial.set(material,list=[]);
    const dummy=new THREE.Object3D(); dummy.position.set(...position); dummy.scale.set(...scale); dummy.rotation.y=rotation; dummy.updateMatrix();list.push(dummy.matrix.clone());
  }
  function mesh(geometry,material,position,scale=[1,1,1],parent=root) {
    const m=new THREE.Mesh(geometry,material);m.position.set(...position);m.scale.set(...scale);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;
  }
  function lineBetween(a,b,material,r=.026) {
    const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),d=bv.clone().sub(av);
    const m=mesh(cylinder,material,av.clone().add(bv).multiplyScalar(.5).toArray(),[r,d.length(),r]);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize()); return m;
  }
  function tree(x,y,z,s=1) {
    prop(cylinder,wood,[x,y+.55*s,z],[.085*s,1.1*s,.085*s]);
    prop(cone,leafDark,[x,y+1.2*s,z],[.36*s,1.75*s,.36*s]);
    prop(cone,leaf,[x+.06*s,y+1.68*s,z],[.23*s,1.2*s,.24*s]);
  }
  function shrub(x,y,z,s=.35) {
    prop(sphere,leaf,[x,y+s*.5,z],[s,s*.7,s]);
    for(let i=0;i<3;i++)prop(sphere,bloom,[x+(rng()-.5)*s,y+s*.75,z+(rng()-.5)*s],[s*.13,s*.16,s*.13]);
  }
  function cottage(x,y,z,s=1) {
    prop(cube,plaster,[x,y+.55*s,z],[1.12*s,1.1*s,.9*s]);
    prop(roof,terracotta,[x,y+1.1*s,z],[1.32*s,1.3*s,1.12*s]);
    prop(cube,wood,[x,y+.31*s,z+.459*s],[.22*s,.62*s,.03*s]);
    for(const side of [-1,1]) {
      prop(cube,wood,[x+side*.32*s,y+.69*s,z+.46*s],[.19*s,.3*s,.04*s]);
      prop(cube,glow,[x+side*.32*s,y+.69*s,z+.484*s],[.115*s,.23*s,.016*s]);
    }
    prop(cube,stone,[x+.32*s,y+1.43*s,z-.16*s],[.2*s,.61*s,.19*s]);
    for(let i=0;i<6;i++) {
      const dx=(i/5-.5)*1.32*s, yy=y+1.12*s+(.66*s-Math.abs(dx))*.82;
      prop(cube,terracotta,[x+dx,yy,z],[.035*s,.035*s,1.14*s]);
    }
  }
  const rippleMat=ownM(new THREE.MeshBasicMaterial({color:0x9bcebc,transparent:true,opacity:.14,depthWrite:false,side:THREE.DoubleSide}));
  function island(x,z,r,h,seed,inhabited=false) {
    const geometry=ownG(stratumGeometry(r,h,seed,0x87968a));
    mesh(geometry,rock,[x,SEA_Y-.35,z]);
    const ring=mesh(ownG(new THREE.RingGeometry(r*.96,r*1.05,48)),rippleMat,[x,SEA_Y+.055,z]);ring.rotation.x=-Math.PI/2;ring.castShadow=false;ring.receiveShadow=false;
    for(let i=0;i<6;i++) {
      const a=i/6*TAU+seed,r1=r*(.44+rng()*.24),yh=h*(1-r1/r)*.74;
      prop(sphere,stoneDark,[x+Math.cos(a)*r1,SEA_Y+yh-.15,z+Math.sin(a)*r1],[r*.25,.16,r*.18]);
      shrub(x+Math.cos(a)*r1,SEA_Y+yh+.06,z+Math.sin(a)*r1,r*.16);
    }
    tree(x-.13*r,SEA_Y+h*.82,z,.40+h*.09);
    tree(x+.32*r,SEA_Y+h*.59,z-.13*r,.32+h*.075);
    tree(x-.27*r,SEA_Y+h*.63,z+.17*r,.33+h*.065);
    if(inhabited)cottage(x+r*.23,SEA_Y+h*.64,z+.12*r,.38+r*.13);
  }

  // The sky and wave pattern are inexpensive illustrative shaders, not a ray tracer.
  const skyMat=ownM(new THREE.ShaderMaterial({side:THREE.BackSide,depthWrite:false,
    vertexShader:'varying vec3 vP; void main(){vP=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader:`varying vec3 vP; void main(){vec3 d=normalize(vP);float h=smoothstep(-.1,.6,d.y);vec3 c=mix(vec3(.94,.62,.48),vec3(.31,.35,.58),h);float sun=pow(max(dot(d,normalize(vec3(-.33,.15,-1.0))),0.0),1700.0);float halo=pow(max(dot(d,normalize(vec3(-.33,.15,-1.0))),0.0),22.0);c+=vec3(.8,.45,.15)*halo*.22+vec3(1.5,1.2,.6)*sun;gl_FragColor=vec4(c,1.0);#include <tonemapping_fragment>\n#include <colorspace_fragment>}`.replace(';#include',';\n#include')
  }));
  const sky=mesh(ownG(new THREE.SphereGeometry(190,32,16)),skyMat,[0,0,0]);sky.castShadow=false;sky.receiveShadow=false;sky.renderOrder=-5;
  const seaMat=ownM(new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uSun:{value:new THREE.Vector3(-.35,.48,-.75).normalize()}},vertexShader:oceanVertex,fragmentShader:oceanFragment}));
  const sea=mesh(ownG(new THREE.PlaneGeometry(240,240,170,170)),seaMat,[0,SEA_Y,0]);sea.rotation.x=-Math.PI/2;sea.castShadow=false;sea.receiveShadow=false;sea.name='Illustrated animated sea';
  scene.fog=new THREE.FogExp2(0x9b94ac,.006);
  const hemisphere=new THREE.HemisphereLight(0xaabbd6,0x28483d,.8);root.add(hemisphere);
  const sun=new THREE.DirectionalLight(0xffc895,2.2);sun.position.set(-18,27,-25);sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-24;sun.shadow.camera.right=24;sun.shadow.camera.top=28;sun.shadow.camera.bottom=-28;
  sun.shadow.camera.near=1;sun.shadow.camera.far=85;sun.shadow.bias=-.0003;sun.shadow.normalBias=.035;sun.target.position.set(0,0,-5);root.add(sun,sun.target);
  const reefs=[{x:-5,z:0,r:2.2},{x:4,z:-4,r:2.4},{x:-5,z:-16,r:2.8}];
  reefs.forEach((r,i)=>island(r.x,r.z,r.r,3.8+i*.7,19+i*53,i===2));
  // Scenery outside the playable channel.
  for(const [x,z,r,h,seed] of [[-18,6,4,5.8,102],[19,-5,4.5,6.2,192],[-20,-21,5,7.8,212],[19,-26,4,6.7,83],[-28,-38,7,11,481],[29,-40,7,9,391],[-3,-49,5,8.5,31],[10,-59,6,10,97],[-35,-65,10,12,135],[39,-71,12,13,69]])island(x,z,r,h,seed,true);

  const wall={z:-10,minX:-14,maxX:14,height:2.8,thickness:.65};
  // Closed iron grilles under the arches make the gameplay barrier visible.
  for(let i=0;i<8;i++) {
    const x=-12.25+i*3.5;
    prop(cube,stone,[x,2.35,-10],[3.5,.65,.72]);
    prop(cube,trim,[x,2.72,-10],[3.5,.16,.9]);
    for(const dx of [-1.7,1.7]) {
      prop(cube,stone,[x+dx,1,-10],[.45,2.65,.88]);
      prop(cube,trim,[x+dx,2.82,-10],[.59,.14,.99]);
    }
    for(let j=0;j<6;j++)prop(cylinder,brass,[x-1.35+j*.54,.73,-10],[.025,2.56,.025]);
    for(let j=0;j<5;j++) {
      const a=Math.PI*(j/4),p=[x+Math.cos(a)*1.35,1.1+Math.sin(a)*.9,-9.96];
      prop(cube,trim,p,[.44,.42,.83],-.6+a*.38);
    }
    prop(cube,stoneDark,[x,1.94,-9.624],[1.1,.08,.03]);
  }
  island(-14.5,-10,2.8,4.9,471,true);island(14.5,-10,2.7,5.2,481,true);

  const beacon={x:0,z:-21};
  island(3,-23,3.25,2.7,832,false);
  const lighthouse=new THREE.Group();lighthouse.position.set(3,1.45,-23);root.add(lighthouse);
  mesh(ownG(new THREE.CylinderGeometry(.43,.65,4.5,14)),plaster,[0,2.25,0],[1,1,1],lighthouse);
  mesh(cylinder,stone,[0,.17,0],[.8,.34,.8],lighthouse);
  mesh(cylinder,trim,[0,4.31,0],[.74,.17,.74],lighthouse);
  mesh(cylinder,brass,[0,4.78,0],[.43,.77,.43],lighthouse);
  mesh(cylinder,glow,[0,4.81,0],[.38,.58,.38],lighthouse);
  mesh(cone,terracotta,[0,5.3,0],[.75,.64,.75],lighthouse);
  for(let i=0;i<8;i++){
    const a=i/8*TAU;
    mesh(cylinder,stoneDark,[Math.cos(a)*.42,4.8,Math.sin(a)*.42],[.025,.79,.025],lighthouse);
    mesh(cylinder,stoneDark,[Math.cos(a)*.72,4.63,Math.sin(a)*.72],[.02,.54,.02],lighthouse);
  }
  for(const yy of [.9,2,3.1])mesh(cube,teal,[0,yy,.56],[.23,.4,.05],lighthouse);
  // Goal dock fits the town's full width and has two clear beacon posts.
  for(let i=0;i<25;i++)prop(cube,wood,[-5+i*.32,-.09,-23.5],[.26,.15,2.4]);
  for(const x of [-5,-3,-1,1,3]) {
    prop(cylinder,wood,[x,-.55,-22.3],[.075,1.45,.075]);
    prop(cylinder,wood,[x,-.55,-24.6],[.075,1.45,.075]);
    prop(cylinder,brass,[x,.49,-24.6],[.025,1,.025]);
    prop(sphere,glow,[x,1.08,-24.6],[.12,.18,.12]);
  }
  for(const x of [-3,3]) {
    prop(cylinder,wood,[x,1.1,-21.7],[.09,2.55,.09]);
    prop(cone,terracotta,[x,2.48,-21.7],[.19,.24,.19]);
    prop(sphere,glow,[x,2.16,-21.7],[.17,.24,.17]);
  }
  const banner=mesh(ownG(new THREE.PlaneGeometry(4.9,.55,12,1)),flag,[0,2.05,-21.71]);animated.push({kind:'banner',mesh:banner,base:new Float32Array(banner.geometry.attributes.position.array)});
  cottage(5.1,1.55,-23.9,1);cottage(3.3,1.9,-25,.72);
  lineBetween([-5,.38,-24.6],[3,.38,-24.6],wood,.027);

  const collectibles=[{id:'first',x:-9,z:6},{id:'second',x:8,z:-5},{id:'third',x:0,z:-17}];
  for(const [i,c] of collectibles.entries()) {
    const group=new THREE.Group();group.position.set(c.x,.55,c.z);root.add(group);
    const outer=ownG(new THREE.TorusGeometry(.34,.02,6,24));
    mesh(outer,brass,[0,.3,0],[1,1,1],group);
    mesh(sphere,glow,[0,.3,0],[.16,.25,.16],group);
    mesh(cone,terracotta,[0,.64,0],[.2,.12,.2],group);
    mesh(cube,teal,[0,-.045,0],[.09,.2,.025],group);
    const halo=mesh(ownG(new THREE.RingGeometry(.45,.51,40)),rippleMat,[0,-1.02,0],[1,1,1],group);halo.rotation.x=-Math.PI/2;
    group.name=`Rescue lantern ${i+1}`;c.group=group;c.mesh=group;c.object=group;
    animated.push({kind:'lantern',mesh:group,x:c.x,y:.55,z:c.z,phase:i*1.3});
  }
  // Small gulls make the scale legible without high object counts.
  for(let i=0;i<9;i++) {
    const g=new THREE.Group();
    const wingL=mesh(cube,plaster,[-.13,0,0],[.3,.025,.065],g);
    const wingR=mesh(cube,plaster,[.13,0,0],[.3,.025,.065],g);
    root.add(g);animated.push({kind:'gull',mesh:g,wingL,wingR,phase:i*1.7,r:8+rng()*7,y:5+rng()*4});
  }
  for(const [geometry,byMaterial] of batches)for(const [material,transforms] of byMaterial) {
    const batch=new THREE.InstancedMesh(geometry,material,transforms.length);transforms.forEach((m,i)=>batch.setMatrixAt(i,m));batch.castShadow=true;batch.receiveShadow=true;batch.instanceMatrix.needsUpdate=true;root.add(batch);
  }
  let disposed=false;
  return {root,reefs,wall,beacon,collectibles,seaY:SEA_Y,
    update(t){
      if(disposed)return;seaMat.uniforms.uTime.value=t;
      for(const a of animated) {
        if(a.kind==='lantern'){a.mesh.position.y=a.y+Math.sin(t*1.6+a.phase)*.11;a.mesh.rotation.y=t*.45+a.phase;}
        else if(a.kind==='gull'){
          const angle=t*.07+a.phase;a.mesh.position.set(Math.cos(angle)*a.r,a.y+Math.sin(t*.4+a.phase)*.35,-13+Math.sin(angle)*a.r*.65);
          a.mesh.rotation.y=-angle;a.wingL.rotation.z=Math.sin(t*4+a.phase)*.32;a.wingR.rotation.z=-a.wingL.rotation.z;
        }else if(a.kind==='banner'){
          const p=a.mesh.geometry.attributes.position;for(let i=0;i<p.count;i++)p.setZ(i,a.base[i*3+2]+Math.sin(a.base[i*3]*2.6+t*2.1)*.055);p.needsUpdate=true;
        }
      }
    },
    dispose(){if(disposed)return;disposed=true;scene.remove(root);for(const g of geometries)g.dispose();for(const m of materials)m.dispose();sun.shadow.dispose();}
  };
}

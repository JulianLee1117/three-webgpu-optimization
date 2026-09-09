import * as THREE from 'three/webgpu';
import {positionLocal, uniform, float, vec3, mix, positionView, dFdx, dFdy} from 'three/tsl';
import {numericProgram} from './programs.js';
import {LIMITS} from './fixtures.mjs';

export function createDisplay({fixture,evaluator,container,onTarget,onError}) {
  const {renderer,owned}=evaluator,scene=new THREE.Scene(),clock=uniform(fixture.target.time);
  renderer.setClearColor(0x0c161f);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  container.appendChild(renderer.domElement);
  const span=4.05,offsets=[-span/2,span/2];
  const camera=new THREE.OrthographicCamera(-4.7,4.7,2.6,-2.6,.1,40);
  const center=new THREE.Vector3(1.4,.75,0);camera.position.copy(center).add(new THREE.Vector3(.5,2.1,9));camera.lookAt(center);
  const ambient=new THREE.HemisphereLight(0xd3f6ff,0x0b161e,2);scene.add(ambient);
  const key=new THREE.DirectionalLight(0xffffff,4);key.position.set(1,4,5);scene.add(key);
  const fill=new THREE.DirectionalLight(0x58d8ca,2);fill.position.set(-4,1,-1);scene.add(fill);
  const sections=192,sides=10,vertices=[],indices=[];
  for(let i=0;i<=sections;i++)for(let j=0;j<=sides;j++){
    const angle=j/sides*2*Math.PI;vertices.push(i/sections,Math.cos(angle)*LIMITS.crossSectionRadius,Math.sin(angle)*LIMITS.crossSectionRadius);
  }
  for(let i=0;i<sections;i++)for(let j=0;j<sides;j++){
    const a=i*(sides+1)+j,b=a+sides+1;indices.push(a,b,a+1,a+1,b,b+1);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.setIndex(indices);geometry.computeVertexNormals();owned.push(geometry);
  const handles=[],pathLines=[];
  for(let side=0;side<2;side++){
    const group=new THREE.Group();group.position.x=offsets[side];scene.add(group);
    const values=side?undefined:fixture.initial.map(v=>float(v));
    const forward=evaluator.bind(positionLocal,clock,values);
    const material=new THREE.MeshStandardNodeMaterial({roughness:.3,metalness:.55,side:THREE.DoubleSide});owned.push(material);
    material.positionNode=forward;
    // Fragment derivatives use the actual deformed view position. Rest normals
    // would otherwise shade the GPU-deformed tube incorrectly.
    material.normalNode=dFdx(positionView).cross(dFdy(positionView)).normalize();
    const palette=positionLocal.x.div(3).clamp(0,1);
    material.colorNode=side?mix(vec3(.13,.6,.43),vec3(.64,.97,.64),palette):mix(vec3(.14,.30,.43),vec3(.5,.7,.86),palette);
    const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;group.add(mesh);
    const obstacleGeometry=new THREE.SphereGeometry(fixture.obstacle.radius,24,16),obstacleMaterial=new THREE.MeshStandardNodeMaterial({color:0xff935a,roughness:.4,metalness:.2});owned.push(obstacleGeometry,obstacleMaterial);
    const sphere=new THREE.Mesh(obstacleGeometry,obstacleMaterial);sphere.position.set(...fixture.obstacle.center);group.add(sphere);
    const haloGeometry=new THREE.SphereGeometry(fixture.obstacle.radius+LIMITS.crossSectionRadius,16,10),haloMaterial=new THREE.MeshBasicNodeMaterial({color:0xffae79,wireframe:true,transparent:true,opacity:.16});owned.push(haloGeometry,haloMaterial);
    const halo=new THREE.Mesh(haloGeometry,haloMaterial);halo.position.copy(sphere.position);group.add(halo);
    const anchorGeometry=new THREE.SphereGeometry(.065,16,10),anchorMaterial=new THREE.MeshStandardNodeMaterial({color:0x7b9aa9,metalness:.7,roughness:.25});owned.push(anchorGeometry,anchorMaterial);group.add(new THREE.Mesh(anchorGeometry,anchorMaterial));
    const handleGeometry=new THREE.SphereGeometry(.06,16,10),handleMaterial=new THREE.MeshBasicNodeMaterial({color:side?0xbcffdd:0xabc5d4});owned.push(handleGeometry,handleMaterial);
    const handle=new THREE.Mesh(handleGeometry,handleMaterial);group.add(handle);handles.push(handle);
    const trailGeometry=new THREE.BufferGeometry(),trailMaterial=new THREE.LineBasicNodeMaterial({color:side?0x7fbfa9:0x425c6e,transparent:true,opacity:.45});owned.push(trailGeometry,trailMaterial);
    const trail=new THREE.Line(trailGeometry,trailMaterial);group.add(trail);pathLines.push(trail);
    const grid=new THREE.GridHelper(3.6,18,0x29424e,0x192d38);grid.position.set(1.4,-.65,0);group.add(grid);owned.push(grid.geometry,grid.material);
  }
  const targetGeometry=new THREE.TorusGeometry(.10,.013,8,36),targetMaterial=new THREE.MeshBasicNodeMaterial({color:0xafffda});owned.push(targetGeometry,targetMaterial);
  const target=new THREE.Mesh(targetGeometry,targetMaterial);scene.add(target);
  let current=[...fixture.initial],disposed=false,raf=0,playing=false,lastFrame=0,start=0;
  const fail=error=>{if(disposed)return;pause();onError?.(error);};
  const gpuError=event=>fail(event.error);evaluator.device.addEventListener('uncapturederror',gpuError);
  evaluator.device.lost.then(info=>{if(!disposed)fail(Error('GPU device lost: '+info.message));});
  function targetPosition(){target.position.set(fixture.target.position[0]+offsets[1],fixture.target.position[1],fixture.target.position[2]);target.quaternion.copy(camera.quaternion);}
  function refreshPaths(){
    for(let side=0;side<2;side++){
      const w=side?current:fixture.initial,points=[];
      for(let i=0;i<=96;i++)points.push(new THREE.Vector3(...numericProgram(fixture.kind,[1,0,0],i/96,w)));
      pathLines[side].geometry.setFromPoints(points);
    }
  }
  function render(time=clock.value){
    if(disposed)return;
    try{clock.value=time;
      for(let side=0;side<2;side++)handles[side].position.set(...numericProgram(fixture.kind,[1,0,0],time,side?current:fixture.initial));
      targetPosition();renderer.render(scene,camera);evaluator.check();
    }catch(error){fail(error);}
  }
  function resize(){
    if(disposed)return;
    // During a viewport/layout transition the container can briefly have no
    // drawable area. Retain the last valid target until layout settles.
    const width=Math.min(1260,Math.floor(container.clientWidth)),height=Math.floor(container.clientHeight);
    if(width<2||height<2)return;
    renderer.setSize(width,height,false);const aspect=width/height,halfWidth=4.6,halfHeight=halfWidth/aspect;
    camera.left=-halfWidth;camera.right=halfWidth;camera.top=halfHeight;camera.bottom=-halfHeight;camera.updateProjectionMatrix();render();
  }
  function setParameters(values){current=Array.from(values);evaluator.load(current);refreshPaths();render(fixture.target.time);}
  function pause(){playing=false;cancelAnimationFrame(raf);raf=0;}
  function play(){
    pause();playing=true;start=performance.now();lastFrame=0;
    const frame=now=>{if(!playing||disposed)return;if(now-start>=12000){pause();render(fixture.target.time);return;}if(now-lastFrame>=1000/30){render(((now-start)/4000+fixture.target.time)%1);lastFrame=now;}if(playing&&!disposed)raf=requestAnimationFrame(frame);};
    raf=requestAnimationFrame(frame);
  }
  const ray=new THREE.Raycaster(),mouse=new THREE.Vector2(),plane=new THREE.Plane(new THREE.Vector3(0,0,1),-fixture.target.position[2]);let dragging=false;
  function cursor(event){const bounds=renderer.domElement.getBoundingClientRect();mouse.set(2*(event.clientX-bounds.left)/bounds.width-1,1-2*(event.clientY-bounds.top)/bounds.height);ray.setFromCamera(mouse,camera);}
  const down=event=>{cursor(event);const projected=target.position.clone().project(camera),dx=(projected.x-mouse.x)*renderer.domElement.clientWidth/2,dy=(projected.y-mouse.y)*renderer.domElement.clientHeight/2;
    if(Math.hypot(dx,dy)<=14||ray.intersectObject(target).length){dragging=true;pause();renderer.domElement.setPointerCapture(event.pointerId);}};
  const move=event=>{if(!dragging)return;cursor(event);const p=ray.ray.intersectPlane(plane,new THREE.Vector3());if(!p)return;
    // This fixture has no longitudinal tip control; keep its known fixed x.
    const x=fixture.kind==='ribbon'?3:Math.max(.2,Math.min(3.1,p.x-offsets[1]));
    const next=[x,Math.max(-.5,Math.min(2.4,p.y)),fixture.target.position[2]];onTarget?.(next);targetPosition();render(fixture.target.time);};
  const up=()=>dragging=false;
  renderer.domElement.addEventListener('pointerdown',down);renderer.domElement.addEventListener('pointermove',move);renderer.domElement.addEventListener('pointerup',up);renderer.domElement.addEventListener('pointercancel',up);
  function dispose(){pause();disposed=true;evaluator.device.removeEventListener('uncapturederror',gpuError);renderer.domElement.removeEventListener('pointerdown',down);renderer.domElement.removeEventListener('pointermove',move);renderer.domElement.removeEventListener('pointerup',up);renderer.domElement.removeEventListener('pointercancel',up);}
  refreshPaths();resize();
  return {render,resize,setParameters,play,pause,dispose,get playing(){return playing;},get parameters(){return [...current];},get targetScreen(){const p=target.position.clone().project(camera);return {x:(p.x+1)/2*renderer.domElement.clientWidth,y:(1-p.y)/2*renderer.domElement.clientHeight};}};
}

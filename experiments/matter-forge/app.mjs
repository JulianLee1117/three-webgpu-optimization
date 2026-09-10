import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createSimulator } from './gpu.mjs';
import { transmute, particleTotals, makeState, regularParticles, mergeStates } from './cpu.mjs';
import { castingFixtureV2 } from './casting-fixture-v2.mjs';
import { buildEnvironment } from './environment.mjs';
import { voxelizeMeshes } from './voxelize.mjs';
import { importGLBMatter } from './import-glb.mjs';
import { createLiquidSurface } from './liquid-surface.mjs';

const $=id=>document.getElementById(id), evidence={prepared:false,errors:[],edits:[],runs:[],frames:0};
let renderer,device,scene,camera,controls,environment,sim,state,fixture,solid,liquid,brush,removable,liquidSurface;
let stage='pour',castSnapshot=null,releaseFinished=false;
let importedSource=null,bodyColors=null,sourceLoading=false;
let ready=false,running=false,pending=false,disposed=false,resetPromise=null,tool='orbit',painting=false,queuedPoint=null,cutQueued=false,frame=0,runStart=0,runStep=0;
const dummy=new THREE.Object3D(),color=new THREE.Color(),pointer=new THREE.Vector2(),ray=new THREE.Raycaster(),plane=new THREE.Plane(new THREE.Vector3(0,0,1),-11.5);
const setStatus=text=>{$('status').textContent=text;};
function defaultFixture(){
 const f=castingFixtureV2(),source=new THREE.Mesh(new THREE.BoxGeometry(12,4,3),new THREE.MeshBasicNodeMaterial({color:0x70d9ce}));source.position.set(12,15,11.5);source.updateMatrixWorld(true);
 let sampled;try{sampled=voxelizeMeshes([source],{spacing:.5,maxParticles:4096});}finally{source.geometry.dispose();source.material.dispose();}
 if(sampled.pointCount!==f.bodyCount||sampled.points.some((v,i)=>v!==f.state.x[i]))throw Error('Closed-mesh sampling changed the frozen casting fixture');
 // The live state uses the actual mesh sampler output. The exact comparison
 // above keeps this default scene tied to the independently checked fixture.
 f.state.x.set(sampled.points,0);f.state.mass.set(sampled.masses,0);f.state.volume.set(sampled.volumes,0);f.beamCount=f.bodyCount;
 f.initialMass=particleTotals(f.state).mass;bodyColors=sampled.colors;evidence.source={kind:'closed Three BoxGeometry',triangles:12,sampling:sampled.diagnostics};return f;
}
function selectedFixture(){
 if(!importedSource)return defaultFixture();const f=castingFixtureV2(),s=importedSource;
 f.state=mergeStates([makeState({x:s.points,mass:s.masses,volume:s.volumes}),regularParticles(f.protocol.load)]);f.bodyCount=f.beamCount=s.pointCount;f.initialMass=particleTotals(f.state).mass;bodyColors=s.colors;evidence.source={kind:'local closed GLB',metadata:s.source,sampling:s.diagnostics};return f;
}
function allocateParticles(){
 for(const mesh of [solid,liquid])if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
 solid=new THREE.InstancedMesh(new THREE.BoxGeometry(.51,.51,.51),new THREE.MeshStandardNodeMaterial({metalness:.32,roughness:.27}),state.ids.length);
 liquid=new THREE.InstancedMesh(new THREE.IcosahedronGeometry(.33,1),new THREE.MeshStandardNodeMaterial({metalness:.55,roughness:.16}),state.ids.length);
 for(const mesh of [solid,liquid]){mesh.frustumCulled=false;mesh.castShadow=true;mesh.receiveShadow=true;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(mesh);}
 $('particle-count').textContent='WEBGPU · '+state.ids.length.toLocaleString()+' PARTICLES';
}
function error(e){stop();evidence.errors.push(String(e.stack??e));setStatus('Stopped: '+e.message);$('loading').textContent=e.message;}
function display(){
 if(!ready||disposed)return;
 for(let i=0;i<state.ids.length;i++){
  const isSolid=!!state.mode[i],cargo=i>=fixture.beamCount;
  dummy.position.fromArray(state.x,i*3);dummy.scale.setScalar(isSolid?1:0);dummy.updateMatrix();
  if(isSolid){const k=9*i,F=state.F;dummy.matrix.set(F[k],F[k+1],F[k+2],state.x[3*i],F[k+3],F[k+4],F[k+5],state.x[3*i+1],F[k+6],F[k+7],F[k+8],state.x[3*i+2],0,0,0,1);}
  solid.setMatrixAt(i,dummy.matrix);
  dummy.scale.setScalar(isSolid?0:1);dummy.updateMatrix();liquid.setMatrixAt(i,dummy.matrix);
  if(cargo)color.set(0xffb55c);else color.fromArray(bodyColors,i*3);
  color.multiplyScalar((isSolid?1:.85)*(.85+.15*(i%7)/6));solid.setColorAt(i,color);liquid.setColorAt(i,color);
 }
 for(const m of [solid,liquid]){m.instanceMatrix.needsUpdate=true;m.instanceColor.needsUpdate=true;}
 if(liquidSurface){const stats=liquidSurface.update(state,fixture.beamCount,bodyColors);liquid.visible=!liquidSurface.mesh.visible;evidence.surface={...stats};}
 $('time').textContent=state.time.toFixed(2)+' s';$('mass').textContent=(100*particleTotals(state).mass/fixture.initialMass).toFixed(0)+'%';
 renderer.render(scene,camera);evidence.frames++;
}
function selectTool(next){queuedPoint=null;tool=next;controls.enabled=next==='orbit';for(const id of ['orbit','melt','freeze'])$(id).classList.toggle('active',id===next);brush.visible=false;display();}
function applyQueuedEdit(){
 if(!queuedPoint&&!cutQueued)return;
 const selection=[];
 for(let i=0;i<fixture.beamCount;i++){
  const x=state.x[3*i],y=state.x[3*i+1],z=state.x[3*i+2];
  const selected=cutQueued?true:Math.hypot(x-queuedPoint.x,y-queuedPoint.y,z-queuedPoint.z)<2.2;
  if(selected&&state.mode[i]!==((cutQueued||tool==='melt')?0:1))selection.push(i);
 }
 const mode=(cutQueued||tool==='melt')?0:1;queuedPoint=null;cutQueued=false;
 if(selection.length){state=transmute(state,selection,mode);sim.replaceState(state);evidence.edits.push({step:state.steps,mode,count:selection.length,mass:selection.reduce((s,i)=>s+state.mass[i],0)});setStatus(`${mode?'Solidified':'Liquefied'} ${selection.length} particles. Positions, velocity and mass retained.`);}
}
async function tick(){
 if(!running||disposed)return;
 if(document.hidden||performance.now()-runStart>12000||state.steps-runStep>=960){stop();setStatus('Paused. Continue the current stage, or reset.');return;}
 pending=true;
 try{applyQueuedEdit();state=await sim.advance(8);display();checkStage();}catch(e){error(e);}finally{pending=false;}
 if(running)frame=requestAnimationFrame(tick);
}
function start(){if(!ready||running||pending||disposed||resetPromise||sourceLoading)return;running=true;runStart=performance.now();runStep=state.steps;evidence.runs.push({startStep:state.steps,startTime:state.time});$('play').textContent='Pause';frame=requestAnimationFrame(tick);}
function stop(){running=false;cancelAnimationFrame(frame);if(ready)$('play').textContent=stage==='cast'?'Solidify & release':'Continue';}
function checkStage(){
 if(stage==='pour'&&state.steps>=960){stop();stage='cast';castSnapshot=makeState(state);$('play').textContent='Solidify & release';$('control').classList.remove('hidden');$('control').disabled=!!resetPromise||sourceLoading;setStatus('The cast is ready. Solidify and remove the mold, or release the same material as liquid.');evidence.castReady=true;}
 if(stage==='released'&&!releaseFinished&&state.steps>=1920){releaseFinished=true;stop();setStatus('Load test complete. Melt the bridge and continue to change what supports the cargo.');}
}
function makeMold(config){
 const group=new THREE.Group();group.name='Visible removable mold, grille and load suspension';
 for(const box of config.colliders.filter(b=>!b.name.includes('bank'))){
  const size=box.min.map((v,i)=>box.max[i]-v),center=box.min.map((v,i)=>(v+box.max[i])/2),grille=box.name.includes('grille'),holder=box.name.includes('holder');
  const geometry=new THREE.BoxGeometry(...size),material=new THREE.MeshStandardNodeMaterial({color:grille?0x705346:0x79c9ce,metalness:grille?.7:.15,roughness:grille?.42:.3,transparent:!grille,opacity:holder?.055:grille?1:.09,depthWrite:grille});
  const mesh=new THREE.Mesh(geometry,material);mesh.position.set(...center);mesh.name=box.name;mesh.userData.collider=structuredClone(box);mesh.castShadow=grille;group.add(mesh);
  const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicNodeMaterial({color:holder?0xffbd79:0x83adb0,transparent:true,opacity:.5}));edges.position.set(...center);group.add(edges);
 }
 return group;
}
function release(mode=1){
 if(sourceLoading)return Promise.reject(Error('Wait for the model import'));
 if(resetPromise)return resetPromise;if(stage!=='cast'||pending||disposed)return Promise.reject(Error('Finish casting before release'));
 resetPromise=(async()=>{
  stop();for(const id of ['play','control','reset','orbit','melt','freeze','cut','load'])$(id).disabled=true;
  state=mode===1?transmute(state,Array.from({length:fixture.beamCount},(_,i)=>i),1):makeState(state);
  sim.dispose();sim=await createSimulator(device,fixture.releasedConfig,state);if(disposed){sim.dispose();return;}stage='released';releaseFinished=false;removable.visible=false;
  evidence.release={mode,step:state.steps,particleCount:state.ids.length,totalMass:particleTotals(state).mass};$('control').classList.add('hidden');$('play').textContent='Drop cargo';display();
  setStatus(`${mode?'Solidified':'Liquid'} cast. Mold, grille and cargo holder removed. Only the two banks and end clamps remain.`);
 })().finally(()=>{resetPromise=null;if(!disposed)for(const id of ['play','reset','orbit','melt','freeze','cut','load'])$(id).disabled=false;});
 return resetPromise;
}
function reset(){
 if(resetPromise)return resetPromise;if(disposed)return Promise.reject(Error('Demo disposed'));
 resetPromise=(async()=>{
  stop();for(const id of ['play','control','reset','orbit','melt','freeze','cut','load'])$(id).disabled=true;
  while(pending)await new Promise(r=>setTimeout(r,10));sim?.dispose();fixture=selectedFixture();state=fixture.state;allocateParticles();
  sim=await createSimulator(device,fixture.castingConfig,state);if(disposed){sim.dispose();return;}stage='pour';castSnapshot=null;removable.visible=true;$('control').classList.add('hidden');cutQueued=false;queuedPoint=null;evidence.resetCount=(evidence.resetCount??0)+1;display();$('play').textContent='Pour the material';setStatus('Cyan liquid falls through the bronze grille into the outlined mold. Amber cargo is held above it.');
 })().finally(()=>{resetPromise=null;if(!disposed)for(const id of ['play','reset','orbit','melt','freeze','cut','load'])$(id).disabled=false;});
 return resetPromise;
}
async function prepare(){
 $('prepare').disabled=true;$('loading').textContent='Building the foundry and compiling bounded physics…';
 try{
  renderer=new THREE.WebGPURenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setSize(innerWidth,innerHeight);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.05;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap;document.body.prepend(renderer.domElement);await renderer.init();
  if(disposed){await renderer.dispose();return;}
  if(!renderer.backend.isWebGPUBackend)throw Error('WebGPU is required');device=renderer.backend.device;device.addEventListener('uncapturederror',event=>error(event.error));
  device.lost.then(info=>{if(!disposed)error(Error('GPU device lost: '+info.message));});
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,.1,320);camera.position.set(33,26,37);camera.lookAt(12,8,11.5);
  fixture=castingFixtureV2();fixture.beamCount=fixture.bodyCount;environment=buildEnvironment(scene,fixture.releasedConfig);removable=makeMold(fixture.castingConfig);scene.add(removable);liquidSurface=createLiquidSurface();liquidSurface.mesh.castShadow=liquidSurface.mesh.receiveShadow=true;scene.add(liquidSurface.mesh);
  controls=new OrbitControls(camera,renderer.domElement);controls.target.set(12,8,11.5);controls.minDistance=16;controls.maxDistance=65;controls.maxPolarAngle=Math.PI*.48;controls.addEventListener('change',()=>{if(!running)display();});controls.update();
  brush=new THREE.Mesh(new THREE.SphereGeometry(2.2,16,8),new THREE.MeshBasicNodeMaterial({color:0xffffff,wireframe:true,transparent:true,opacity:.2,depthWrite:false}));brush.visible=false;scene.add(brush);
  ready=true;await reset();if(disposed)return;evidence.prepared=true;$('intro').classList.add('hidden');for(const id of ['play','reset','orbit','melt','freeze','cut'])$(id).disabled=false;
 }catch(e){error(e);await dispose();}
}
function pointAt(event){pointer.set(event.clientX/innerWidth*2-1,1-event.clientY/innerHeight*2);ray.setFromCamera(pointer,camera);const hit=new THREE.Vector3();return ray.ray.intersectPlane(plane,hit)?hit:null;}
function paint(event){
 if(!ready||tool==='orbit'||disposed||resetPromise||sourceLoading||event.target!==renderer.domElement)return;
 const hit=pointAt(event);if(!hit)return;brush.position.copy(hit);brush.visible=true;brush.material.color.set(tool==='melt'?0xffbd76:0xafffff);
 if(painting){queuedPoint=hit.clone();if(!pending){applyQueuedEdit();display();}}
 if(!running)display();
}
async function dispose(){stop();disposed=true;try{await resetPromise;}catch{}while(pending)await new Promise(r=>setTimeout(r,10));sim?.dispose();controls?.dispose();environment?.dispose();liquidSurface?.dispose();removable?.traverse(m=>{m.geometry?.dispose();m.material?.dispose();});for(const m of [solid,liquid,brush]){m?.geometry.dispose();m?.material.dispose();}const loss=device?.lost;await renderer?.dispose();if(loss)evidence.deviceLossReason=(await loss).reason;evidence.disposed=true;}
$('prepare').onclick=prepare;$('play').onclick=()=>stage==='cast'?release(1).then(start).catch(error):running?stop():start();$('reset').onclick=()=>reset().catch(error);$('control').onclick=()=>release(0).then(start).catch(error);
for(const id of ['orbit','melt','freeze'])$(id).onclick=()=>selectTool(id);
$('cut').onclick=()=>{cutQueued=true;if(!pending){applyQueuedEdit();display();}};
$('load').onclick=()=>{$('file').value='';$('file').click();};
$('file').onchange=async()=>{
 const file=$('file').files[0];if(!file||resetPromise||disposed||sourceLoading)return;stop();sourceLoading=true;for(const id of ['play','control','reset','orbit','melt','freeze','cut','load'])$(id).disabled=true;
 try{while(pending)await new Promise(r=>setTimeout(r,10));const imported=await importGLBMatter(file);if(disposed)return;importedSource=imported;await reset();setStatus(`Loaded ${file.name}: ${imported.pointCount} particles, ${(imported.pointCount*.125).toFixed(1)} units of material. Reset repeats this model.`);}
 catch(e){setStatus('Model not loaded: '+e.message);}finally{sourceLoading=false;if(!disposed){for(const id of ['play','reset','orbit','melt','freeze','cut','load'])$(id).disabled=false;$('control').disabled=stage!=='cast';}}
};
window.addEventListener('pointerdown',e=>{painting=true;paint(e);});window.addEventListener('pointermove',paint);window.addEventListener('pointerup',()=>painting=false);
window.addEventListener('resize',()=>{if(ready){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);display();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',()=>{stop();sim?.dispose();renderer?.dispose();disposed=true;});
window.matterForge={prepare,start,stop,reset,release,dispose,evidence,getState:()=>state,getStage:()=>stage,async advance(steps){stop();if(pending||resetPromise)throw Error('Wait for pending work');pending=true;try{applyQueuedEdit();state=await sim.advance(steps);display();checkStage();return state;}finally{pending=false;}},cut(){cutQueued=true;},selectTool};

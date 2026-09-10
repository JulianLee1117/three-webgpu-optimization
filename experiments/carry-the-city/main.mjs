import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {captureClip,samplePoint} from './motion.mjs';
import {createTown} from './town.mjs';
import {createWorld} from './world.mjs';
import {createMotionRig} from './rig.mjs';
import {createWorldMotionHistory} from './world-motion.mjs';
import {addFins} from './fins.mjs';

const $=id=>document.getElementById(id),clamp=THREE.MathUtils.clamp,lerp=THREE.MathUtils.lerp;
const evidence={ready:false,errors:[],events:[],frames:0,maxFrameMs:0,disposed:false};
const SWIM_Y=-.3;
const state={playing:false,paused:false,phase:'intro',mode:'swim',blend:0,time:0,elapsed:0,x:0,z:7,y:SWIM_Y,yaw:Math.PI/2,speed:0,target:null,lanterns:new Set(),rescued:0,bumps:0};
const clips={swim:captureClip({kind:'swim'}),fly:captureClip({kind:'fly'})};
const motionHistory=createWorldMotionHistory(),cargoAnchor=[-1.5,0,1.6];let cargoPending=false;
let renderer,scene,camera,composer,controls,world,town,rig,environment,pmrem,fish,moth,routeMesh,ring,gangway;
let disposed=false,raf=0,last=0,pointerDown=false,toastUntil=0,manualCameraUntil=0,lastInteraction=performance.now(),impactUntil=0,flightHint=false,dockHint=false,dockStarted=0,sound=false,audio;
const keys=new Set(),ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),seaPlane=new THREE.Plane(new THREE.Vector3(0,1,0),0),vec=new THREE.Vector3();
const destination={x:0,z:-17.2};
const sparks=[],loose=[],sourceParts=[];
const screenSamples=[];
function event(type,data={}){evidence.events.push({type,time:state.time,...data});}
function showToast(text,duration=3){$('toast').textContent=text;$('toast').classList.add('visible');toastUntil=state.time+duration;}
function note(frequency=660,duration=.35){
 if(!sound)return;audio??=new AudioContext();if(audio.state==='suspended')audio.resume();const oscillator=audio.createOscillator(),gain=audio.createGain();oscillator.type='sine';oscillator.frequency.setValueAtTime(frequency,audio.currentTime);gain.gain.setValueAtTime(0,audio.currentTime);gain.gain.linearRampToValueAtTime(.065,audio.currentTime+.015);gain.gain.exponentialRampToValueAtTime(.0001,audio.currentTime+duration);oscillator.connect(gain).connect(audio.destination);oscillator.start();oscillator.stop(audio.currentTime+duration+.02);
}
function fail(error){evidence.errors.push(String(error.stack??error));pause();const box=document.createElement('div');box.className='error';box.textContent=error.message;document.body.append(box);}
function mat(color,roughness=.6,metalness=0){return new THREE.MeshStandardMaterial({color,roughness,metalness});}
function buildCreature(kind){
 const group=new THREE.Group();group.userData.kind=kind;
 const make=(geometry,material,position=[0,0,0],rotation=[0,0,0])=>{geometry.rotateX(rotation[0]);geometry.rotateY(rotation[1]);geometry.rotateZ(rotation[2]);geometry.translate(...position);const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=true;group.add(mesh);sourceParts.push({mesh,kind,rest:geometry.attributes.position.array.slice()});return mesh;};
 if(kind==='swim'){
  const orange=mat(0xed803f,.34),ivory=mat(0xfff0d1,.35),black=mat(0x162d35,.22);
  const body=new THREE.SphereGeometry(1,24,12);body.scale(3,.52,.73);make(body,ivory);
  for(const x of [-1.4,.15,1.5]){const patch=new THREE.SphereGeometry(1,12,8);patch.scale(.62,.14,.53);make(patch,orange,[x,.43,.03]);}
  const tail=new THREE.ConeGeometry(1.12,1.35,3);tail.scale(1,.3,1);make(tail,orange,[-3.05,0,0],[0,0,Math.PI/2]);
  for(const side of [-1,1]){const fin=new THREE.ConeGeometry(.58,.95,3);fin.scale(1,.15,1);make(fin,orange,[.3,-.05,side*.8],[0,side*.55,Math.PI/2]);make(new THREE.SphereGeometry(.13,8,6),black,[2.45,.2,side*.39]);}
  group.scale.setScalar(.33);
 }else{
  const wing=mat(0xffdcaa,.65),pink=mat(0xedbaa9,.7),body=mat(0x65535a,.6);
  for(const side of [-1,1]){const shape=new THREE.SphereGeometry(1,18,10);shape.scale(1.35,.055,1.18);make(shape,wing,[.2,0,side*.95]);const lower=new THREE.SphereGeometry(1,12,8);lower.scale(.92,.05,.88);make(lower,pink,[-1,0,side*.72]);const eye=new THREE.SphereGeometry(.32,12,8);eye.scale(1,.15,1);make(eye,mat(0x855772),[.45,.07,side*1.25]);}
  const b=new THREE.SphereGeometry(1,12,8);b.scale(1.8,.22,.2);make(b,body);
  group.scale.setScalar(.47);
 }
 scene.add(group);return group;
}
function particleBurst(position,color=0xffdf8a,count=20){
 for(let i=0;i<count&&sparks.length<140;i++){
  const mesh=new THREE.Mesh(new THREE.IcosahedronGeometry(.06,0),new THREE.MeshBasicMaterial({color,transparent:true}));mesh.position.copy(position);scene.add(mesh);
  const a=i*2.39996;sparks.push({mesh,life:1.4,v:new THREE.Vector3(Math.cos(a)*1.7,.4+(i%7)*.25,Math.sin(a)*1.7)});
 }
}
function mode(kind){
 if(!state.playing||state.phase==='docking'||state.phase==='done')return;
 if(kind==='swim'&&Math.abs(state.z+10)<5){showToast('Stay airborne until the whole town clears the wall.');return;}
 state.mode=kind;$('swim').classList.toggle('active',kind==='swim');$('fly').classList.toggle('active',kind==='fly');
 showToast(kind==='fly'?'A moth’s wingbeat. A city takes flight.':'A koi’s rhythm. The streets begin to swim.');
 particleBurst(new THREE.Vector3(state.x,state.y+1,state.z),kind==='fly'?0xffd8af:0x98fff0,28);note(kind==='fly'?880:520);event('motion-borrowed',{kind,clip:clips[kind].id});
}
function start(){
 if(!evidence.ready||disposed)return;
 state.playing=true;state.paused=false;state.phase='voyage';lastInteraction=performance.now();
 $('intro').classList.add('hidden');for(const id of ['goal','bottom','progress','pause','restart','map'])$(id).classList.remove('hidden');
 showToast('The town is yours. Drag the water to lead it.',4);particleBurst(new THREE.Vector3(0,1,7),0xaaffdf,26);note(520);event('start');last=performance.now();schedule();
}
function pause(){if(disposed)return;state.paused=true;cancelAnimationFrame(raf);raf=0;if(state.playing)$('paused').classList.remove('hidden');}
function resume(){if(disposed||!evidence.ready)return;state.paused=false;$('paused').classList.add('hidden');lastInteraction=performance.now();last=performance.now();schedule();}
function clearEffects(){for(const p of [...sparks,...loose]){scene.remove(p.mesh);p.mesh.geometry.dispose();p.mesh.material.dispose();}sparks.length=loose.length=0;}
function restart(){
 clearEffects();motionHistory.clear();cargoPending=false;for(const resident of town.residents){const a=resident.attachment??resident;if(a.detached){town.root.add(a.object);a.detached=false;}a.object.visible=true;}
 state.lanterns.clear();Object.assign(state,{playing:false,paused:false,phase:'intro',mode:'swim',blend:0,time:0,elapsed:0,x:0,z:7,y:SWIM_Y,yaw:Math.PI/2,speed:0,target:null,rescued:0,bumps:0});
 flightHint=dockHint=false;impactUntil=0;gangway.visible=false;for(const c of world.collectibles??[])(c.group??c.mesh).visible=true;
 for(const id of ['finish','goal','bottom','progress','paused','map'])$(id).classList.add('hidden');$('intro').classList.remove('hidden');$('swim').classList.add('active');$('fly').classList.remove('active');$('people').textContent='● ● ● ● ● ● ● ●';$('objective').textContent='Bring eight little lives to the lighthouse.';update(0);render();event('restart');
}
function setTarget(x,z){if(!state.playing||state.phase!=='voyage')return;lastInteraction=performance.now();state.target={x:clamp(x,-13,13),z:clamp(z,-22,14)};if(state.target.z<-18&&Math.abs(state.target.x)<5)state.target={...destination};}
function aimedPoint(e){pointer.set(e.clientX/innerWidth*2-1,1-e.clientY/innerHeight*2);ray.setFromCamera(pointer,camera);return ray.ray.intersectPlane(seaPlane,new THREE.Vector3());}
function steer(e){if(!pointerDown||e.target!==renderer.domElement)return;const point=aimedPoint(e);if(point)setTarget(point.x,point.z);}
function updateResidents(dt){
 for(let i=0;i<town.residents.length;i++){
  const resident=town.residents[i],a=resident.attachment??resident;if(a.detached)continue;
  if(!a.homeAnchor)a.homeAnchor=[...a.anchor];
  a.anchor[0]=a.homeAnchor[0]+Math.sin(state.time*.6+i*1.7)*.25;a.anchor[2]=a.homeAnchor[2];
 }
}
function addLooseCargo(){
 const mesh=new THREE.Mesh(new THREE.BoxGeometry(.27,.27,.27),mat(0xbc8a5d));const released=motionHistory.release('cargo',{kick:[0,1.8,0]});
 mesh.position.set(...released.position);scene.add(mesh);loose.push({mesh,v:new THREE.Vector3(...released.velocity),life:3});event('cargo-released',released);
}
function collision(previousX,previousZ,previousYaw){
 if(state.y<2.9){
  // Approximate gameplay capsules follow the deformed street's centerline.
  // The ocean cliffs and the town's ornamental fins are not triangle colliders.
  const centers=[-3,-1,1,3].map(x=>rig.field([x,0,0],state.time,state.blend));
  const clearance=(cx,cz,yaw,reef)=>{
   const c=Math.cos(yaw),s=Math.sin(yaw),points=centers.map(p=>[cx+c*p[0]+s*p[2],cz-s*p[0]+c*p[2]]);let d=Infinity;
   for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],dx=b[0]-a[0],dz=b[1]-a[1],u=clamp(((reef.x-a[0])*dx+(reef.z-a[1])*dz)/(dx*dx+dz*dz),0,1);d=Math.min(d,Math.hypot(reef.x-a[0]-u*dx,reef.z-a[1]-u*dz));}
   return d-reef.r-2;
  };
  for(const reef of world.reefs??[]){const next=clearance(state.x,state.z,state.yaw,reef),before=clearance(previousX,previousZ,previousYaw,reef);if(next<0&&next<=before+1e-6){state.x=previousX;state.z=previousZ;state.yaw=previousYaw;state.speed*=.1;if(state.time>impactUntil){impactUntil=state.time+2;state.bumps++;showToast('A little scrape. Steer around the reefs—or fly.');cargoPending=true;note(170,.5);event('reef-impact');}return;}}
 }
 if(state.y<3.7&&Math.abs(state.z+10)<4.6&&Math.abs(state.x)<15){state.x=previousX;state.z=previousZ;state.speed=0;if(state.time>impactUntil){impactUntil=state.time+2;showToast('Borrow the moth’s motion to cross the sea wall.');note(250,.3);}}
}
function update(dt){
 state.time+=dt;if(state.playing&&state.phase==='voyage')state.elapsed+=dt;
 state.blend=THREE.MathUtils.damp(state.blend,state.mode==='fly'?1:0,2.6,dt);
 state.y=THREE.MathUtils.damp(state.y,state.mode==='fly'?5.5:SWIM_Y,2,dt);
 if(state.phase==='intro')state.y=SWIM_Y+Math.sin(state.time)*.06;
 if(state.playing&&state.phase==='voyage'){
  const dir=new THREE.Vector3((keys.has('d')||keys.has('ArrowRight')?1:0)-(keys.has('a')||keys.has('ArrowLeft')?1:0),0,(keys.has('s')||keys.has('ArrowDown')?1:0)-(keys.has('w')||keys.has('ArrowUp')?1:0));
  if(dir.lengthSq()){dir.normalize();state.target={x:state.x+dir.x*5,z:state.z+dir.z*5};lastInteraction=performance.now();}
  let wantedSpeed=0;const previousYaw=state.yaw;
  if(state.target){const dx=state.target.x-state.x,dz=state.target.z-state.z,distance=Math.hypot(dx,dz);if(distance>.25){const wanted=Math.atan2(-dz,dx),delta=Math.atan2(Math.sin(wanted-state.yaw),Math.cos(wanted-state.yaw));state.yaw+=clamp(delta,-dt*2.4,dt*2.4);wantedSpeed=Math.min(state.mode==='fly'?4.2:3.1,distance*1.25)*Math.max(.2,Math.cos(delta));}}
  state.speed=THREE.MathUtils.damp(state.speed,wantedSpeed,3,dt);
  const px=state.x,pz=state.z;state.x+=Math.cos(state.yaw)*state.speed*dt;state.z-=Math.sin(state.yaw)*state.speed*dt;collision(px,pz,previousYaw);
  if(state.z<-2&&!flightHint){flightHint=true;$('objective').textContent='The sea wall is ahead. Borrow a moth’s wingbeat.';showToast('Try Moth · fly. Even houses can have wings.',4);}
  if(state.z<-15&&!dockHint){dockHint=true;$('objective').textContent='Return to the water and dock at the lighthouse.';showToast('Swim again, then steer into the lighthouse’s golden ring.',4);}
  for(const c of world.collectibles??[]){if(!state.lanterns.has(c.id)&&Math.hypot(state.x-c.x,state.z-c.z)<2.2){state.lanterns.add(c.id);(c.group??c.mesh).visible=false;particleBurst(new THREE.Vector3(c.x,state.y+1,c.z));note(660+state.lanterns.size*110);event('lantern',{id:c.id});}}
  const dd=Math.hypot(state.x-destination.x,state.z-destination.z);
  if(dd<2.15&&state.mode==='swim'&&state.y<.75){state.phase='docking';state.target=null;state.speed=0;dockStarted=state.time;$('objective').textContent='Home is a little closer.';showToast('You made it. Everyone ashore.',3);note(880,.8);event('docking');}
 }
 if(state.phase==='docking'){
  state.x=THREE.MathUtils.damp(state.x,destination.x,2,dt);state.z=THREE.MathUtils.damp(state.z,destination.z,2,dt);state.yaw=THREE.MathUtils.damp(state.yaw,Math.PI/2,2,dt);
 }
 town.root.position.set(state.x,state.y,state.z);town.root.rotation.y=state.yaw;updateResidents(dt);rig.update(state.time,state.blend);
 motionHistory.record('cargo',rig.worldPoint(cargoAnchor,state.time,state.blend),state.time);if(cargoPending){cargoPending=false;addLooseCargo();}
 if(state.phase==='docking'||state.phase==='done'){
  const from=rig.worldPoint([4.1,.15,0],state.time,state.blend),to=new THREE.Vector3(0,.3,-22.2),mid=from.clone().add(to).multiplyScalar(.5),direction=to.clone().sub(from);
  gangway.visible=true;gangway.position.copy(mid);gangway.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),direction.clone().normalize());gangway.scale.set(.75,.09,direction.length());
  const t=state.time-dockStarted;
  for(let i=0;i<town.residents.length;i++){
   const r=town.residents[i],a=r.attachment??r,progress=clamp((t-i*.5)/3.5,0,1);
   if(progress>0&&!a.detached){scene.attach(a.object);a.detached=true;a.departure=a.object.position.clone();}
   if(a.detached){const along=progress<.55?a.departure.clone().lerp(from,progress/.55):from.clone().lerp(to,(progress-.55)/.45);a.object.position.copy(along);a.object.position.y+=.025*Math.sin(state.time*14+i);a.object.quaternion.identity();if(progress===1)a.object.visible=false;}
  }
  state.rescued=Math.min(8,Math.max(0,Math.floor((t-3.5)/.5)+1));$('people').textContent=Array.from({length:8},(_,i)=>i<state.rescued?'✦':'●').join(' ');
  if(t>8&&state.phase!=='done'){state.rescued=8;state.phase='done';$('finish').classList.remove('hidden');$('bottom').classList.add('hidden');$('result').textContent=`Everyone is safe. ${Math.floor(state.elapsed)} seconds, ${state.lanterns.size} lanterns, ${state.bumps} little scrapes. Try a different route or a different rhythm.`;particleBurst(to,0xffdf91,40);note(1100,.8);event('win',{elapsed:state.elapsed,lanterns:state.lanterns.size,bumps:state.bumps});}
 }
 world.update?.(state.time);
 fish.position.set(state.x+Math.cos(state.time*.4)*6,-.5,state.z+Math.sin(state.time*.4)*3);fish.rotation.y=state.time*.4+Math.PI/2;
 moth.position.set(state.x-5,4.8+Math.sin(state.time*1.5)*.4,state.z-3);moth.rotation.y=-state.time*.3;
 for(const part of sourceParts){const positions=part.mesh.geometry.attributes.position;for(let i=0;i<positions.count;i++)positions.setXYZ(i,...samplePoint(clips[part.kind],state.time,[part.rest[3*i],part.rest[3*i+1],part.rest[3*i+2]]));positions.needsUpdate=true;part.mesh.geometry.computeVertexNormals();part.mesh.frustumCulled=false;}
 for(let i=sparks.length-1;i>=0;i--){const p=sparks[i];p.life-=dt;p.v.y-=dt*.7;p.mesh.position.addScaledVector(p.v,dt);p.mesh.material.opacity=Math.max(0,p.life/1.4);if(p.life<=0){scene.remove(p.mesh);p.mesh.geometry.dispose();p.mesh.material.dispose();sparks.splice(i,1);}}
 for(let i=loose.length-1;i>=0;i--){const p=loose[i];p.life-=dt;p.v.y-=dt*5;p.mesh.position.addScaledVector(p.v,dt);p.mesh.rotation.x+=dt*2;p.mesh.rotation.z+=dt;if(p.mesh.position.y<-.6)p.mesh.position.y=-.6;if(p.life<=0){scene.remove(p.mesh);p.mesh.geometry.dispose();p.mesh.material.dispose();loose.splice(i,1);}}
 if(state.target){ring.visible=true;ring.position.set(state.target.x,-.3,state.target.z);ring.rotation.z=state.time*.25;}else ring.visible=false;
 if(state.time>toastUntil)$('toast').classList.remove('visible');
 $('lanterns').textContent=Array.from({length:3},(_,i)=>i<state.lanterns.size?'✦':'○').join('  ');$('elapsed').textContent=Math.floor(state.elapsed/60)+':'+String(Math.floor(state.elapsed%60)).padStart(2,'0');
 $('map-ship').setAttribute('transform',`translate(${state.x} ${state.z}) rotate(${-state.yaw*180/Math.PI})`);for(const c of world.collectibles){const mark=$('map-'+c.id);mark.setAttribute('cx',c.x);mark.setAttribute('cy',c.z);mark.classList.toggle('collected',state.lanterns.has(c.id));}
 const now=performance.now();if(now>manualCameraUntil){const intro=state.phase==='intro',focus=intro?new THREE.Vector3(state.x-2.7,state.y+1.1,state.z+.56):new THREE.Vector3(state.x,state.y+1.1,state.z-1.5),offset=intro?new THREE.Vector3(9,4.8,12):new THREE.Vector3(9,10.5,13);camera.position.lerp(focus.clone().add(offset),dt?1-Math.exp(-dt*2):1);controls.target.lerp(focus,dt?1-Math.exp(-dt*3):1);}
 controls.update();
}
function render(){if(disposed||!composer)return;const begin=performance.now();composer.render();evidence.frames++;evidence.maxFrameMs=Math.max(evidence.maxFrameMs,performance.now()-begin);}
function loop(now){raf=0;if(disposed||state.paused||document.hidden)return;const dt=clamp((now-last)/1000,0,.035);last=now;try{update(dt);render();}catch(e){fail(e);return;}if(now-lastInteraction>90000){pause();return;}if(state.phase!=='intro')schedule();}
function schedule(){if(!raf&&!disposed&&!state.paused&&!document.hidden)raf=requestAnimationFrame(loop);}
async function init(){
 try{
  renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'default'});renderer.setPixelRatio(Math.min(devicePixelRatio,1.25));renderer.setSize(innerWidth,innerHeight);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.95;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;document.body.prepend(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();if(!disposed)fail(Error('Graphics context was lost. Reload to continue.'));});
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(43,innerWidth/innerHeight,.1,250);world=createWorld(scene);town=createTown();addFins(town);scene.add(town.root);rig=createMotionRig(town,clips);
  if(town.residents.length<8)throw Error('The town needs eight residents');
  pmrem=new THREE.PMREMGenerator(renderer);const room=new RoomEnvironment();environment=pmrem.fromScene(room,.08);scene.environment=environment.texture;scene.environmentIntensity=.28;room.dispose();pmrem.dispose();
  controls=new OrbitControls(camera,renderer.domElement);controls.enablePan=false;controls.enableDamping=false;controls.minDistance=9;controls.maxDistance=38;controls.maxPolarAngle=Math.PI*.46;controls.minPolarAngle=.25;controls.mouseButtons.LEFT=null;controls.mouseButtons.RIGHT=THREE.MOUSE.ROTATE;controls.addEventListener('start',()=>{manualCameraUntil=performance.now()+5000;lastInteraction=performance.now();});controls.addEventListener('change',()=>{if(state.phase==='intro')render();});
  composer=new EffectComposer(renderer);composer.addPass(new RenderPass(scene,camera));composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),.18,.5,1.4));composer.addPass(new OutputPass());
  fish=buildCreature('swim');moth=buildCreature('fly');
  ring=new THREE.Mesh(new THREE.RingGeometry(.48,.52,48),new THREE.MeshBasicMaterial({color:0xffdf9c,transparent:true,opacity:.8,side:THREE.DoubleSide,depthWrite:false}));ring.rotation.x=-Math.PI/2;scene.add(ring);
  gangway=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),mat(0xc6b18b));gangway.visible=false;gangway.receiveShadow=true;scene.add(gangway);
  update(0);render();evidence.ready=true;$('begin').disabled=false;$('begin').textContent='Borrow the koi’s motion';event('ready',{revision:THREE.REVISION,houses:town.attachments.length,roadVertices:town.surfaces.reduce((n,s)=>n+s.mesh.geometry.attributes.position.count,0),renderer:'WebGL2',source:'authored sampled motion clips'});
 }catch(e){fail(e);$('begin').textContent='Could not open the world';}
}
$('begin').onclick=start;$('swim').onclick=()=>mode('swim');$('fly').onclick=()=>mode('fly');$('pause').onclick=pause;$('resume').onclick=resume;$('restart').onclick=restart;$('again').onclick=()=>{restart();start();};$('sound').onclick=()=>{sound=!sound;$('sound').textContent=sound?'Sound on':'Sound off';if(sound)note(660);};
$('chart').addEventListener('pointerdown',e=>{const rect=$('chart').getBoundingClientRect();setTarget(-17+(e.clientX-rect.left)/rect.width*34,-26+(e.clientY-rect.top)/rect.height*42);});
window.addEventListener('pointerdown',e=>{if(e.button!==0||e.target!==renderer?.domElement)return;pointerDown=true;lastInteraction=performance.now();pointer.set(e.clientX/innerWidth*2-1,1-e.clientY/innerHeight*2);ray.setFromCamera(pointer,camera);if(ray.intersectObject(fish,true).length){if(state.phase==='intro')start();else mode('swim');return;}if(ray.intersectObject(moth,true).length){mode('fly');return;}steer(e);});window.addEventListener('pointermove',steer);window.addEventListener('pointerup',()=>pointerDown=false);
window.addEventListener('keydown',e=>{if(['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code))e.preventDefault();keys.add(e.key);lastInteraction=performance.now();if(e.repeat)return;if(e.code==='Space')mode(state.mode==='swim'?'fly':'swim');if(e.key==='1')mode('swim');if(e.key==='2')mode('fly');if(e.key.toLowerCase()==='r')restart();if(e.key==='Escape')state.paused?resume():pause();});window.addEventListener('keyup',e=>keys.delete(e.key));
window.addEventListener('resize',()=>{if(!renderer||disposed)return;camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);composer.setSize(innerWidth,innerHeight);render();});document.addEventListener('visibilitychange',()=>{if(document.hidden){keys.clear();pause();}});
function dispose(){if(disposed)return;disposed=true;cancelAnimationFrame(raf);controls?.dispose();world?.dispose();town?.dispose();environment?.dispose();clearEffects();scene?.traverse(o=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose();});composer?.dispose();renderer?.dispose();renderer?.forceContextLoss();audio?.close();evidence.disposed=true;}
window.addEventListener('pagehide',dispose);
window.carryCity={evidence,state,start,mode,setTarget,pause,resume,restart,dispose,advance(seconds,{renderEvery=10}={}){if(seconds<0||seconds>30)throw Error('Bounded test step');cancelAnimationFrame(raf);raf=0;const steps=Math.ceil(seconds*60);for(let i=0;i<steps;i++){update(seconds/steps);if(i%renderEvery===0)render();}render();return {phase:state.phase,x:state.x,y:state.y,z:state.z,mode:state.mode,rescued:state.rescued,elapsed:state.elapsed};},project(x,y,z){const p=new THREE.Vector3(x,y,z).project(camera);return{x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2};}};
init();

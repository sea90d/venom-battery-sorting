import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js";
import { BASE_HEIGHT, baseToScene, cadToScene } from "./scene-geometry.mjs";
import { validatePlan, samplePlan } from "./plan-player.mjs";
import { sampleJoints, validateJointPlayback } from "./joint-playback.mjs";

const MM = 1000;
const COLORS = {
  red: 0xef6262,
  yellow: 0xf3c85b,
  blue: 0x5e9df6,
  green: 0x58d59a,
};
const COLOR_NAMES = { red: "红色", yellow: "黄色", blue: "蓝色", green: "绿色" };
const SCENE_BASE_Y = BASE_HEIGHT;
// Supplied CAD handle points along negative Z; it must become scene +Y.
const BATTERY_MODEL_URL = "/tmp/pdfs/battery.stl";

const ui = {
  viewport: document.querySelector("#viewport"),
  loading: document.querySelector("#loading"),
  taskMode: document.querySelector("#taskMode"),
  basicColor: document.querySelector("#basicColor"),
  basicColorField: document.querySelector("#basicColorField"),
  sequenceOrder: document.querySelector("#sequenceOrder"),
  sequenceColorField: document.querySelector("#sequenceColorField"),
  rebuild: document.querySelector("#rebuildButton"),
  play: document.querySelector("#playButton"),
  pause: document.querySelector("#pauseButton"),
  step: document.querySelector("#stepButton"),
  reset: document.querySelector("#resetButton"),
  timeline: document.querySelector("#timeline"),
  time: document.querySelector("#timeLabel"),
  duration: document.querySelector("#durationLabel"),
  counter: document.querySelector("#stepCounter"),
  phase: document.querySelector("#phaseLabel"),
  action: document.querySelector("#currentAction"),
  currentPhase: document.querySelector("#currentPhase"),
  tcpX: document.querySelector("#tcpX"),
  tcpY: document.querySelector("#tcpY"),
  tcpZ: document.querySelector("#tcpZ"),
  roll: document.querySelector("#toolRoll"),
  gripper: document.querySelector("#gripperState"),
  feedback: document.querySelector("#objectFeedback"),
  sceneLabel: document.querySelector("#sceneModeLabel"),
  stepList: document.querySelector("#stepList"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b111b);
scene.fog = new THREE.Fog(0x0b111b, 1.6, 3.8);
const camera = new THREE.PerspectiveCamera(44, 1, 0.01, 10);
camera.position.set(0, 1.45, 0.55);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
ui.viewport.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.27, -0.21);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2 - 0.06;
controls.minDistance = 0.55;
controls.maxDistance = 2.6;

scene.add(new THREE.HemisphereLight(0xc9e7ff, 0x111827, 1.8));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
keyLight.position.set(0.8, 1.6, 0.6);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x4fd8ed, 1.8, 2.6);
rimLight.position.set(-0.8, 0.7, -0.8);
scene.add(rimLight);

const world = new THREE.Group();
scene.add(world);
const robotGroup = new THREE.Group();
world.add(robotGroup);
const targetGroup = new THREE.Group();
world.add(targetGroup);
const batteryGroup = new THREE.Group();
world.add(batteryGroup);
const trajectoryGroup = new THREE.Group();
world.add(trajectoryGroup);

const table = new THREE.Mesh(
  new THREE.BoxGeometry(1.65, 0.055, 1.25),
  new THREE.MeshStandardMaterial({ color: 0x263244, roughness: 0.85, metalness: 0.05 }),
);
table.position.set(0.03, SCENE_BASE_Y - 0.028, -0.08);
table.receiveShadow = true;
world.add(table);
const grid = new THREE.GridHelper(1.5, 30, 0x36516d, 0x203044);
grid.position.y = SCENE_BASE_Y + 0.002;
grid.position.z = -0.08;
grid.material.transparent = true;
grid.material.opacity = 0.42;
world.add(grid);

function robotPoint(point) {
  return new THREE.Vector3(...baseToScene(point));
}

function makeLabel(text, color = "#f3c85b") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 72;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(6, 12, 20, .88)";
  ctx.roundRect(4, 4, 248, 64, 12, 12);
  ctx.fill();
  ctx.font = "700 28px Arial";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 37);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(0.16, 0.045, 1);
  return sprite;
}

function addTarget(name, xy, kind, definition=null) {
  const [x, y] = xy;
  const pos = robotPoint([x, y, (definition?.pose[2] ?? 0)+0.004]);
  const group = new THREE.Group();
  group.position.copy(pos);
  const outer = kind === "T0" ? new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.006, 0.10), new THREE.MeshStandardMaterial({ color: 0x9d793c, transparent: true, opacity: 0.56 })) : new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.006, 64), new THREE.MeshStandardMaterial({ color: 0x9d793c, transparent: true, opacity: 0.52 }));
  outer.rotation.x = kind === "T0" ? 0 : 0;
  group.add(outer);
  if(definition?.outline_base_xy){
    outer.visible=false;
    const points=definition.outline_base_xy.map(p=>robotPoint([...p,definition.pose[2]+.002]).sub(pos));
    group.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xf3c85b})));
  }
  if (kind !== "T0") {
    const inner = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.008, 64), new THREE.MeshBasicMaterial({ color: 0x151d2a, transparent: true, opacity: 0.8 }));
    inner.position.y = 0.004;
    group.add(inner);
  }
  const label = makeLabel(name);
  label.scale.set(0.060, 0.017, 1);
  label.position.set(0, 0.012, -0.065);
  group.add(label);
  targetGroup.add(group);
  return group;
}

let targetObjects = {};

function makeBattery(color, index) {
  const group = new THREE.Group();
  // Keep a small fallback until the supplied STL has loaded.  It is removed
  // as soon as the real mesh is ready, so the simulation never shows two
  // batteries at once during an asynchronous browser load.
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.078, 0.04), new THREE.MeshStandardMaterial({ color: COLORS[color], roughness: 0.42, metalness: 0.12 }));
  body.castShadow = true;
  group.add(body);
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.053, 0.005, 0.026), new THREE.MeshStandardMaterial({ color: 0xf1f4f7, roughness: 0.5 }));
  top.position.y = 0.042;
  group.add(top);
  const handle = new THREE.Mesh(new THREE.BoxGeometry(0.052, 0.018, 0.008), new THREE.MeshStandardMaterial({ color: 0xdbe2ea, metalness: 0.6, roughness: 0.3 }));
  handle.position.set(0, 0.055, 0);
  group.add(handle);
  group.userData = { color, index, held: false, target: null, fallback: [body, top, handle], model: null };
  batteryGroup.add(group);
  return group;
}

const batteryObjects = Object.fromEntries(Object.keys(COLORS).map((color, i) => [color, makeBattery(color, i)]));

function loadSuppliedBatteryModel() {
  const loader = new STLLoader();
  return new Promise((resolve, reject) => {
  loader.load(
    BATTERY_MODEL_URL,
    (geometry) => {
      const vertices = geometry.getAttribute("position");
      for (let i = 0; i < vertices.count; i += 1) {
        vertices.setXYZ(i, ...cadToScene([vertices.getX(i), vertices.getY(i), vertices.getZ(i)]));
      }
      vertices.needsUpdate = true;
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      for (const [color, object] of Object.entries(batteryObjects)) {
        const material = new THREE.MeshStandardMaterial({
          color: COLORS[color],
          roughness: 0.34,
          metalness: 0.18,
        });
        const mesh = new THREE.Mesh(geometry.clone(), material);
        // STL coordinates are millimetres and the supplied model is
        // approximately 70 x 80 x 80 mm.
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        object.userData.fallback.forEach((part) => { part.visible = false; });
        object.userData.model = mesh;
        object.add(mesh);
      }
      document.querySelector("#modelStatus").textContent = "原始 CAD · 把手朝上 · 70 × 80 × 80 mm";
      resolve();
    },
    undefined,
    (error) => {
      // The procedural fallback remains visible and the rest of the
      // simulation stays usable if a local static server cannot expose STL.
      console.warn("无法加载提供的 battery.stl，继续使用简化后备模型", error);
      document.querySelector("#modelStatus").textContent = "CAD 加载失败，当前为简化模型";
      reject(new Error("CAD 加载失败，请检查本地静态服务器"));
    },
  );
  });
}

function createRobot() {
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.08, 0.035, 48), new THREE.MeshStandardMaterial({ color: 0xbac4d0, metalness: 0.62, roughness: 0.3 }));
  base.position.copy(robotPoint([0, 0, 0.0175]));
  base.castShadow = true;
  robotGroup.add(base);
  const baseRing = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.003, 48), new THREE.MeshStandardMaterial({ color: 0x425064, metalness: 0.7 }));
  baseRing.position.copy(robotPoint([0, 0, 0.0015]));
  robotGroup.add(baseRing);
  const links = [];
  for (let i = 0; i < 6; i += 1) {
    const material = new THREE.MeshStandardMaterial({ color: i % 2 ? 0x9da9b8 : 0xd2d9e1, metalness: 0.72, roughness: 0.28 });
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(i < 2 ? 0.045 : 0.035, i < 2 ? 0.05 : 0.04, 1, 20), material);
    mesh.castShadow = true;
    robotGroup.add(mesh);
    links.push(mesh);
  }
  const joints = [];
  for (let i = 0; i < 6; i += 1) {
    const joint = new THREE.Mesh(new THREE.SphereGeometry(i < 3 ? 0.045 : 0.024, 24, 16), new THREE.MeshStandardMaterial({ color: 0x526176, metalness: 0.8, roughness: 0.25 }));
    joint.castShadow = true;
    robotGroup.add(joint);
    joints.push(joint);
  }
  const tool = new THREE.Group();
  const baseEnvelope=new THREE.Group();
  const baseMaterial=new THREE.MeshStandardMaterial({color:0x98a9b9,transparent:true,opacity:.25});
  baseEnvelope.add(new THREE.Mesh(new THREE.CylinderGeometry(.1,.1,.035,40),baseMaterial));
  baseEnvelope.children[0].position.copy(robotPoint([0,0,.0175]));
  for(const z of [0,.035]){const cap=new THREE.Mesh(new THREE.SphereGeometry(.1,32,20),baseMaterial);cap.position.copy(robotPoint([0,0,z]));baseEnvelope.add(cap);}
  baseEnvelope.visible=false;
  robotGroup.add(baseEnvelope);
  const envelope = new THREE.Mesh(new THREE.CylinderGeometry(1,1,1,48),new THREE.MeshStandardMaterial({color:0x34d7e8,transparent:true,opacity:.48,roughness:.4}));
  envelope.visible=false;
  robotGroup.add(envelope);
  const wrist = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.09, 24), new THREE.MeshStandardMaterial({ color: 0x9aa8b9, metalness: 0.7, roughness: 0.25 }));
  wrist.rotation.z = Math.PI / 2;
  wrist.position.y = 0.10;
  tool.add(wrist);
  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.045, 0.08), new THREE.MeshStandardMaterial({ color: 0x667589, metalness: 0.4, roughness: 0.35 }));
  palm.position.y = 0.057;
  tool.add(palm);
  const jaws = [];
  for (const side of [-1, 1]) {
    const jaw = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.07, 0.065), new THREE.MeshStandardMaterial({ color: 0xcbd3dc, metalness: 0.68, roughness: 0.25 }));
    jaw.position.x = side * 0.027;
    jaw.position.y = 0.005;
    jaw.castShadow = true;
    tool.add(jaw);
    jaws.push(jaw);
  }
  robotGroup.add(tool);
  return { links, joints, tool, jaws, envelope, baseEnvelope };
}

const robotVisual = createRobot();

const DH_A = [0, -0.425, -0.3922, 0, 0, 0];
const DH_D = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823];
const DH_ALPHA = [Math.PI / 2, 0, 0, Math.PI / 2, -Math.PI / 2, 0];

function dhMatrix(theta, d, a, alpha) {
  const c = Math.cos(theta); const s = Math.sin(theta); const ca = Math.cos(alpha); const sa = Math.sin(alpha);
  return new THREE.Matrix4().set(c, -s * ca, s * sa, a * c, s, c * ca, -c * sa, a * s, 0, sa, ca, d, 0, 0, 0, 1);
}

function fk(q) {
  const origin = new THREE.Matrix4();
  const points = [new THREE.Vector3(0, 0, 0)];
  const matrices = [];
  for (let i = 0; i < 6; i += 1) {
    origin.multiply(dhMatrix(q[i], DH_D[i], DH_A[i], DH_ALPHA[i]));
    matrices.push(origin.clone());
    points.push(new THREE.Vector3().setFromMatrixPosition(origin));
  }
  return { points, matrix: origin, matrices };
}

function solve3(matrix, vector) {
  const m = matrix;
  const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (Math.abs(det) < 1e-10) return [0, 0, 0];
  const inv = [
    (m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
    (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[3] * m[2] - m[0] * m[5]) / det,
    (m[3] * m[7] - m[4] * m[6]) / det, (m[6] * m[1] - m[0] * m[7]) / det, (m[0] * m[4] - m[3] * m[1]) / det,
  ];
  return [inv[0] * vector[0] + inv[1] * vector[1] + inv[2] * vector[2], inv[3] * vector[0] + inv[4] * vector[1] + inv[5] * vector[2], inv[6] * vector[0] + inv[7] * vector[1] + inv[8] * vector[2]];
}

function ikPosition(target, seed) {
  const q = seed.slice();
  for (let iter = 0; iter < 180; iter += 1) {
    const current = fk(q).points.at(-1);
    const error = [target[0] - current.x, target[1] - current.y, target[2] - current.z];
    const norm = Math.hypot(...error);
    if (norm < 0.0008) break;
    const jac = Array.from({ length: 3 }, () => Array(6).fill(0));
    const delta = 0.0005;
    for (let j = 0; j < 6; j += 1) {
      const probe = q.slice(); probe[j] += delta;
      const p = fk(probe).points.at(-1);
      jac[0][j] = (p.x - current.x) / delta;
      jac[1][j] = (p.y - current.y) / delta;
      jac[2][j] = (p.z - current.z) / delta;
    }
    const lambda = 0.004;
    const jjT = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) for (let k = 0; k < 6; k += 1) jjT[r * 3 + c] += jac[r][k] * jac[c][k];
    jjT[0] += lambda; jjT[4] += lambda; jjT[8] += lambda;
    const correction = solve3(jjT, error);
    for (let j = 0; j < 6; j += 1) {
      let dq = 0;
      for (let r = 0; r < 3; r += 1) dq += jac[r][j] * correction[r];
      q[j] = THREE.MathUtils.clamp(q[j] + dq * 0.72, -Math.PI, Math.PI);
    }
  }
  return q;
}

function setBetween(mesh, start, end, radiusScale = 1) {
  const midpoint = start.clone().add(end).multiplyScalar(0.5);
  const direction = end.clone().sub(start);
  const length = direction.length();
  mesh.position.copy(midpoint);
  mesh.scale.set(radiusScale, length, radiusScale);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
}

function applyRobot(q, roll = 0, collisionGeometry=null) {
  const result = fk(q);
  const points = result.points.map((p) => robotPoint([p.x, p.y, p.z]));
  for (let i = 0; i < robotVisual.links.length; i += 1){
    const mesh=robotVisual.links[i];
    if(collisionGeometry){
      if(!mesh.userData.solidCylinder){mesh.geometry.dispose();mesh.geometry=new THREE.CylinderGeometry(1,1,1,32);mesh.userData.solidCylinder=true;}
      setBetween(mesh,points[i],points[i+1],collisionGeometry.robot_link_radius_m);
    }else{
      if(mesh.userData.solidCylinder){mesh.geometry.dispose();mesh.geometry=new THREE.CylinderGeometry(i<2?.045:.035,i<2?.05:.04,1,20);mesh.userData.solidCylinder=false;}
      setBetween(mesh,points[i],points[i+1],1);
    }
  }
  for (let i = 0; i < robotVisual.joints.length; i += 1){
    const joint=robotVisual.joints[i];joint.position.copy(points[i+1]);
    joint.scale.setScalar(collisionGeometry?collisionGeometry.robot_link_radius_m/(i<3?.045:.024):1);
    joint.visible=!(collisionGeometry&&i===5); // Flat mounting face, not a sphere sticking into the tool.
  }
  let toolPos = robotPoint(state.currentPose);
  robotVisual.envelope.visible=!!collisionGeometry;
  robotVisual.baseEnvelope.visible=!!collisionGeometry;
  robotVisual.tool.visible=!collisionGeometry; // Display exactly the requested solid tool proxy.
  if(collisionGeometry){
    const p=result.points.at(-1).clone();
    const axis=new THREE.Vector3(0,0,1).transformDirection(result.matrix);
    const tcp=p.addScaledVector(axis,collisionGeometry.tool_length_m);
    toolPos=robotPoint([tcp.x,tcp.y,tcp.z]);
    setBetween(robotVisual.envelope,points.at(-1),toolPos,collisionGeometry.tool_radius_m);
  }
  robotVisual.tool.position.copy(toolPos);
  robotVisual.tool.rotation.set(0, roll, 0);
  const open = state.gripperOpen;
  robotVisual.jaws[0].position.x = open ? -0.027 : -0.017;
  robotVisual.jaws[1].position.x = open ? 0.027 : 0.017;
  return toolPos;
}

function createTrajectoryLine(points, color = 0x58d3e8) {
  trajectoryGroup.clear();
  if (points.length < 2) return;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.65 }));
  trajectoryGroup.add(line);
  for (const point of points.filter((_, i) => i % Math.max(3,Math.ceil(points.length/30)) === 0)) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.006, 10, 8), new THREE.MeshBasicMaterial({ color }));
    marker.position.copy(point);
    trajectoryGroup.add(marker);
  }
}

const state = {
  plan: null, time: 0, duration: 0, playing: false, speed: 1,
  q: [0, -1.15, 1.45, -1.2, -1.57, 0],
  currentQ: [0, -1.15, 1.45, -1.2, -1.57, 0],
  currentPose: [0,0,0,0,0,0], gripperOpen: true, sample: null,
};
let displayObjects = {};
let catalog = {};
let importGeneration = 0;
let cadReady = false;
const sceneRotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
  -1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1));
const toolGeometryRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0), -Math.PI/2);
const phaseNames = {move:"运动",open:"张开",close:"合拢",dwell:"等待",initialize:"初始化",check_object:"持物检查",visual_accept:"放置检查"};

function status(message, error=false) {
  const element = document.querySelector("#planStatus");
  element.textContent = message;
  element.classList.toggle("error",error);
}
function clearTrajectories() {
  trajectoryGroup.traverse(obj => { obj.geometry?.dispose(); obj.material?.dispose(); });
  trajectoryGroup.clear();
}
function loadPlan(raw) {
  if(!cadReady)throw new Error("电池模型尚未加载完成");
  // Validate completely before replacing the last good scene.
  const plan = validatePlan(raw);
  validateJointPlayback(plan);
  state.playing=false;
  state.plan=plan;
  state.time=0;
  state.currentQ=state.q.slice();
  state.duration=plan.steps.reduce((s,a)=>s+a.duration_s,0);
  const paperZ=plan.scene?.table_z_m ?? 0;
  table.position.y=SCENE_BASE_Y+paperZ-.028;
  grid.position.y=SCENE_BASE_Y+paperZ+.002;
  for (const obj of Object.values(displayObjects)) batteryGroup.remove(obj);
  displayObjects={};
  for (const definition of plan.objects) {
    const object=batteryObjects[definition.color].clone(true);
    object.visible=true;
    object.scale.set(definition.size_m[0]/.07,definition.size_m[2]/.08,definition.size_m[1]/.08);
    displayObjects[definition.id]=object;
    batteryGroup.add(object);
  }
  targetGroup.traverse(obj=>{obj.geometry?.dispose(); obj.material?.dispose();});
  targetGroup.clear();
  targetObjects={};
  for (const target of plan.targets) {
    targetObjects[target.id]=addTarget(target.id,target.pose.slice(0,2),target.id==="T0"?"T0":"P",target);
  }
  clearTrajectories();
  createTrajectoryLine([plan.initial_pose,...plan.steps.filter(s=>s.kind==="move").flatMap(s=>
    s.joint_samples?.[0]?.tcp_m_rad?s.joint_samples.map(r=>r.tcp_m_rad):[s.end_pose])].map(robotPoint));
  ui.timeline.max=String(state.duration);
  const recorded=plan.source.data_kind==='recorded';
  const trial=plan.source.simulation_variant==='timing_trial';
  ui.duration.textContent=state.duration.toFixed(2)+(recorded?" s（记录区间）":" s（名义）");
  ui.sceneLabel.textContent=plan.source.label;
  document.querySelector("#planKind").textContent=trial?"新提速仿真":recorded?"真实日志":plan.source.data_kind==="demo"?"演示输入":"导入计划";
  document.querySelector("#plannerName").textContent=plan.source.planner;
  document.querySelector("#poseCount").textContent=plan.steps.filter(s=>s.kind==="move").length+" 个";
  document.querySelector("#planHash").textContent=(recorded?plan.source.audit_sha256:plan.source.planner_sha256)?.slice(0,12) ?? "未提供";
  document.querySelector('#planHashLabel').textContent=recorded?'运行日志摘要':'规划器版本摘要';
  document.querySelector('#timingNote').textContent=trial?'速度 0.5 m/s · 加速度 1.0 m/s² · 搬运高度为抓取上方 15 cm · 最后 10 mm 保持 0.04 m/s、0.1 m/s²。 '+plan.source.timing_note+' '+plan.source.geometry:recorded?[plan.source.timing_note ?? '阶段时间来自记录；段内轨迹为插值重建。',plan.source.geometry ?? '', '释放后物块位置按刚体变换推算；中心误差取自记录的视觉验收。'].join(' '):'位置和旋转读取自导入计划；时间为速度/加速度估算，夹爪和检查事件假设成功。';
  const verification=plan.source.verification;
  status((trial?'新提速方案仿真已导入 · 非实测':recorded?'真实运行日志已导入':plan.source.data_kind==="demo"?"Python 生成 · 演示数据":"Python 计划已导入")+" · "+plan.steps.length+" 个动作"+(recorded&&verification?' · 中心误差 '+(verification.center_error_m*1000).toFixed(2)+' mm':''));
  renderStepList();
  framePlan(false);
  updateAtTime(0);
  ui.play.disabled=false;
}
function framePlan(top) {
  if(!state.plan)return;
  const points=[state.plan.initial_pose,...state.plan.steps.map(s=>s.end_pose),[0,0,0]].map(robotPoint);
  const bounds=new THREE.Box3().setFromPoints(points);
  const center=bounds.getCenter(new THREE.Vector3()), size=bounds.getSize(new THREE.Vector3());
  const distance=Math.max(1.25,size.length()*1.9);
  controls.target.copy(center);
  const folded=state.plan.steps.some(s=>s.motion_space==='joint');
  camera.position.copy(center).add(top?new THREE.Vector3(0,distance,.001):folded?
    new THREE.Vector3(-distance*.8,distance*.65,distance*.7):new THREE.Vector3(0,distance*.9,distance*.65));
  controls.maxDistance=Math.max(2.6,distance*2);
  controls.update();
}
function updateAtTime(time) {
  if(!state.plan)return;
  state.time=THREE.MathUtils.clamp(time,0,state.duration);
  const sample=samplePlan(state.plan,state.time);
  state.sample=sample;
  state.currentPose=sample.pose;
  state.gripperOpen=sample.gripperOpen;
  const joints=sampleJoints(state.plan.steps[sample.index],sample.local);
  state.currentQ=joints?joints.q:ikPosition(sample.pose.slice(0,3),state.currentQ);
  applyRobot(state.currentQ,0,joints?state.plan.collision_geometry:null);
  const tcpQ=new THREE.Quaternion(...sample.quaternion);
  if(joints){
    const flangeQ=new THREE.Quaternion().setFromRotationMatrix(fk(state.currentQ).matrix);
    robotVisual.tool.quaternion.copy(sceneRotation).multiply(flangeQ).multiply(toolGeometryRotation);
    robotVisual.envelope.material.color.set(joints.clearance<0?0xff4545:0x34d7e8);
  }else robotVisual.tool.quaternion.copy(sceneRotation).multiply(tcpQ).multiply(toolGeometryRotation);
  document.querySelector('#collisionState').textContent=joints?
    `夹爪 Ø${(state.plan.collision_geometry.tool_radius_m*2000).toFixed(0)} mm / 长 ${(state.plan.collision_geometry.tool_length_m*1000).toFixed(0)} mm · ${joints.conservative?'帧间净空下界':'采样净空'} ${(joints.clearance*1000).toFixed(1)} mm · ${state.plan.collision_check.passed?'该模型全程通过':'该模型存在碰撞或未通过'}`:
    '此计划没有实体碰撞检查，关节为位置逆解示意';
  for(const [id,item] of Object.entries(sample.objects)) {
    const object=displayObjects[id];
    object.position.copy(robotPoint(item.position));
    object.quaternion.copy(sceneRotation).multiply(new THREE.Quaternion(...item.quaternion)).multiply(sceneRotation.clone().invert());
  }
  const step=state.plan.steps[sample.index];
  ui.timeline.value=String(state.time);
  ui.time.textContent=state.time.toFixed(2)+" s";
  ui.counter.textContent=(sample.index+1)+" / "+state.plan.steps.length;
  ui.phase.textContent=phaseNames[step.kind];
  ui.action.textContent=step.label;
  ui.currentPhase.textContent=phaseNames[step.kind];
  [ui.tcpX,ui.tcpY,ui.tcpZ].forEach((el,i)=>el.textContent=(sample.pose[i]*1000).toFixed(3)+" mm");
  ui.roll.textContent=sample.pose.slice(3).map(x=>x.toFixed(4)).join(", ")+" rad";
  ui.gripper.textContent=sample.gripperOpen?"OPEN":"CLOSED";
  ui.gripper.className="state-chip "+(sample.gripperOpen?"open":"closed");
  ui.feedback.textContent=sample.holding?sample.holding+" · 模拟持物":"未持物";
  document.querySelectorAll(".step-item").forEach((el,i)=>el.classList.toggle("active",i===sample.index));
}
function renderStepList() {
  ui.stepList.replaceChildren();
  let elapsed=0;
  state.plan.steps.forEach((step,index)=>{
    const row=document.createElement("div");
    row.className="step-item";
    const info=[String(index+1),step.label,step.kind.toUpperCase()];
    ["step-index","step-name","step-type"].forEach((className,i)=>{
      const element=document.createElement("span");
      element.className=className;
      element.textContent=info[i]; // Imported labels must never be interpreted as HTML.
      row.appendChild(element);
    });
    const start=elapsed;
    elapsed+=step.duration_s;
    row.title=step.end_pose.join(", ")+" · "+step.duration_s.toFixed(3)+" s";
    row.addEventListener("click",()=>{state.playing=false;updateAtTime(start);});
    ui.stepList.appendChild(row);
  });
}
async function loadExample() {
  const request=++importGeneration;
  state.playing=false;
  const task=ui.taskMode.value;
  ui.basicColorField.classList.toggle("hidden",task!=="basic");
  ui.sequenceColorField.classList.toggle("hidden",task!=="sequence");
  const colors=task==="basic"?ui.basicColor.value:ui.sequenceOrder.value;
  try {
    const path=catalog[task+":"+colors];
    if(!path)throw new Error("没有这项 Python 示例，请重新生成计划目录");
    const response=await fetch(path);
    if(!response.ok)throw new Error("计划读取失败："+response.status);
    const data=await response.json();
    if(request===importGeneration)loadPlan(data);
  } catch(error){if(request===importGeneration)status(error.message,true);}
}
ui.taskMode.addEventListener("change",loadExample);
ui.rebuild.addEventListener("click",loadExample);
document.querySelector("#frontView").addEventListener("click",()=>framePlan(false));
document.querySelector("#topView").addEventListener("click",()=>framePlan(true));
document.querySelector("#planFile").addEventListener("change",async event=>{
  const file=event.target.files[0];
  if(!file)return;
  const request=++importGeneration;
  state.playing=false;
  try{
    if(file.size>8*1024*1024)throw new Error("计划文件超过 8 MB");
    const raw=JSON.parse(await file.text());
    if(request===importGeneration)loadPlan(raw);
  }catch(error){if(request===importGeneration)status("导入失败："+error.message,true);}
  event.target.value="";
});
document.querySelector("#importText").addEventListener("click",()=>{
  ++importGeneration;
  state.playing=false;
  try{loadPlan(JSON.parse(document.querySelector("#planText").value));}
  catch(error){status("导入失败："+error.message,true);}
});
async function loadRecorded(path='./plans/recorded-green-grasp.json'){
  const request=++importGeneration;
  state.playing=false;
  try{
    const response=await fetch(path);
    if(!response.ok)throw new Error('现场计划文件不存在，请先从 grasp_lift plan 导出');
    const data=await response.json();
    if(request===importGeneration)loadPlan(data);
  }catch(error){if(request===importGeneration)status(error.message,true);}
}
document.querySelector('#loadRecorded').addEventListener('click',()=>loadRecorded());
document.querySelector('#loadRecordedRed').addEventListener('click',()=>loadRecorded('./plans/recorded-red-t0.json'));
document.querySelector('#loadAcceleratedRed').addEventListener('click',()=>loadRecorded('./plans/accelerated-red-t0.json'));
document.querySelector('#loadCollision').addEventListener('click',()=>loadRecorded('./plans/tool-body-collision.json'));
ui.play.addEventListener("click",()=>{if(state.plan){if(state.time>=state.duration)updateAtTime(0);state.playing=true;}});
ui.pause.addEventListener("click",()=>{state.playing=false;});
ui.reset.addEventListener("click",()=>{state.playing=false;state.currentQ=state.q.slice();updateAtTime(0);});
ui.step.addEventListener("click",()=>{
  if(!state.plan)return;
  state.playing=false;
  let end=0;
  for(const step of state.plan.steps){end+=step.duration_s;if(end>state.time+1e-9){updateAtTime(end);return;}}
  updateAtTime(state.duration);
});
ui.timeline.addEventListener("input",event=>{state.playing=false;updateAtTime(Number(event.target.value));});
document.querySelectorAll("#speedButtons button").forEach(button=>button.addEventListener("click",()=>{
  state.speed=Number(button.dataset.speed);
  document.querySelectorAll("#speedButtons button").forEach(other=>other.classList.toggle("active",other===button));
}));
function resize(){
  camera.aspect=ui.viewport.clientWidth/ui.viewport.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(ui.viewport.clientWidth,ui.viewport.clientHeight,false);
}
window.addEventListener("resize",resize);
function animate(now){
  requestAnimationFrame(animate);
  if(state.playing){
    state.time+=Math.min(.05,(now-(animate.last??now))/1000)*state.speed;
    if(state.time>=state.duration){state.time=state.duration;state.playing=false;}
    updateAtTime(state.time);
  }
  animate.last=now;
  controls.update();
  renderer.render(scene,camera);
}
resize();
requestAnimationFrame(animate);
(async()=>{
  try{
    await loadSuppliedBatteryModel();
    cadReady=true;
    for(const object of Object.values(batteryObjects))object.visible=false;
    const response=await fetch("./plans/catalog.json");
    if(!response.ok)throw new Error("找不到 Python 导出的计划目录");
    catalog=await response.json();
    await loadRecorded('./plans/tool-body-collision.json');
  }catch(error){status(error.message,true);}
  finally{ui.loading.classList.add("hidden");}
})();

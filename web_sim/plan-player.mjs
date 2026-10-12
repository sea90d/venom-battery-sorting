// Pure data playback. Quaternion algebra adapted from the reviewed DeepSeek
// draft; contract validation and deterministic event replay are Codex-owned.
const KINDS=new Set(['move','open','close','dwell','initialize','check_object','visual_accept']);
const COLORS=new Set(['red','blue','yellow','green']);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const requireValue=(ok,message)=>{if(!ok)throw new Error(message);};
const vector=(a,n,name)=>requireValue(Array.isArray(a)&&a.length===n&&a.every(finite),name+' 必须包含 '+n+' 个有限数值');
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
const norm=a=>Math.hypot(...a);
const inverse=q=>[-q[0],-q[1],-q[2],q[3]];
export function quaternionFromRotvec(v){
  const angle=norm(v);
  return angle<1e-12?[0,0,0,1]:[...v.map(x=>x*Math.sin(angle/2)/angle),Math.cos(angle/2)];
}
export function multiply(a,b){
  const [ax,ay,az,aw]=a,[bx,by,bz,bw]=b;
  return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
    aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz];
}
export function rotate(q,v){return multiply(multiply(q,[...v,0]),inverse(q)).slice(0,3);}
export function slerp(a,b,t){
  let d=dot(a,b);
  if(d<0){b=b.map(x=>-x);d=-d;}
  d=Math.min(1,Math.max(-1,d));
  if(d>.999999){const q=a.map((x,i)=>x+(b[i]-x)*t);const n=norm(q);return q.map(x=>x/n);}
  const angle=Math.acos(d),den=Math.sin(angle);
  return a.map((x,i)=>(x*Math.sin((1-t)*angle)+b[i]*Math.sin(t*angle))/den);
}
function rotvec(q){
  if(q[3]<0)q=q.map(v=>-v);
  const n=norm(q.slice(0,3));
  if(n<1e-12)return [0,0,0];
  return q.slice(0,3).map(v=>v*2*Math.atan2(n,q[3])/n);
}
function samePose(a,b){
  return norm(a.slice(0,3).map((v,i)=>v-b[i]))<1e-8 &&
    1-Math.abs(dot(quaternionFromRotvec(a.slice(3)),quaternionFromRotvec(b.slice(3))))<1e-10;
}
const validId=v=>typeof v==='string'&&/^[a-zA-Z0-9_-]{1,64}$/.test(v)&&!['__proto__','constructor','prototype'].includes(v);
export function validatePlan(raw){
  requireValue(raw&&typeof raw==='object'&&!Array.isArray(raw),'计划必须是 JSON 对象');
  requireValue(raw.schema==='ur5-simulation'&&raw.version===1,'不支持的计划格式/版本');
  requireValue(raw.frame==='ur_base'&&raw.units?.position==='m'&&raw.units?.rotation==='rad','要求 UR Base 坐标、米、弧度');
  requireValue(raw.source&&typeof raw.source.label==='string'&&typeof raw.source.planner==='string'&&['demo','configured','recorded'].includes(raw.source.data_kind),'缺少计划来源');
  if(raw.scene)requireValue(finite(raw.scene.table_z_m),'纸面 Base Z 无效');
  vector(raw.initial_pose,6,'initial_pose');
  requireValue(typeof raw.initial_gripper_open==='boolean','缺少夹爪初始状态');
  requireValue(Array.isArray(raw.objects)&&raw.objects.length<=100,'objects 必须是数组（最多 100 项）');
  requireValue(Array.isArray(raw.targets)&&raw.targets.length<=100,'targets 必须是数组（最多 100 项）');
  const ids=new Set(),targets=new Set();
  for(const obj of raw.objects){
    requireValue(obj&&validId(obj.id)&&!ids.has(obj.id),'物块 ID 无效或重复');ids.add(obj.id);
    requireValue(COLORS.has(obj.color),'未知物块颜色');
    vector(obj.center,3,'object.center');vector(obj.quaternion,4,'object.quaternion');vector(obj.size_m,3,'object.size_m');
    requireValue(Math.abs(norm(obj.quaternion)-1)<1e-6,'物块四元数必须归一化');
    requireValue(obj.size_m.every(v=>v>0&&v<2),'物块尺寸无效');
  }
  for(const target of raw.targets){
    requireValue(target&&validId(target.id)&&!targets.has(target.id),'目标 ID 无效或重复');targets.add(target.id);
    vector(target.pose,6,'target.pose');
    if(target.outline_base_xy){
      requireValue(Array.isArray(target.outline_base_xy)&&target.outline_base_xy.length>=3&&target.outline_base_xy.length<=64,'目标轮廓无效');
      target.outline_base_xy.forEach(p=>vector(p,2,'target.outline_base_xy'));
    }
  }
  requireValue(Array.isArray(raw.steps)&&raw.steps.length>0&&raw.steps.length<=10000,'动作列表必须包含 1–10000 项');
  let previous=raw.initial_pose,held=null,total=0;
  for(const [i,s] of raw.steps.entries()){
    requireValue(s&&KINDS.has(s.kind),'不支持的动作类型：'+s?.kind);
    requireValue(typeof s.label==='string'&&s.label.length<=500,'动作名称无效');
    vector(s.start_pose,6,'start_pose');vector(s.end_pose,6,'end_pose');
    requireValue(samePose(previous,s.start_pose),'第 '+(i+1)+' 步起点不连续');
    requireValue(finite(s.duration_s)&&s.duration_s>=0,'动作时间无效');
    requireValue(s.object_id===null||ids.has(s.object_id),'动作引用了未知物块');
    requireValue(s.target_id===null||targets.has(s.target_id),'动作引用了未知目标');
    if(s.kind==='move'){
      if(s.motion_space==='joint'){
        vector(s.joint_start_rad,6,'joint_start_rad');vector(s.joint_end_rad,6,'joint_end_rad');
        requireValue(finite(s.speed_rad_s)&&s.speed_rad_s>0&&finite(s.acceleration_rad_s2)&&s.acceleration_rad_s2>0,'关节运动缺少角速度/角加速度');
      }else requireValue(finite(s.speed_mps)&&s.speed_mps>0&&finite(s.acceleration_mps2)&&s.acceleration_mps2>0,'运动缺少有效速度/加速度');
      requireValue(typeof s.profile==='string','运动缺少速度档');
      requireValue(samePose(s.start_pose,s.end_pose)||s.duration_s>0,'非零运动不能使用零时长');
      const joint=s.motion_space==='joint';
      const distance=joint?Math.max(...s.joint_end_rad.map((v,j)=>Math.abs(v-s.joint_start_rad[j]))):norm(s.end_pose.slice(0,3).map((v,j)=>v-s.start_pose[j]));
      const nominal=translationTiming(distance,joint?s.speed_rad_s:s.speed_mps,joint?s.acceleration_rad_s2:s.acceleration_mps2).duration;
      requireValue(s.duration_s+1e-7>=nominal,'运动时长短于指定速度/加速度允许的时间');
    }else{
      requireValue(samePose(s.start_pose,s.end_pose),'非运动动作改变了位姿');
      requireValue(s.speed_mps===null&&s.acceleration_mps2===null&&s.profile===null,'非运动动作的速度字段必须为 null');
    }
    if(s.kind==='close'){requireValue(s.object_id!==null&&held===null,'夹持动作缺少物块或夹爪已持物');held=s.object_id;}
    if(s.kind==='check_object')requireValue(s.object_id!==null&&held===s.object_id,'持物检查与夹持状态不符');
    if(s.kind==='open'){requireValue(s.object_id===null||held===s.object_id,'释放动作与夹持物块不符');held=null;}
    previous=s.end_pose;total+=s.duration_s;
  }
  requireValue(finite(total)&&total<=86400,'计划总时长无效或超过一天');
  return structuredClone(raw);
}
function translationTiming(distance,speed,acceleration){
  if(distance===0)return {duration:0,ramp:0};
  const ramp=Math.min(speed/acceleration,Math.sqrt(distance/acceleration));
  return {duration:distance>=speed*speed/acceleration?distance/speed+speed/acceleration:2*ramp,ramp};
}
export function motionFraction(step,t){
  t=Math.max(0,Math.min(1,t));
  if(step.kind!=='move'||t===0||t===1)return t;
  const joint=step.motion_space==='joint';
  const d=joint?Math.max(...step.joint_end_rad.map((v,i)=>Math.abs(v-step.joint_start_rad[i]))):norm(step.end_pose.slice(0,3).map((v,i)=>v-step.start_pose[i]));
  if(d<1e-12)return t; // Rotation-only segment: shortest-arc SLERP.
  const timing=translationTiming(d,joint?step.speed_rad_s:step.speed_mps,joint?step.acceleration_rad_s2:step.acceleration_mps2);
  const r=timing.ramp/timing.duration; // Normalize when rotation stretches a segment.
  if(t<r)return t*t/(2*r*(1-r));
  if(t>1-r)return 1-(1-t)*(1-t)/(2*r*(1-r));
  return (t-r/2)/(1-r);
}
function interpolatePose(s,t){
  const f=motionFraction(s,t);
  if(f===0)return s.start_pose.slice();
  if(f===1)return s.end_pose.slice();
  if(s.motion_space==='joint')return poseFromJoints(s.joint_start_rad.map((v,i)=>v+(s.joint_end_rad[i]-v)*f));
  return [...s.start_pose.slice(0,3).map((x,i)=>x+(s.end_pose[i]-x)*f),
    ...rotvec(slerp(quaternionFromRotvec(s.start_pose.slice(3)),quaternionFromRotvec(s.end_pose.slice(3)),f))];
}
// Same nominal UR5 DH and 246 mm active TCP as the collision exporter.
export function poseFromJoints(q){
  const a=[0,-.425,-.3922,0,0,0],d=[.089159,0,0,.10915,.09465,.0823],alpha=[Math.PI/2,0,0,Math.PI/2,-Math.PI/2,0];
  let m=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for(let i=0;i<6;i++){
    const c=Math.cos(q[i]),s=Math.sin(q[i]),ca=Math.cos(alpha[i]),sa=Math.sin(alpha[i]);
    const h=[[c,-s*ca,s*sa,a[i]*c],[s,c*ca,-c*sa,a[i]*s],[0,sa,ca,d[i]],[0,0,0,1]];
    m=m.map(row=>[0,1,2,3].map(j=>row.reduce((sum,v,k)=>sum+v*h[k][j],0)));
  }
  // Convert rotation matrix to a stable quaternion (including pi rotations).
  const trace=m[0][0]+m[1][1]+m[2][2];let r;
  if(trace>0){const z=2*Math.sqrt(trace+1);r=[(m[2][1]-m[1][2])/z,(m[0][2]-m[2][0])/z,(m[1][0]-m[0][1])/z,z/4];}
  else{
    const i=[m[0][0],m[1][1],m[2][2]].indexOf(Math.max(m[0][0],m[1][1],m[2][2])),j=(i+1)%3,k=(i+2)%3;
    const z=2*Math.sqrt(1+m[i][i]-m[j][j]-m[k][k]);r=[0,0,0,0];r[i]=z/4;r[j]=(m[j][i]+m[i][j])/z;r[k]=(m[k][i]+m[i][k])/z;r[3]=(m[k][j]-m[j][k])/z;
  }
  return [...m.slice(0,3).map(row=>row[3]+row[2]*.246),...rotvec(r)];
}
export function samplePlan(plan,time){
  requireValue(finite(time),'播放时间必须有限');
  time=Math.max(0,time);
  const objects=Object.fromEntries(plan.objects.map(o=>[o.id,{position:o.center.slice(),quaternion:o.quaternion.slice()}]));
  let pose=plan.initial_pose.slice(),gripperOpen=plan.initial_gripper_open,holding=null,grasp=null,elapsed=0;
  let index=plan.steps.length-1,local=1;
  const updateHeld=()=>{
    if(holding===null)return;
    const q=quaternionFromRotvec(pose.slice(3));
    objects[holding]={position:rotate(q,grasp.offset).map((v,i)=>v+pose[i]),quaternion:multiply(q,grasp.quaternion)};
  };
  for(let i=0;i<plan.steps.length;i++){
    const s=plan.steps[i],end=elapsed+s.duration_s;
    if(s.duration_s>0&&time<end-1e-10){
      index=i;local=(time-elapsed)/s.duration_s;pose=interpolatePose(s,local);updateHeld();break;
    }
    pose=s.end_pose.slice();updateHeld(); // Freeze latest pose BEFORE release.
    if(s.kind==='close'){
      gripperOpen=false;holding=s.object_id;
      const inv=inverse(quaternionFromRotvec(pose.slice(3))),obj=objects[holding];
      grasp={offset:rotate(inv,obj.position.map((v,j)=>v-pose[j])),quaternion:multiply(inv,obj.quaternion)};
    }
    if(s.kind==='open'){gripperOpen=true;holding=null;grasp=null;}
    elapsed=end;
  }
  return {index,local,pose,quaternion:quaternionFromRotvec(pose.slice(3)),gripperOpen,holding,objects};
}

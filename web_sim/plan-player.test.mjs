import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validatePlan,samplePlan,quaternionFromRotvec,slerp,motionFraction,rotate} from './plan-player.mjs';
const read=p=>JSON.parse(readFileSync(new URL(p,import.meta.url),'utf8'));
const index=read('./plans/catalog.json');
const near=(a,b,eps=1e-8)=>a.forEach((v,i)=>assert(Math.abs(v-b[i])<eps,`${a} != ${b}`));
for(const path of Object.values(index)){
  const plan=validatePlan(read(path));
  let total=0;
  for(const step of plan.steps){
    total+=step.duration_s;
    if(step.kind==='move'){
      near(samplePlan(plan,Math.max(0,total-1e-9)).pose.slice(0,3),step.end_pose.slice(0,3));
    }
  }
  const final=samplePlan(plan,total);
  near(final.pose,plan.steps.at(-1).end_pose);
  assert.equal(final.holding,null);
  assert.equal(final.gripperOpen,true);
  for(const step of plan.steps.filter(s=>s.kind==='open'&&s.object_id)){
    const target=plan.targets.find(t=>t.id===step.target_id);
    near(final.objects[step.object_id].position.slice(0,2),target.pose.slice(0,2));
  }
  const middle=samplePlan(plan,total*.41);
  samplePlan(plan,total*.9);
  assert.deepEqual(samplePlan(plan,total*.41),middle,'Backwards seek must be deterministic');
}
const plan=validatePlan(read('./plans/basic-green.json'));
for(const mutate of [p=>p.units.position='mm',p=>p.steps[0].kind='moveJ',p=>p.steps[3].start_pose[0]+=.1,
  p=>p.steps[0].duration_s=-1,p=>p.initial_pose[0]=NaN,p=>p.steps[0].object_id='missing',
  p=>p.objects[0].quaternion=[0,0,0,0],p=>p.objects[0].id='__proto__',p=>p.steps[3].duration_s=0]){
  const broken=structuredClone(plan);mutate(broken);assert.throws(()=>validatePlan(broken));
}
const rotating=structuredClone(plan);
// A tiny explicit plan tests a 90-degree rigid grasp independently of label parsing.
const origin=[0,0,.3,0,0,0],turned=[.1,0,.3,0,0,Math.PI/2];
rotating.initial_pose=origin;rotating.initial_gripper_open=true;
rotating.objects=[{id:'part',color:'red',center:[.02,0,.3],quaternion:[0,0,0,1],size_m:[.07,.08,.08]}];
const action=(kind,start,end,duration=0,id=null)=>({kind,label:'arbitrary label',start_pose:start,end_pose:end,duration_s:duration,
  speed_mps:kind==='move'?.05:null,acceleration_mps2:kind==='move'?.1:null,profile:kind==='move'?'transit':null,object_id:id,target_id:null});
rotating.steps=[action('close',origin,origin,0,'part'),action('move',origin,turned,5),action('open',turned,turned,0,'part')];
validatePlan(rotating);
near(samplePlan(rotating,0).objects.part.position,[.02,0,.3]);
const after=samplePlan(rotating,5);
near(after.objects.part.position,[.1,.02,.3]);
near(after.objects.part.quaternion,quaternionFromRotvec([0,0,Math.PI/2]));
assert.equal(after.holding,null);
near(slerp([0,0,0,1],[0,0,0,-1],.5),[0,0,0,1]);
assert(motionFraction(rotating.steps[1],.1)<.1);
console.log('PASS: 28 actual Python exports; exact endpoints, release poses, rewind, rigid rotation, malformed imports');
const real=validatePlan(read('./plans/recorded-green-grasp.json'));
assert.equal(real.source.planner,'battery_sorting.grasp_lift');
const moves=real.steps.filter(s=>s.kind==='move');
assert.equal(moves.length,28);
assert.equal(Math.min(...moves.map(s=>s.end_pose[2])),.069207);
for(const m of moves){near(m.start_pose.slice(0,2),m.end_pose.slice(0,2));near(m.start_pose.slice(3),m.end_pose.slice(3));}
const realEnd=samplePlan(real,real.steps.reduce((sum,s)=>sum+s.duration_s,0));
near(realEnd.pose,real.initial_pose);assert.equal(realEnd.holding,'green');
assert.equal(realEnd.gripperOpen,false);
assert(realEnd.objects.green.position[2]>.4);
console.log('PASS: historical calibrated grasp: 14 descent + 14 lift, exact Z, held at finish');
const recorded=validatePlan(read('./plans/recorded-red-t0.json'));
const recordedTotal=recorded.steps.reduce((sum,s)=>sum+s.duration_s,0);
assert.equal(recorded.source.data_kind,'recorded');
assert.equal(recorded.source.timing,'recorded_event_intervals');
assert(Math.abs(recordedTotal-recorded.source.captured_frame_offset_s)<1e-8);
assert(Math.abs(recordedTotal-48.238906)<1e-8);
assert.equal(recorded.steps.filter(s=>s.kind==='move').length,10);
assert.equal(recorded.targets[0].pose[2],recorded.scene.table_z_m);
assert.equal(recorded.targets[0].outline_base_xy.length,4);
const carrying=samplePlan(recorded,25);
assert.equal(carrying.holding,'red');
assert.equal(carrying.gripperOpen,false);
const recordedEnd=samplePlan(recorded,recordedTotal);
near(recordedEnd.pose,recorded.steps.at(-1).end_pose);
assert.equal(recordedEnd.holding,null);
assert.equal(recordedEnd.gripperOpen,true);
for(const obj of recorded.objects.filter(o=>o.id!=='red')){
  near(recordedEnd.objects[obj.id].position,obj.center);
}
assert.deepEqual(samplePlan(recorded,25),carrying,'Recorded rewind must restore the rigid grasp');
for(const mutate of [p=>p.scene.table_z_m=NaN,p=>p.targets[0].outline_base_xy=[[0,0],[1,1]]]){
  const broken=structuredClone(recorded);mutate(broken);assert.throws(()=>validatePlan(broken));
}
console.log('PASS: real red T0 recording: recorded timing, carry, release, home, untouched neighbours and rewind');
const trial=validatePlan(read('./plans/accelerated-red-t0.json'));
assert.equal(trial.source.simulation_variant,'timing_trial');
assert.equal(trial.source.data_kind,'configured');
assert.equal(trial.source.verification,undefined,'Do not present archived acceptance as a new trial result');
const trialTotal=trial.steps.reduce((sum,s)=>sum+s.duration_s,0);
assert(Math.abs(trialTotal-trial.source.predicted_timing.motion_and_stability_s)<1e-8);
assert(trialTotal<20&&trialTotal>10);
const transfer=trial.steps.find(s=>s.label.includes('搬运平面'));
assert.equal(transfer.speed_mps,.5);assert.equal(transfer.acceleration_mps2,1);
assert(Math.abs(transfer.end_pose[2]-.219207)<1e-8);
const redAxis=rotate(trial.objects.find(o=>o.id==='red').quaternion,[1,0,0]);
for(const obj of trial.objects){
  const axis=rotate(obj.quaternion,[1,0,0]);
  assert(Math.abs(axis.reduce((v,x,i)=>v+x*redAxis[i],0))>.99,'Oct11 measured handle axes must agree; do not introduce a synthetic 90-degree swap');
}
const trialEnd=samplePlan(trial,trialTotal);
assert.equal(trialEnd.holding,null);assert.equal(trialEnd.gripperOpen,true);
near(trialEnd.pose,trial.initial_pose);
near(trialEnd.objects.red.quaternion,trial.objects.find(o=>o.id==='red').quaternion);
console.log('PASS: current accelerated cycle: correct dynamics, carrying height, nominal timing, release/home and aligned handle axes');

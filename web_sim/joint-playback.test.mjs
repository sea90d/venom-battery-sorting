import assert from 'node:assert/strict';
import fs from 'node:fs';
import {sampleJoints,validateJointPlayback} from './joint-playback.mjs';
import {validatePlan,samplePlan,poseFromJoints,quaternionFromRotvec} from './plan-player.mjs';

const plan=validatePlan(JSON.parse(fs.readFileSync(new URL('./plans/tool-body-collision.json',import.meta.url),'utf8')));
validateJointPlayback(plan);
assert.equal(plan.collision_check.passed,true);
assert(plan.collision_check.swept_clearance_lower_bound_m>=.01);
assert.equal(plan.collision_geometry.thickness_ratio,1.5);
assert.equal(plan.collision_geometry.tool_length_m,.246);
const duration=plan.steps.reduce((s,v)=>s+v.duration_s,0);
for(let t=0;t<=duration;t+=.02){
  const frame=samplePlan(plan,t);
  const joints=sampleJoints(plan.steps[frame.index],frame.local);
  assert(joints.q.every(Number.isFinite));
  assert(joints.clearance>=.01);
}
const step=plan.steps.find(s=>s.kind==='move'&&s.joint_samples.length>5);
assert.deepEqual(sampleJoints(step,.75),sampleJoints(step,.75),'Seeking must be deterministic');
assert.deepEqual(sampleJoints(step,1).q,step.joint_samples.at(-1).q_rad);
const malformed=structuredClone(plan);malformed.steps[0].joint_samples[0].q_rad[0]=NaN;
assert.throws(()=>validateJointPlayback(malformed));
const discontinuous=structuredClone(plan);discontinuous.steps[1].joint_samples[0].q_rad[0]+=1;
assert.throws(()=>validateJointPlayback(discontinuous));
console.log('PASS: full-pose joint playback, swept solid clearance, forward/seek and malformed data');
const cycle=validatePlan(JSON.parse(fs.readFileSync(new URL('./plans/accelerated-red-t0.json',import.meta.url),'utf8')));
validateJointPlayback(cycle);
assert(cycle.collision_check.passed);
assert(!cycle.steps.some(s=>s.label.startsWith('startup_')));
assert(cycle.steps.every(s=>s.joint_samples));
console.log('PASS: existing accelerated button also replays the audited full-pose joint cycle');
const jointMoves=cycle.steps.filter(s=>s.motion_space==='joint');
assert.equal(jointMoves.length,2);
for(const move of jointMoves){
  for(const row of move.joint_samples){
    const pose=poseFromJoints(row.q_rad);
    assert(Math.hypot(...pose.slice(0,3).map((v,i)=>v-row.tcp_m_rad[i]))<1e-8);
    const a=quaternionFromRotvec(pose.slice(3)),b=quaternionFromRotvec(row.tcp_m_rad.slice(3));
    assert(1-Math.abs(a.reduce((v,x,i)=>v+x*b[i],0))<1e-8);
  }
}
const wrongCommand=structuredClone(cycle);wrongCommand.steps[0].joint_end_rad[0]+=.1;
assert.throws(()=>validateJointPlayback(wrongCommand));
console.log('PASS: folded joint motion TCP follows FK, including rotation, and command/sample mismatches are rejected');

// Replay offline full-pose IK samples, independent of browser seek history.
import {motionFraction} from './plan-player.mjs';

export function sampleJoints(step, local) {
  const rows=step.joint_samples;
  if(!rows)return null;
  const f=motionFraction(step,local);
  if(f>=1)return {q:rows.at(-1).q_rad.slice(), clearance:rows.at(-1).tool_body_clearance_m, conservative:false};
  const index=Math.max(0,rows.findIndex((r,i)=>i+1<rows.length && f>=r.fraction && f<rows[i+1].fraction));
  const a=rows[index],b=rows[index+1];
  const t=(f-a.fraction)/(b.fraction-a.fraction);
  return {q:a.q_rad.map((v,i)=>v+(b.q_rad[i]-v)*t),
    clearance:a.next_swept_clearance_lower_bound_m ?? Math.min(a.tool_body_clearance_m,b.tool_body_clearance_m),
    conservative:a.next_swept_clearance_lower_bound_m!==undefined};
}

export function validateJointPlayback(plan) {
  if(!plan.collision_geometry && !plan.steps.some(s=>s.joint_samples))return;
  const g=plan.collision_geometry;
  if(!plan.collision_check||typeof plan.collision_check.passed!=='boolean')throw new Error('缺少碰撞检查结果');
  if(!g || !Number.isFinite(g.robot_link_radius_m) || g.robot_link_radius_m<=0 || g.robot_link_radius_m>.2 ||
    !Number.isFinite(g.tool_radius_m) || g.tool_radius_m<=0 || g.tool_radius_m>.3 ||
    !Number.isFinite(g.tool_length_m) || g.tool_length_m<=0 || g.tool_length_m>1)
    throw new Error('碰撞几何尺寸无效');
  let previous=null;
  for(const step of plan.steps){
    const rows=step.joint_samples;
    if(!Array.isArray(rows)||rows.length<2||rows.length>10000||rows[0].fraction!==0||rows.at(-1).fraction!==1)
      throw new Error('完整姿态关节采样必须覆盖每步 0–1');
    for(const [i,row] of rows.entries()){
      if(!Number.isFinite(row.fraction)||(i>0&&row.fraction<=rows[i-1].fraction)||
        !Array.isArray(row.q_rad)||row.q_rad.length!==6||!row.q_rad.every(Number.isFinite)||
        !Number.isFinite(row.tool_body_clearance_m)||
        (row.next_swept_clearance_lower_bound_m!==undefined&&!Number.isFinite(row.next_swept_clearance_lower_bound_m)))
        throw new Error('关节或净空采样无效');
    }
    if(previous && previous.some((v,i)=>Math.abs(v-rows[0].q_rad[i])>.01))throw new Error('动作间关节轨迹不连续');
    if(step.motion_space==='joint' && (rows[0].q_rad.some((v,i)=>Math.abs(v-step.joint_start_rad[i])>.01)||
      rows.at(-1).q_rad.some((v,i)=>Math.abs(v-step.joint_end_rad[i])>.01)))throw new Error('关节采样与实际关节命令不一致');
    previous=rows.at(-1).q_rad;
  }
}

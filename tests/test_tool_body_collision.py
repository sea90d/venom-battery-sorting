import json
from pathlib import Path
import numpy as np
import pytest
from tools.tool_body_collision import point_cylinder_distance,segment_cylinder_distance,tool_body_clearances
from tools.audit_t0_trajectory import fk
from tools.simulate_tool_body_collision import check_interval,clearance


def test_finite_tool_includes_flange_start_without_backwards_round_cap():
    assert point_cylinder_distance([0,0,0])==0
    assert point_cylinder_distance([.07,0,.001])==0
    assert point_cylinder_distance([0,0,-.04])==pytest.approx(.04)
    assert point_cylinder_distance([.10,0,.30])==pytest.approx(np.hypot(.025,.054))


def test_segment_cylinder_checks_between_endpoints():
    assert segment_cylinder_distance([-.2,0,.1],[.2,0,.1])==pytest.approx(0,abs=1e-9)
    assert segment_cylinder_distance([-.2,0,-.1],[.2,0,-.1])==pytest.approx(.1)
    assert segment_cylinder_distance([.10,0,0],[.10,0,.246])==pytest.approx(.025)


def test_old_home_intersects_solid_forearm_and_new_home_clears():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'output/trajectory_audit_20261011/report.json').read_text(encoding='utf-8'))
    q=old['segments'][1]['samples'][-1]['q_rad']
    flange,points=fk(q)
    assert tool_body_clearances(flange,points)['forearm']<0
    new=json.loads((root/'web_sim/plans/tool-body-collision.json').read_text(encoding='utf-8'))
    assert new['collision_check']['passed'] is True
    assert new['collision_check']['minimum_clearance_m']>.027
    assert new['collision_check']['swept_clearance_lower_bound_m']>=.01


def test_sweep_bound_subdivides_large_joint_interval():
    first=np.array([.67,-1.97,1.17,-.8,-1.81,.8])
    last=first.copy();last[0]+=.02
    first_gap=clearance(first)[0];last_gap=clearance(last)[0]
    bound,checks,failure=check_interval(first,last,first_gap,last_gap)
    assert checks>0
    assert failure is None
    assert bound>=.01

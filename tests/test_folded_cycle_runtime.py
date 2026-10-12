"""Test the actual folded-cycle callback pose, not a saved simulation export."""
from dataclasses import replace

import pytest

from battery_sorting.closed_loop import ROOT, Setup
from battery_sorting.config import GripperConfig, Pose
from battery_sorting.robot import FakeRobot
from tools import run_smooth_t0 as runner


def test_folded_cycle_verifies_downward_before_returning_sideways(monkeypatch):
    setup=Setup(ROOT/'config/closed_loop.json')
    observation=setup.pick_view
    robot=FakeRobot(observation)
    robot.active_tcp_offset=lambda:setup.raw['active_tcp_offset_m_rad']
    now=[0.]
    class Gripper:
        config=GripperConfig(open_position=1000)
        last_grip_state=1
        def has_object(self): return self.last_grip_state==2
        def position(self): return 230 if self.last_grip_state==2 else self.config.open_position
        def close(self): self.last_grip_state=2; return True
        def open(self): self.last_grip_state=1; return True
    events=[]
    folded=[]
    def fold(selected,device,destination,log,**kwargs):
        assert destination=='home'
        assert device._pose==observation
        assert events[-1]=='verified'
        folded.append(destination)
        device._pose=selected.home
        return selected.home
    def verify():
        assert robot._pose==observation
        assert robot._pose!=setup.home
        assert now[0]>=3.1
        events.append('verified')
    monkeypatch.setattr(runner,'folded_transition',fold)
    pick=Pose(-.495,-.102,setup.site.safe_high_z_m,*observation.as_list()[3:])
    place=replace(pick,x=-.445,y=.223)
    runner.execute_cycle(setup,runner.transfer_setup(setup,None),robot,Gripper(),
        observation,pick,place,lambda *args,**kwargs:None,runner.PROFILES['trajectory-check'],
        clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt),
        cycle_started_at=0.,verify_at_observation=verify)
    assert robot.records[-1].label=='direct_return_home'
    assert robot.records[-1].pose==observation
    assert folded==['home']
    assert robot._pose==setup.home

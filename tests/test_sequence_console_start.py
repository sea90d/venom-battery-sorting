"""The console's clicked start must not wait for hidden terminal input."""
from battery_sorting import closed_loop
from battery_sorting.robot import FakeRobot
import json
import pytest
from battery_sorting.config import Pose


@pytest.mark.parametrize('start',['home','sequence-view'])
def test_clicked_sequence_start_uses_fresh_runtime_without_terminal_input(tmp_path,monkeypatch,start):
    setup=closed_loop.Setup(closed_loop.ROOT/'config/closed_loop_sequence.json')
    robot=FakeRobot(setup.home)
    if start=='sequence-view':
        state=json.loads((closed_loop.ROOT/'output/sequence_start_check_20261011/state.json').read_text(encoding='utf-8'))
        robot._pose=Pose(*state['rtde']['tcp_m_rad'])
        robot._joints=state['rtde']['q_rad']
    robot.connect_readonly=lambda:None
    robot.connect_control=lambda:None
    robot.active_tcp_offset=lambda:setup.raw['active_tcp_offset_m_rad']
    records=[]
    class Camera:
        def start(self,**kwargs):records.append(('camera_start',kwargs))
        def configure_color(self,settings):pass
        def stop(self):pass
    class Gripper:
        last_grip_state=1
        def has_object(self):return False
        def connect(self):pass
        def disconnect(self):pass
    class Runtime:
        def __init__(self,selected,camera,robot,gripper,output):
            self.setup=selected
        def log(self,event,**detail):records.append((event,detail))
        def run(self,task,colors):
            self.setup.require_ready(task,site_test=self.setup.site_test)
            assert robot.snapshot().tcp_pose==pytest.approx(self.setup.home.as_list(),abs=.001)
            records.append(('run',task,colors))
    monkeypatch.setattr(closed_loop,'RealUR5',lambda *a,**k:robot)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',Camera)
    monkeypatch.setattr(closed_loop,'create_gripper',lambda *a:Gripper())
    monkeypatch.setattr(closed_loop,'Runtime',Runtime)
    monkeypatch.setattr('builtins.input',lambda *a:(_ for _ in ()).throw(AssertionError('Hidden terminal input')))
    assert closed_loop.main(['run','--task','sequence','--execute','--auto-start','--site-test',
        '--output',str(tmp_path/'sequence')])==0
    assert ('run','sequence',['yellow','green','blue']) in records
    assert records[0][0]=='field_test_started'
    assert records[0][1]['site_verified'] is False
    assert not any(record[0]=='camera_start' for record in records)  # Deferred until unfold.
    if start=='sequence-view':
        assert len(robot.records)==1 and robot.records[0].label=='return_folded_standby'
        assert robot.records[0].speed==.15 and robot.records[0].acceleration==.3
        assert next(i for i,r in enumerate(records) if r[0]=='startup_home_complete')<next(i for i,r in enumerate(records) if r[0]=='run')
    else:
        assert robot.records==[]


def test_missing_sequence_roi_still_blocks_before_hardware(tmp_path,monkeypatch):
    import json
    import pytest
    source=closed_loop.ROOT/'config/closed_loop_sequence.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    data['sequence_target_roi']=None
    for key in ('site_config','grasp_config','vision_config'):
        data[key]=str(source.parent/data[key])
    config=tmp_path/'config.json';config.write_text(json.dumps(data),encoding='utf-8')
    def forbidden(*a,**k):raise AssertionError('Hardware factory called')
    monkeypatch.setattr(closed_loop,'RealUR5',forbidden)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',forbidden)
    with pytest.raises(ValueError,match='sequence_target_roi'):
        closed_loop.main(['run','--task','sequence','--execute','--auto-start','--site-test','--config',str(config)])

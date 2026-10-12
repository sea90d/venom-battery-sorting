import math
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest
from battery_sorting.realsense import RealSenseCamera,RealSenseError
from battery_sorting.scene import SceneError
from tools.run_smooth_t0 import initial_red_estimate


def scene(x=0.,yaw=0.):
    result={c:SimpleNamespace(handle=None,center_base_m=(.1,.1,.05)) for c in ('red','yellow','green','blue')}
    result['red'].handle={'center_base_m':[x,0.,.05],'yaw_rad':math.radians(yaw)}
    return result


def test_other_colors_need_bodies_not_handles():
    xy,yaw=initial_red_estimate([scene(yaw=179),scene(yaw=1),scene(yaw=0)])
    assert xy==pytest.approx([0,0])
    assert min(yaw,math.pi-yaw)<1e-6


def test_observed_three_degree_jitter_allowed():
    _,yaw=initial_red_estimate([scene(yaw=v) for v in [176.4165,173.8739,179.7634]])
    assert math.degrees(yaw)==pytest.approx(176.4165)


@pytest.mark.parametrize('samples',[[scene(x=0),scene(x=.001),scene(x=.004)],[scene(yaw=0),scene(yaw=0),scene(yaw=6)]])
def test_outlier_is_not_silently_discarded(samples):
    with pytest.raises(SceneError,match='unstable'):initial_red_estimate(samples)


def test_missing_red_is_not_threshold_retry():
    found=scene();del found['red']
    with pytest.raises(ValueError,match='four batteries'):initial_red_estimate([found])


def test_missing_target_handle_rejects():
    found=scene();found['red'].handle=None
    with pytest.raises(SceneError,match='Red handle missing'):initial_red_estimate([found])


def test_target_occupancy_uses_floor_above_printed_red(monkeypatch):
    import tools.run_smooth_t0 as task
    observation=SimpleNamespace(
        colour=np.zeros((2,2,3),np.uint8),depth_m=np.ones((2,2),np.float32),
        base_T_camera=np.eye(4))
    setup=SimpleNamespace(
        vision=SimpleNamespace(colors=('red',)),
        raw={'paper_z_m':0.,'target_roi':[0,0,2,2],'max_height_mad_m':.005})
    camera=SimpleNamespace(normalize_color_pixels=object())
    seen={}
    def fake_detect(*args,**kwargs):
        seen.update(kwargs)
        return {}
    monkeypatch.setattr(task,'detect_batteries',fake_detect)
    assert task.detect_target_occupancy(observation,setup,camera)=={}
    assert seen['min_pixels']==task.TARGET_OCCUPANCY_MIN_PIXELS==500
    assert seen['roi']==[0,0,2,2]


@pytest.mark.parametrize('settles',[True,False])
def test_exposure_readback_settles_or_gives_values(monkeypatch,settles):
    monkeypatch.setattr('battery_sorting.realsense.time.sleep',lambda _:None)
    camera=RealSenseCamera(warmup_frames=1)
    rs=SimpleNamespace(option=SimpleNamespace(**{k:k for k in ['exposure','gain','white_balance','enable_auto_exposure','enable_auto_white_balance']}))
    camera._rs=rs;camera._profile=MagicMock();camera.capture=MagicMock()
    sensor=camera._profile.get_device.return_value.first_color_sensor.return_value
    sensor.get_option_range.return_value=SimpleNamespace(min=0,max=10000,step=1)
    readings=[]
    def read(option):
        readings.append(option)
        if option=='exposure':return 156 if settles and readings.count(option)>=3 else 80
        return {'gain':64,'white_balance':4600}[option]
    sensor.get_option.side_effect=read
    if settles:
        camera.configure_color({'exposure':156,'gain':64,'white_balance':4600})
        assert readings.count('exposure')==3
        camera.capture.assert_called_once()
    else:
        with pytest.raises(RealSenseError,match='requested=156, actual=80'):
            camera.configure_color({'exposure':156,'gain':64,'white_balance':4600})
        assert readings.count('exposure')==10
        camera.capture.assert_not_called()
    assert sum(call.args==('exposure',156) for call in sensor.set_option.call_args_list)==1


def test_main_depth_retry_is_bounded_and_unfolds_once_before_camera(tmp_path,monkeypatch):
    import tools.run_smooth_t0 as task
    from battery_sorting.robot import FakeRobot
    setup=task.Setup(task.ROOT/'config/closed_loop.json')
    robot=FakeRobot(setup.home)
    robot.active_tcp_offset=lambda:setup.raw['active_tcp_offset_m_rad']
    camera=MagicMock()
    gripper=MagicMock()
    gripper.config=setup.grasp.gripper
    gripper.last_grip_state=1
    gripper.has_object.return_value=False
    gripper.position.return_value=1000
    monkeypatch.setattr(task,'create_gripper',lambda _:gripper)
    observation=MagicMock()
    observation.metadata={'camera':{'frame_number':1}}
    calls=[]
    def reject(*a,**k):
        calls.append(1)
        raise SceneError('Low original color depth coverage: green')
    monkeypatch.setattr(task,'RealUR5',lambda _,**kwargs:robot)
    monkeypatch.setattr(task,'RealSenseCamera',lambda:camera)
    monkeypatch.setattr(task,'read_tcp_info',lambda *a,**k:{'tcp_base_m_rad':setup.home.as_list(),'active_tcp_offset_m_rad':setup.raw['active_tcp_offset_m_rad']})
    monkeypatch.setattr(task,'capture_observation',lambda *a,**k:observation)
    monkeypatch.setattr(task,'detect_batteries',reject)
    monkeypatch.setattr(task.sys,'argv',['run_smooth_t0','--execute','--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match=f'after {task.MAX_INITIAL_ATTEMPTS} attempts'):
        task.main()
    assert len(calls)==task.MAX_INITIAL_ATTEMPTS
    assert [r.label for r in robot.records]==['unfold_to_observation']
    assert robot.records[0].pose.as_list()[:3]==setup.raw['target_view_pose_m_rad'][:3]
    gripper.close.assert_not_called()


def test_prepared_start_rejects_tcp_mismatch_before_opening_gripper(tmp_path,monkeypatch):
    import tools.run_smooth_t0 as task
    from battery_sorting.robot import FakeRobot
    setup=task.Setup(task.ROOT/'config/closed_loop.json')
    robot=FakeRobot(setup.home)
    factory=MagicMock()
    monkeypatch.setattr(task,'RealUR5',lambda _,**kwargs:robot)
    monkeypatch.setattr(task,'RealSenseCamera',MagicMock())
    monkeypatch.setattr(task,'create_gripper',factory)
    monkeypatch.setattr(task,'read_tcp_info',lambda *a,**k:{'tcp_base_m_rad':setup.home.as_list(),'active_tcp_offset_m_rad':[0.]*6})
    monkeypatch.setattr(task.sys,'argv',['run_smooth_t0','--execute','--wait-for-start','--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='before preparation'):task.main()
    factory.assert_not_called()
    assert not robot.control_connected
    assert robot.records==[]

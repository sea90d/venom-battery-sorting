import math
from dataclasses import replace
import json
import numpy as np
import pytest
from battery_sorting.closed_loop import main,Setup,resolve_task_colors
from battery_sorting.table_view import require_stationary_board
from .closed_loop_fakes import environment,ROOT


@pytest.mark.parametrize('task,colors',[('basic',['red']),('sequence',['yellow','green','blue'])])
def test_actual_images_through_complete_runtime(tmp_path,task,colors):
    rt,world,sleeps=environment(tmp_path,task,sideways=True)
    initial={c:item['xy'].copy() for c,item in world.items()}
    timing=rt.run(task,colors)
    assert rt.events[-1]['event']=='complete'
    assert rt.robot.snapshot().tcp_pose==pytest.approx(rt.setup.home.as_list())
    assert rt.gripper.held is None
    assert rt.gripper.actions.count('close')==len(colors)
    assert timing['task_run_time_s']>0
    assert [item['color'] for item in timing['battery_run_times']]==colors
    summary=json.loads((rt.output/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed'] is True
    assert summary['timing']==timing
    cycle=next(i for i,e in enumerate(rt.events) if e['event']=='cycle_started')
    first_capture=next(i for i,e in enumerate(rt.events) if e['event']=='capture')
    assert cycle<first_capture
    pick_views=[r for r in rt.robot.records if r.label.startswith('pick_view:')]
    assert pick_views
    for record in pick_views:
        assert record.pose.as_list()[3:]==pytest.approx(rt.setup.pick_view.as_list()[3:])
    from battery_sorting.competition_start import require_competition_start
    require_competition_start(rt.setup.home)
    assert rt.events[-2]['event']=='home_arrived'
    assert sum(s>=3 for s in sleeps)==len(colors)
    for i,c in enumerate(colors):
        target=rt.targets['T0' if task=='basic' else f'P{i+1}']
        assert world[c]['xy']==pytest.approx(target['center'],abs=.002)
        pick=[r for r in rt.robot.records if r.label.startswith(c+':pick:')]
        assert pick[-1].pose.z==.069207
        assert pick[-1].pose.as_list()[:2]==pytest.approx(initial[c],abs=.002)
        assert pick[-1].speed==rt.setup.grasp.contact_speed_mps
    poses=[rt.setup.home]+[r.pose for r in rt.robot.records]
    for a,b in zip(poses,poses[1:]):
        assert abs(a.z-b.z)<=.03+1e-8
        if math.dist((a.x,a.y),(b.x,b.y))>1e-8:
            assert a.z==b.z==rt.setup.site.safe_high_z_m


@pytest.mark.parametrize('fault',['close_comm','drop_lift','drop_transfer','release','board','offset','stale','shift'])
def test_faults_stop_without_unverified_recovery(tmp_path,monkeypatch,fault):
    rt,world,_=environment(tmp_path,sideways=True)
    def failure(*a,**k): raise RuntimeError('synthetic communication failure')
    if fault=='close_comm': monkeypatch.setattr(rt.gripper,'close',failure)
    if fault in ('drop_lift','drop_transfer'):
        original=rt.gripper.has_object
        def feedback():
            label=rt.robot.records[-1].label if rt.robot.records else ''
            token=':lift:' if fault=='drop_lift' else ':place_high:'
            return False if token in label else original()
        monkeypatch.setattr(rt.gripper,'has_object',feedback)
    if fault=='release': rt.gripper.release_shift=.025
    if fault=='board':
        original=rt.gripper.open
        def release():
            held=rt.gripper.held
            result=original()
            if held: rt.camera.board_shift=.015
            return result
        monkeypatch.setattr(rt.gripper,'open',release)
    if fault=='offset': monkeypatch.setattr(rt.robot,'active_tcp_offset',lambda:[.02,0,0,0,0,0])
    if fault in ('stale','shift'):
        original=rt.scene
        def scene(label,roi=None,**kwargs):
            obs,found=original(label,roi,**kwargs)
            if label=='grasp_hover':
                if fault=='stale': obs=replace(obs,captured_monotonic=obs.captured_monotonic-1000)
                else: found['green']=replace(found['green'],center_base_m=(9,9,.05))
            return obs,found
        monkeypatch.setattr(rt,'scene',scene)
    with pytest.raises((RuntimeError,ValueError)):
        rt.run('basic',['green'])
    assert rt.events[-1]['event']=='failed'
    assert rt.robot.stop_count==1
    assert not any(e['event']=='complete' for e in rt.events)
    if fault in ('close_comm','drop_lift','drop_transfer'):
        assert rt.gripper.actions.count('open')==1
        assert not any(':retreat:' in r.label for r in rt.robot.records)


def test_missing_data_stops_before_hardware_factories(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('Hardware factory called')
    monkeypatch.setattr('battery_sorting.closed_loop.RealUR5',forbidden)
    monkeypatch.setattr('battery_sorting.closed_loop.RealSenseCamera',forbidden)
    assert main(['readiness'])==0
    with pytest.raises(ValueError,match='Missing site data'):
        main(['run','--execute'])
    with pytest.raises(ValueError,match='geometry_and_gripper_verified_by_task.sequence'):
        main(['run','--task','sequence','--execute'])
    assert Setup(ROOT/'config/closed_loop.json').grasp.final_grasp_z_m==.069207


def test_sequence_rejects_downward_waiting_pose_before_any_motion(tmp_path):
    rt,_,_=environment(tmp_path,'sequence')
    with pytest.raises(ValueError,match='long axis is not parallel'):
        rt.run('sequence',['yellow','green','blue'])
    assert rt.robot.records==[]
    assert rt.gripper.actions==[]


def test_noncentral_handle_is_not_guessed(tmp_path):
    rt,_,_=environment(tmp_path)
    rt.setup.raw['grasp_center_offset_xy_m']=[.01,0]
    with pytest.raises(ValueError,match='undirected'):
        rt.setup.require_ready()


def test_already_placed_battery_is_rechecked_after_next_release(tmp_path,monkeypatch):
    rt,world,_=environment(tmp_path,'sequence',sideways=True)
    original=rt.gripper.open
    def release():
        held=rt.gripper.held
        result=original()
        if held=='red': world['blue']['xy']+=np.array([.025,0])
        return result
    monkeypatch.setattr(rt.gripper,'open',release)
    with pytest.raises(ValueError,match='within target'):
        rt.run('sequence',['blue','red','yellow'])
    assert [e['color'] for e in rt.events if e['event']=='battery_complete']==['blue']


def test_tilted_target_view_uses_metric_rectification(tmp_path):
    from battery_sorting.config import Pose
    from battery_sorting.placement_vision import detect_t0
    rt,_,_=environment(tmp_path)
    rt.robot._pose=Pose(0,0,.447856,2.95,.05,.02)
    view=rt.table(rt.frame('tilted'))
    target=detect_t0(view.image)
    assert target.valid,target.reason
    assert view.to_base([target.center_px])[0]==pytest.approx([-.23,.02],abs=.002)


def test_invalid_full_configuration_fails_offline(tmp_path):
    rt,_,_=environment(tmp_path)
    rt.setup.raw['grip_position_range']=[-1,1001]
    with pytest.raises(ValueError,match='grip_position_range'):
        rt.setup.require_ready()
    assert rt.robot.records==[]


def test_legacy_execute_rejected_before_opening_devices():
    import main as legacy
    from battery_sorting.config import ConfigError
    with pytest.raises(ConfigError,match='末端 D455'):
        legacy.run(['--execute','--task','basic','--colors','red'])


def test_official_notice_defaults_and_explicit_bench_override():
    assert resolve_task_colors('basic')==['red']
    assert resolve_task_colors('sequence')==['yellow','green','blue']
    assert resolve_task_colors('basic',['green'])==['green']
    with pytest.raises(ValueError):
        resolve_task_colors('sequence',['yellow','yellow','blue'])


def test_sequence_requires_its_own_target_view_and_roi():
    setup=Setup(ROOT/'config/closed_loop.json')
    setup.raw['sequence_target_view_pose_m_rad']=None
    setup.raw['sequence_target_roi']=None
    assert 'closed_loop.sequence_target_view_pose_m_rad' in setup.blockers('sequence')
    assert 'closed_loop.sequence_target_roi' in setup.blockers('sequence')
    assert 'closed_loop.grasp_center_offset_xy_m' not in setup.blockers('sequence')
    assert 'closed_loop.grasp_center_offset_xy_m' in setup.blockers('basic')
    assert not any('sequence_target' in item for item in setup.blockers('basic'))


def test_grasp_orientation_uses_handle_axis_not_ambiguous_body_axis(tmp_path):
    rt,_,_=environment(tmp_path)
    _,scene=rt.scene('handle_axis',rt.setup.raw['pick_roi'])
    item=scene['yellow']
    assert item.handle is not None
    swapped=replace(item,yaw_rad=(item.yaw_rad+math.pi/2)%math.pi)
    assert rt.handle_yaw(swapped)==pytest.approx(rt.handle_yaw(item))
    assert rt.orientation(rt.handle_yaw(swapped))==pytest.approx(rt.orientation(rt.handle_yaw(item)))
    missing=replace(item,handle=None)
    with pytest.raises(ValueError,match='handle missing'):
        rt.handle_yaw(missing)


def test_sequence_uses_observed_handle_center_and_preserves_body_center(tmp_path):
    rt,_,_=environment(tmp_path)
    _,scene=rt.scene('offset_handle',rt.setup.raw['pick_roi'])
    item=scene['yellow']
    shifted=dict(item.handle)
    shifted['center_base_m']=[item.center_base_m[0]+.008,
                              item.center_base_m[1]-.003,
                              item.handle['center_base_m'][2]]
    item=replace(item,handle=shifted)
    assert rt.grip_xy(item)==pytest.approx(shifted['center_base_m'][:2])
    target={'center':[-.2,.1]}
    tcp=rt.placement_xy(item,target,'sequence',rt.handle_yaw(item))
    assert tcp==pytest.approx([-.192,.097])


def test_analyse_uses_same_pick_roi_as_runtime(tmp_path,monkeypatch):
    rt,_,_=environment(tmp_path)
    obs=rt.frame('offline_roi_test')
    monkeypatch.setattr('battery_sorting.closed_loop.Setup',lambda _:rt.setup)
    monkeypatch.setattr('battery_sorting.closed_loop.Observation.load',lambda _:obs)
    def detector(*args,**kwargs):
        assert kwargs['roi']==rt.setup.raw['pick_roi']
        raise RuntimeError('ROI passed correctly')
    monkeypatch.setattr('battery_sorting.closed_loop.detect_batteries',detector)
    with pytest.raises(RuntimeError,match='ROI passed correctly'):
        main(['analyse','--bundle','unused-local-bundle'])

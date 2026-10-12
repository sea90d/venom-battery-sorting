from dataclasses import replace
import numpy as np
import pytest
from battery_sorting.observation import Observation,capture_observation,recorded_camera_model
from .closed_loop_fakes import environment


def capture(rt):
    return capture_observation(rt.camera,rt.robot,np.eye(4),rt.setup.calibration_sha256,rt.session_id,[0.]*6,'234222302865')


def test_saved_bundle_recomputes_same_transform_but_never_live(tmp_path):
    rt,_,_=environment(tmp_path)
    obs=capture(rt)
    obs.require_live(rt.session_id,rt.setup.calibration_sha256,5)
    obs.save(tmp_path/'bundle')
    restored=Observation.load(tmp_path/'bundle')
    assert np.allclose(restored.base_T_camera,obs.base_T_camera)
    with pytest.raises(ValueError,match='replayed'):
        restored.require_live(rt.session_id,rt.setup.calibration_sha256,5)
    path=tmp_path/'bundle'/'colour.png'
    path.write_bytes(path.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='integrity'):
        Observation.load(tmp_path/'bundle')


@pytest.mark.parametrize('mutation',['tcp','rgb','depth','expired','session','calibration'])
def test_live_authority_rejects_invalid_data(tmp_path,mutation):
    rt,_,_=environment(tmp_path)
    obs=capture(rt)
    if mutation=='tcp': obs.metadata['tcp_before'][0]+=9
    if mutation=='rgb': obs.colour[0,0]=0
    if mutation=='depth': obs.depth_m[0,0]=1
    if mutation=='expired': obs=replace(obs,captured_monotonic=obs.captured_monotonic-100)
    session=rt.session_id if mutation!='session' else 'wrong'
    calibration=rt.setup.calibration_sha256 if mutation!='calibration' else 'wrong'
    with pytest.raises(ValueError): obs.require_live(session,calibration,5)


def test_moving_tcp_and_repeated_frame_rejected(tmp_path,monkeypatch):
    rt,_,_=environment(tmp_path)
    with pytest.raises(ValueError,match='out-of-order'):
        capture_observation(rt.camera,rt.robot,np.eye(4),'hash','run',[0.]*6,'234222302865',previous_frame_number=999)
    original=rt.robot.snapshot
    count=0
    def drifting():
        nonlocal count
        count+=1
        result=original()
        return replace(result,tcp_pose=(result.tcp_pose[0]+count*.001,*result.tcp_pose[1:]))
    monkeypatch.setattr(rt.robot,'snapshot',drifting)
    with pytest.raises(ValueError,match='moved'):
        capture(rt)


def test_real_d455_distortion_math_replays_without_hardware(tmp_path,monkeypatch):
    import json
    import pyrealsense2 as rs
    from .closed_loop_fakes import ROOT
    def forbidden(*a,**k): raise AssertionError('Hardware was opened')
    monkeypatch.setattr(rs,'pipeline',forbidden)
    monkeypatch.setattr(rs,'context',forbidden)
    rt,_,_=environment(tmp_path)
    obs=capture(rt)
    saved=json.loads((ROOT/'output/handeye/session_20260910/session.json').read_text())['camera']
    intr=obs.metadata['camera']['intrinsics']
    intr.update(width=saved['width'],height=saved['height'],fx=saved['intrinsics']['fx'],fy=saved['intrinsics']['fy'],
                ppx=saved['intrinsics']['cx'],ppy=saved['intrinsics']['cy'],model=saved['distortion_model'],coeffs=saved['distortion_coefficients'])
    camera=recorded_camera_model(obs.metadata)
    pixels=np.array([[100.,100.],[645.,360.],[1100.,600.]])
    normalized=camera.normalize_color_pixels(pixels)
    projected=camera.project_color_points(np.column_stack((normalized,np.ones(3))))
    assert projected==pytest.approx(pixels,abs=.03)

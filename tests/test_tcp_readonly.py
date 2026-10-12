import struct

import pytest

from battery_sorting.tcp_readonly import parse_cartesian_info, read_tcp_info


POSE=(.12,-.25,.069207,-2.24,-2.18,0.)
OFFSET=(.001,.002,.15,0.,0.,0.)


def frame(values=POSE+OFFSET):
    sub=struct.pack('!IB12d',101,4,*values)
    return struct.pack('!IB',5+len(sub),16)+sub


def test_cartesian_info_distinguishes_base_pose_and_tool_offset():
    result=parse_cartesian_info(frame())
    assert result['tcp_base_m_rad']==list(POSE)
    assert result['active_tcp_offset_m_rad']==list(OFFSET)


@pytest.mark.parametrize('packet',[
    b'',frame()[:-1],struct.pack('!IBIB',10,16,100,4),
    frame(POSE+OFFSET[:-1]+(float('nan'),)),
])
def test_corrupt_frames_are_rejected(packet):
    with pytest.raises(ValueError):
        parse_cartesian_info(packet)


def test_fragmented_stream_skips_messages_without_sending(monkeypatch):
    data=bytearray(struct.pack('!IB',5,20)+frame())
    class Stream:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def settimeout(self,value): assert 0<value<=5
        def recv(self,count):
            chunk=bytes(data[:min(count,3)])
            del data[:len(chunk)]
            return chunk
        def send(self,*args): raise AssertionError('No commands allowed')
        def sendall(self,*args): raise AssertionError('No commands allowed')
    def connect(address,timeout):
        assert address==('192.168.1.10',30011)
        return Stream()
    monkeypatch.setattr('battery_sorting.tcp_readonly.socket.create_connection',connect)
    result=read_tcp_info('192.168.1.10')
    assert result['active_tcp_offset_m_rad']==list(OFFSET)
    assert result['matches_historical_calibration'] is None


def test_tcp_read_cli_has_no_robot_camera_or_gripper_factories(monkeypatch,tmp_path):
    from battery_sorting import closed_loop
    def forbidden(*args,**kwargs): raise AssertionError('Hardware factory called')
    monkeypatch.setattr(closed_loop,'RealUR5',forbidden)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',forbidden)
    monkeypatch.setattr(closed_loop,'create_gripper',forbidden)
    monkeypatch.setattr(closed_loop,'read_tcp_info',lambda host:parse_cartesian_info(frame()))
    assert closed_loop.main(['tcp-read','--output',str(tmp_path/'tcp.json')])==0


@pytest.mark.parametrize('changed',[False,True])
def test_collect_reads_offset_with_null_config_and_rejects_changes(monkeypatch,tmp_path,changed):
    from dataclasses import dataclass
    from battery_sorting import closed_loop
    original_setup=closed_loop.Setup
    def unconfigured_setup(path):
        setup=original_setup(path)
        setup.raw['active_tcp_offset_m_rad']=None
        return setup
    monkeypatch.setattr(closed_loop,'Setup',unconfigured_setup)
    saved=[]
    class Robot:
        def __init__(self,*args): pass
        def connect_readonly(self): pass
        def disconnect(self): pass
        def connect_control(self): raise AssertionError('Control connection forbidden')
    class Camera:
        def start(self,**kwargs): pass
        def stop(self): pass
    @dataclass
    class Captured:
        metadata: dict
        live: bool = True
        def save(self,path): saved.append(self)
    def capture(*args):
        assert args[5]==list(OFFSET)
        return Captured({'tcp_before':list(POSE),'active_tcp_offset':args[5],'camera':{}})
    offsets=iter([list(OFFSET),[.02,*OFFSET[1:]] if changed else list(OFFSET)])
    monkeypatch.setattr(closed_loop,'RealUR5',Robot)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',Camera)
    monkeypatch.setattr(closed_loop,'capture_observation',capture)
    monkeypatch.setattr(closed_loop,'read_tcp_info',lambda host:{'active_tcp_offset_m_rad':next(offsets)})
    args=['collect','--output',str(tmp_path/'capture')]
    if changed:
        with pytest.raises(ValueError,match='offset changed'):
            closed_loop.main(args)
        assert not saved
    else:
        assert closed_loop.main(args)==0
        assert not saved[0].live
        assert saved[0].metadata['active_tcp_offset_source']=='primary_readonly_cartesian_info'


def test_collect_declared_offset_mismatch_stops_before_capture(monkeypatch,tmp_path):
    from battery_sorting import closed_loop
    original_setup=closed_loop.Setup
    def configured_setup(path):
        setup=original_setup(path)
        setup.raw['active_tcp_offset_m_rad']=[0,0,.246,0,0,0]
        return setup
    class Device:
        def __init__(self,*args): pass
        def connect_readonly(self): pass
        def disconnect(self): pass
        def start(self,**kwargs): pass
        def stop(self): pass
    def forbidden(*args,**kwargs):
        raise AssertionError('Capture or gripper must not be reached')
    monkeypatch.setattr(closed_loop,'Setup',configured_setup)
    monkeypatch.setattr(closed_loop,'RealUR5',Device)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',Device)
    monkeypatch.setattr(closed_loop,'capture_observation',forbidden)
    monkeypatch.setattr(closed_loop,'create_gripper',forbidden)
    monkeypatch.setattr(closed_loop,'read_tcp_info',lambda _:dict(active_tcp_offset_m_rad=list(OFFSET)))
    with pytest.raises(ValueError,match='differs from declared'):
        closed_loop.main(['collect','--output',str(tmp_path/'never_saved')])
    assert not (tmp_path/'never_saved').exists()


@pytest.mark.parametrize('task',['basic','sequence'])
def test_hardware_preflight_uses_broadcast_offset_without_control(monkeypatch,capsys,task):
    import json
    from battery_sorting import closed_loop
    from battery_sorting.robot import RobotSnapshot
    calls=[]
    class Robot:
        def __init__(self,*args): pass
        def connect_readonly(self): calls.append('receive')
        def disconnect(self): calls.append('disconnect')
        def snapshot(self): return RobotSnapshot(tuple(POSE),7,1,1,False,False)
        def active_tcp_offset(self): raise AssertionError('RTDE Control TCP query is forbidden')
        def pose_within_safety_limits(self,*args): raise AssertionError('RTDE Control safety query is forbidden')
        def connect_control(self): raise AssertionError('Control connection is forbidden')
    class Camera:
        def start(self,*args,**kwargs): raise AssertionError('No camera stream during this hardware probe')
    class Gripper:
        def connect(self): calls.append('gripper read')
        def disconnect(self): pass
        def position(self): return 1000
        def has_object(self): return False
    monkeypatch.setattr(closed_loop,'RealUR5',Robot)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',Camera)
    monkeypatch.setattr(closed_loop,'create_gripper',lambda *args:Gripper())
    monkeypatch.setattr(closed_loop,'read_tcp_info',lambda host:{'active_tcp_offset_m_rad':[0,0,.246,0,0,0]})
    args=['preflight','--task',task]
    if task=='sequence': args+=['--site-test']
    assert closed_loop.main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report['tcp_offset_source']=='primary_readonly_cartesian_info'
    assert report['controller_path_check'].startswith('deferred')
    assert calls==['receive','disconnect','gripper read']
    if task=='basic':
        assert 'geometry_and_gripper_verified_by_task.basic=false' in report['mission_blockers']


def test_preflight_wrong_tcp_never_reaches_gripper(monkeypatch):
    from battery_sorting import closed_loop
    from battery_sorting.robot import RobotSnapshot
    class Robot:
        def __init__(self,*args): pass
        def connect_readonly(self): pass
        def disconnect(self): pass
        def snapshot(self): return RobotSnapshot(tuple(POSE),7,1,1,False,False)
    class Camera: pass
    def forbidden(*args): raise AssertionError('Gripper must not be reached with incorrect TCP')
    monkeypatch.setattr(closed_loop,'RealUR5',Robot)
    monkeypatch.setattr(closed_loop,'RealSenseCamera',Camera)
    monkeypatch.setattr(closed_loop,'create_gripper',forbidden)
    monkeypatch.setattr(closed_loop,'read_tcp_info',lambda host:{'active_tcp_offset_m_rad':list(OFFSET)})
    with pytest.raises(ValueError,match='differs from hand-eye'):
        closed_loop.main(['preflight','--task','sequence','--site-test'])

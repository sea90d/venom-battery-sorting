from dataclasses import replace
import pytest
from battery_sorting.config import GripperConfig
from battery_sorting.gripper import AG105145Gripper,GripperError
from tools.run_smooth_t0 import prepare_startup_gripper
from tools.reset_empty_gripper import main


class Client:
    connected=True
    def __init__(self,states,positions,initialized=1):
        self.states=iter(states);self.positions=iter(positions)
        self.initialized=initialized;self.writes=[]
    def read_register(self,register):
        if register==0x0200:return self.initialized
        return next(self.states if register==0x0201 else self.positions)
    def write_register(self,register,value):
        self.writes.append((register,value))
        if register==0x0100:self.initialized=1


def gripper(states,positions,initialized=1):
    g=AG105145Gripper(GripperConfig(driver='ag105145_modbus',open_position=400,poll_interval=.001,command_timeout=.1))
    g.client=Client(states,positions,initialized)
    return g


@pytest.mark.parametrize('initialized',[0,1])
def test_reset_latched_drop_requires_three_open_position_samples(initialized):
    g=gripper([3,3,0,1,1,1],[0,200,400,400,400],initialized)
    report=g.reset_empty_open(empty_and_clear_confirmed=True)
    assert report['final_state']==1 and report['position']==400
    assert report['stable_samples']==3
    writes=g.client.writes
    assert (0x0100,0xA5) not in writes
    assert writes.count((0x0103,400))==1
    assert ((0x0100,1) in writes)==(not initialized)


@pytest.mark.parametrize('state',[0,2,9])
def test_moving_held_unknown_rejected_before_writes(state):
    g=gripper([state],[])
    with pytest.raises(GripperError):g.reset_empty_open(empty_and_clear_confirmed=True)
    assert g.client.writes==[]


def test_explicit_empty_clear_confirmation_required():
    g=gripper([],[])
    with pytest.raises(GripperError,match='Confirm'):g.reset_empty_open()
    assert g.client.writes==[]


@pytest.mark.parametrize('state,position',[(3,400),(1,0),(0,400)])
def test_neither_drop_nor_wrong_opening_is_success(monkeypatch,state,position):
    import itertools
    g=gripper(itertools.chain([3],itertools.repeat(state)),itertools.repeat(position))
    clock=iter([0,.01,.02,.03,.11])
    monkeypatch.setattr('battery_sorting.gripper.time.monotonic',lambda:next(clock))
    with pytest.raises(GripperError,match='timed out'):
        g.reset_empty_open(empty_and_clear_confirmed=True)


def test_held_state_during_reset_rejected():
    g=gripper([3,2],[239])
    with pytest.raises(GripperError,match='Unexpected object'):
        g.reset_empty_open(empty_and_clear_confirmed=True)


def test_communication_failure_after_open_does_not_retry_actuation():
    g=gripper([3],[])
    original=g.client.read_register
    def read(register):
        if g.client.writes and register==0x0201:
            raise GripperError('communication lost')
        return original(register)
    g.client.read_register=read
    with pytest.raises(GripperError,match='communication lost'):
        g.reset_empty_open(empty_and_clear_confirmed=True)
    assert [p for p in g.client.writes if p[0]==0x0103]==[(0x0103,400)]


def test_normal_open_still_rejects_drop_state():
    g=gripper([3],[400]);g._initialized=True
    with pytest.raises(GripperError,match='掉落'):g.open()


@pytest.mark.parametrize('reset,fail',[(False,False),(True,True),(True,False)])
def test_startup_failure_blocks_caller_and_always_disconnects(reset,fail):
    class Probe:
        last_grip_state=3
        disconnected=False
        def connect(self):pass
        def disconnect(self):self.disconnected=True
        def reset_empty_open(self,**kw):
            if fail:raise GripperError('reset failed')
            self.last_grip_state=1
            return dict(final_state=1,position=400)
    probe=Probe();events=[]
    call=lambda:prepare_startup_gripper(GripperConfig(),reset,lambda e,**d:events.append(e),lambda c:probe)
    if not reset or fail:
        with pytest.raises((RuntimeError,GripperError)):call()
        assert 'startup_empty_reset_complete' not in events
    else:call();assert 'startup_empty_reset_complete' in events
    assert probe.disconnected


def test_reset_plan_never_creates_gripper(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Hardware factory')
    monkeypatch.setattr('tools.reset_empty_gripper.create_gripper',forbidden)
    assert main([])==0

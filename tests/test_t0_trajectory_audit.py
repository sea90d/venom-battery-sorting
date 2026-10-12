import cv2
import numpy as np
import pytest

from tools.audit_t0_trajectory import fk, ik, interpolate, segment_distance, audit_segment


@pytest.mark.parametrize('other_start,other_end,expected', [
    ([.5, -1, 0], [.5, 1, 0], 0),
    ([0, 1, 0], [1, 1, 0], 1),
    ([2, 0, 0], [3, 0, 0], 1),
])
def test_segment_distance_handles_intersections_parallel_and_endpoints(other_start, other_end, expected):
    assert segment_distance([0, 0, 0], [1, 0, 0], other_start, other_end) == pytest.approx(expected)


def test_inverse_kinematics_checks_orientation_and_tcp_offset():
    q = np.array([.67, -1.97, 1.17, -.8, -1.81, .8])
    flange, _ = fk(q)
    pose = np.r_[flange[:3, 3]+flange[:3, 2]*.246, cv2.Rodrigues(flange[:3, :3])[0].ravel()]
    result, error = ik(pose, q+.02)
    assert np.linalg.norm(error[:3]) < .0001
    assert np.linalg.norm(error[3:])/.25 < .0002
    result_flange, _ = fk(result)
    np.testing.assert_allclose(result_flange[:3, :3], flange[:3, :3], atol=.0002)


def test_interpolation_keeps_tcp_fixed_during_a_turn():
    start = [0, 0, .45, 0, 0, 0]
    end = [0, 0, .45, 0, 0, np.pi/2]
    middle = interpolate(start, end, .5)
    np.testing.assert_allclose(middle, [0, 0, .45, 0, 0, np.pi/4], atol=1e-10)


def test_sideways_home_flags_forearm_proximity_on_incident_seed_branch():
    home = [-.06, -.15, .447856, -1.209006647505285, -1.20950001934725, -1.20950001934725]
    q = np.radians([69.8, -79.3, 74.1, 5.2, -110.2, 90])
    segment, _ = audit_segment('sideways_home', home, home, q)
    assert segment['max_ik_position_error_m'] < .0001
    assert segment['nearest_link_at_minimum'] == 'forearm'
    # A centerline separation, not exterior clearance or collision certification.
    assert segment['min_tool_link_centerline_distance_m'] < .070


@pytest.mark.parametrize('start,seed', [
    ([-.4700090206,-.1918393094,.2346585546,-1.6986195947,-1.6333735418,.9645670393],
     np.radians([15.5,-116.1,135.1,-164.2,-98.7,13.2])),
    ([-.1433695183,-.1588968124,.4467879703,-2.1720079213,-1.9277964964,.0285059695],
     [.6705541015,-1.9717505614,1.1702508926,-.7977140586,-1.8156479041,.7984209061]),
])
def test_revised_preparation_avoids_old_forearm_proximity_in_nominal_model(start,seed):
    from tools.run_smooth_t0 import startup_home_targets,startup_move_label
    from tools.revise_t0_start_pose import HOME,OBSERVATION
    from battery_sorting.config import Pose
    home,observation=Pose(*HOME),Pose(*OBSERVATION)
    targets=startup_home_targets(Pose(*start),home,home.z,observation)
    q=np.array(seed)
    previous=start
    labels=[]
    for target in targets:
        labels.append(startup_move_label(Pose(*previous),target,home))
        segment,q=audit_segment(labels[-1],previous,target.as_list(),q)
        assert segment['min_tool_link_centerline_distance_m']>.150
        assert segment['max_ik_position_error_m']<.001
        assert segment['max_ik_rotation_error_rad']<.001
        assert segment['max_joint_step_deg']<10
        previous=target.as_list()
    assert 'startup_turn_down' in labels
    assert labels[-1]=='startup_sideways_home'

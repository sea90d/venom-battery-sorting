from __future__ import annotations

import numpy as np

from battery_sorting.handeye import motion_diversity, pose_vector_to_transform
from battery_sorting.handeye_motion import OFFSETS, generate_targets
from battery_sorting.handeye import HandEyeSample


def test_calibration_motion_plan_is_bounded_and_diverse() -> None:
    start = (-0.046, -0.220, 0.448, -0.041, -3.134, -0.006)
    targets = generate_targets(start)
    assert len(targets) == len(OFFSETS) == 26
    transforms = [pose_vector_to_transform(start)] + [
        pose_vector_to_transform(target.as_list()) for target in targets
    ]
    samples = [
        HandEyeSample(transform, np.eye(4), 0.1) for transform in transforms
    ]
    translation, rotation, second = motion_diversity(samples)
    assert translation >= 0.050
    assert rotation >= 30.0 - 1e-6
    assert second >= 5.0

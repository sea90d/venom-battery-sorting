from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from battery_sorting.placement_cli import run
from tests.test_placement_vision import rounded_rectangle


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_cli_writes_valid_t0_diagnostics(tmp_path: Path) -> None:
    image = np.full((360, 640, 3), 255, np.uint8)
    rounded_rectangle(image, (320, 180), (300, 150))
    source = tmp_path / "scene.png"
    write_image(source, image)
    output = tmp_path / "out"
    assert run(["--image", str(source), "--mode", "t0", "--out", str(output)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["safety"]["robot_connected"] is False
    assert (output / "annotated.png").exists()


def test_cli_saves_fail_closed_report_for_blank_image(tmp_path: Path) -> None:
    source = tmp_path / "blank.png"
    write_image(source, np.full((200, 300, 3), 255, np.uint8))
    output = tmp_path / "out"
    assert run(["--image", str(source), "--mode", "t0", "--out", str(output)]) == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["results"][0]["center_px"] is None

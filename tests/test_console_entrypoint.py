"""Exercise the real launch commands without starting a robot task."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", [
    ["-m", "battery_sorting.competition_console"],
    ["tools/competition_console.py"],
])
def test_console_launch_command_has_cli(entry):
    result = subprocess.run(
        [sys.executable, *entry, "--help"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", timeout=10,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--task {basic,sequence}" in result.stdout
    assert "--no-browser" in result.stdout


@pytest.mark.parametrize("task", ["basic", "sequence"])
def test_module_starts_http_console_and_reuses_it(task):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "battery_sorting.competition_console",
               "--no-browser", "--port", str(port), "--task", task]
    environment = {**os.environ, "PYTHONUTF8": "1"}
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment)
    url = f"http://127.0.0.1:{port}/"
    try:
        deadline = time.monotonic() + 10
        while True:
            assert process.poll() is None, process.communicate()
            try:
                with urlopen(url + "api/status", timeout=.5) as response:
                    status = json.load(response)
                break
            except (URLError, OSError):
                assert time.monotonic() < deadline, "Console did not start"
                time.sleep(.05)
        assert status["state"] == "idle"
        assert status["task"] == task
        assert set(status["available_tasks"]) == {"basic", "sequence"}
        with urlopen(url, timeout=1) as response:
            assert response.status == 200
            assert "第二题" in response.read().decode("utf-8")
        # Launching again must return instead of failing to bind the same port.
        reused = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", timeout=5,
                                env=environment)
        assert reused.returncode == 0, reused.stderr
        assert url in reused.stdout
        assert process.poll() is None
        with urlopen(url + "api/status", timeout=1) as response:
            assert json.load(response)["state"] == "idle"
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)

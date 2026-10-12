"""比赛任务模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskError(ValueError):
    """任务口令不合法。"""


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass(frozen=True)
class BasicTask:
    target_color: Color

    def __post_init__(self) -> None:
        if not isinstance(self.target_color, Color):
            raise TaskError("基础任务颜色必须是 Color")

    @property
    def description(self) -> str:
        return f"basic:{self.target_color.value}->T0"


@dataclass(frozen=True)
class SequenceTask:
    color_order: tuple[Color, Color, Color]

    def __post_init__(self) -> None:
        if len(self.color_order) != 3 or any(not isinstance(color, Color) for color in self.color_order):
            raise TaskError("顺序任务必须恰好给出 3 个已知颜色")
        if len(set(self.color_order)) != 3:
            raise TaskError("顺序任务的 3 个颜色必须互不相同")

    @property
    def description(self) -> str:
        return "sequence:" + "-".join(color.value for color in self.color_order)


Task = BasicTask | SequenceTask

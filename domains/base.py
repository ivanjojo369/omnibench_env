# domains/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from omnibench_env.models import ToolSpec


@dataclass(frozen=True)
class TaskSpec:
    """
    A single evaluable task inside one domain.
    - id: stable identifier for the task
    - instruction: what the agent sees (string)
    - gold: domain-defined target used by the checker
    - meta: optional extra info (difficulty, seed, etc.)
    """
    id: str
    instruction: str
    gold: Any
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckResult:
    """
    Standard checker output for mode="respond".
    reward: numeric reward (0/1 for MVP; can be shaped later)
    done: whether the episode should terminate (usually True on respond)
    info: extra debug payload to return via observation.metadata/last_tool_result
    """
    reward: float
    done: bool
    info: dict[str, Any]


class Domain(Protocol):
    """
    Domain interface for the router.
    Each domain owns:
    - task sampling
    - tool list generation
    - tool execution (pure/offline for MVP)
    - final answer checking
    """
    name: str  # e.g., "web", "finance"

    def sample_task(self, seed: int) -> TaskSpec: ...
    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]: ...
    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]: ...
    def check_final(self, message: str, task: TaskSpec) -> CheckResult: ...

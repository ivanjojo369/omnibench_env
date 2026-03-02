# domains/healthcare.py
from __future__ import annotations

from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


class HealthcareDomain:
    """
    Offline Healthcare domain (synthetic triage rules; NOT medical advice).
    Output labels: LOW / MED / HIGH
    """

    name = "healthcare"

    def _task_triage_1(self) -> TaskSpec:
        return TaskSpec(
            id="health_triage_v1",
            instruction=(
                "Regla sintética:\n"
                "- Si temp>=39 o sat<=92 => HIGH\n"
                "- Si temp>=38 o sat<=94 => MED\n"
                "- Si no => LOW\n\n"
                "Paciente: temp=38.5, sat=95\n"
                "Responde SOLO: LOW/MED/HIGH."
            ),
            gold="MED",
            meta={"temp": 38.5, "sat": 95},
        )

    def _task_triage_2(self) -> TaskSpec:
        return TaskSpec(
            id="health_triage_v2",
            instruction=(
                "Aplica la misma regla sintética.\n"
                "Paciente: temp=39.2, sat=96\n"
                "Responde SOLO: LOW/MED/HIGH."
            ),
            gold="HIGH",
            meta={"temp": 39.2, "sat": 96},
        )

    def _task_triage_3(self) -> TaskSpec:
        return TaskSpec(
            id="health_triage_v3",
            instruction=(
                "Aplica la misma regla sintética.\n"
                "Paciente: temp=37.2, sat=97\n"
                "Responde SOLO: LOW/MED/HIGH."
            ),
            gold="LOW",
            meta={"temp": 37.2, "sat": 97},
        )

    def sample_task(self, seed: int) -> TaskSpec:
        tasks = [self._task_triage_1(), self._task_triage_2(), self._task_triage_3()]
        return tasks[seed % len(tasks)]

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="health.apply_rule",
                description="Aplica la regla sintética a (temp, sat) y devuelve LOW/MED/HIGH.",
                args_schema={
                    "type": "object",
                    "properties": {"temp": {"type": "number"}, "sat": {"type": "number"}},
                    "required": ["temp", "sat"],
                },
            )
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name != "health.apply_rule":
            return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

        temp = float(tool_args["temp"])
        sat = float(tool_args["sat"])

        if temp >= 39.0 or sat <= 92.0:
            label = "HIGH"
        elif temp >= 38.0 or sat <= 94.0:
            label = "MED"
        else:
            label = "LOW"

        return {"ok": True, "label": label}

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip().upper()
        expected = str(task.gold).strip().upper()
        ok = ans == expected
        return CheckResult(
            reward=1.0 if ok else 0.0,
            done=True,
            info={"task_id": task.id, "expected": expected, "got": ans},
        )

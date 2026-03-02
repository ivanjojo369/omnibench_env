# domains/coding.py
from __future__ import annotations

import re
from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


class CodingDomain:
    """
    Offline Coding domain.
    Uses "open_file" + "eval_int" pattern (deterministic, evaluable, tool-friendly).

    Final response is strict:
      - For fix tasks: respond with the EXACT replacement expression (e.g., "a * b")
      - For numeric tasks: respond with exact number as string.
    """

    name = "coding"

    _FILES: dict[str, str] = {
        "math_utils.py": (
            "def multiply(a, b):\n"
            "    # BUG: should multiply\n"
            "    return a + b\n"
            "\n"
            "def add(a, b):\n"
            "    return a + b\n"
        ),
        "string_utils.py": (
            "def shout(s: str) -> str:\n"
            "    # Should return uppercase + '!'\n"
            "    return s + '!'\n"
        ),
    }

    def _task_fix_multiply(self) -> TaskSpec:
        return TaskSpec(
            id="coding_fix_multiply_v1",
            instruction=(
                "Abre el archivo math_utils.py. Encuentra el bug en multiply(a,b).\n"
                "Tu respuesta final debe ser SOLO la expresión correcta del return para multiply.\n"
                "Ejemplo de formato: a * b"
            ),
            gold="a * b",
            meta={"file": "math_utils.py"},
        )

    def _task_fix_shout(self) -> TaskSpec:
        return TaskSpec(
            id="coding_fix_shout_v1",
            instruction=(
                "Abre el archivo string_utils.py. La función shout(s) debe devolver uppercase + '!'.\n"
                "Tu respuesta final debe ser SOLO la expresión correcta del return.\n"
                "Ejemplo: s.upper() + '!'"
            ),
            gold="s.upper() + '!'",
            meta={"file": "string_utils.py"},
        )

    def _task_eval_expression(self) -> TaskSpec:
        return TaskSpec(
            id="coding_eval_expression_v1",
            instruction=(
                "Calcula el valor de la expresión: (12 // 5) + (3 * 4).\n"
                "Responde SOLO con el número."
            ),
            gold="14",
        )

    def sample_task(self, seed: int) -> TaskSpec:
        tasks = [self._task_fix_multiply(), self._task_fix_shout(), self._task_eval_expression()]
        return tasks[seed % len(tasks)]

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="coding.open_file",
                description="Abre un archivo offline y devuelve su contenido.",
                args_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="coding.eval_int",
                description="Evalúa una expresión aritmética entera segura (//, +, -, *, paréntesis, enteros).",
                args_schema={
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                    "additionalProperties": False,
                },
            ),
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name == "coding.open_file":
            path = str(tool_args.get("path", "")).strip()
            text = self._FILES.get(path)
            if text is None:
                return {"ok": False, "error": "not_found", "path": path, "text": ""}
            return {"ok": True, "path": path, "text": text}

        if tool_name == "coding.eval_int":
            expr = str(tool_args.get("expr", "")).strip()
            # Safe allow-list: digits, whitespace, + - * / ( )  (we'll normalize / to // safely)
            allowed = set("0123456789+-*/() \t\n")
            if not expr or any(ch not in allowed for ch in expr) or "**" in expr:
                return {"ok": False, "error": "invalid_expr"}

            # Convert single "/" into "//" but DO NOT break existing "//"
            expr_norm = re.sub(r"(?<!/)/(?!/)", "//", expr)

            try:
                val = eval(expr_norm, {"__builtins__": {}}, {})
            except Exception as e:
                return {"ok": False, "error": "eval_error", "detail": str(e), "expr_norm": expr_norm}

            if not isinstance(val, int):
                return {"ok": False, "error": "not_int", "value_type": str(type(val))}
            return {"ok": True, "value": val}

        return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip()
        expected = str(task.gold).strip()
        ok = ans == expected
        return CheckResult(
            reward=1.0 if ok else 0.0,
            done=True,
            info={"task_id": task.id, "expected": expected, "got": ans},
        )
    
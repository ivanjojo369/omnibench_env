# domains/finance.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


def _q2(x: float) -> str:
    """Format to 2 decimals using Decimal for stable rounding."""
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(d, "f")


class FinanceDomain:
    """
    Offline 'Finance' domain.
    Tools are deterministic and fully evaluable.

    Pattern:
      - mode="tool" to compute something
      - mode="respond" with final answer (number/label) checked by gold
    """

    name = "finance"

    # --- Tasks ----------------------------------------------------------------

    def _task_compound_interest(self) -> TaskSpec:
        # M = P*(1+r/n)^(n*t)
        return TaskSpec(
            id="finance_compound_interest_v1",
            instruction=(
                "Calcula el monto final con interés compuesto.\n"
                "Datos: principal=1000, tasa_anual=0.05, años=2, comp=anual (n=1).\n"
                "Responde SOLO con el número con 2 decimales."
            ),
            gold="1102.50",
            meta={"principal": 1000, "rate": 0.05, "years": 2, "n": 1},
        )

    def _task_percent_change(self) -> TaskSpec:
        return TaskSpec(
            id="finance_percent_change_v1",
            instruction=(
                "Calcula el cambio porcentual de 80 a 100.\n"
                "Fórmula: (new-old)/old * 100.\n"
                "Responde SOLO con el número con 1 decimal (sin %)."
            ),
            gold="25.0",
            meta={"old": 80, "new": 100},
        )

    def _task_compare_options(self) -> TaskSpec:
        return TaskSpec(
            id="finance_compare_options_v1",
            instruction=(
                "Elige la mejor opción (A o B) por MAYOR retorno neto.\n"
                "Opción A: inversión=500, retorno=575.\n"
                "Opción B: inversión=500, retorno=570.\n"
                "Responde SOLO: A o B."
            ),
            gold="A",
            meta={"A": {"invest": 500, "return": 575}, "B": {"invest": 500, "return": 570}},
        )

    def sample_task(self, seed: int) -> TaskSpec:
        tasks = [
            self._task_compound_interest(),
            self._task_percent_change(),
            self._task_compare_options(),
        ]
        return tasks[seed % len(tasks)]

    # --- Tools ----------------------------------------------------------------

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="finance.compound",
                description="Calcula M = P*(1+r/n)^(n*t). Devuelve monto final numérico.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "principal": {"type": "number"},
                        "rate": {"type": "number"},
                        "years": {"type": "number"},
                        "n": {"type": "number"},
                    },
                    "required": ["principal", "rate", "years", "n"],
                },
            ),
            ToolSpec(
                name="finance.percent_change",
                description="Calcula el cambio porcentual de old a new: (new-old)/old*100.",
                args_schema={
                    "type": "object",
                    "properties": {"old": {"type": "number"}, "new": {"type": "number"}},
                    "required": ["old", "new"],
                },
            ),
            ToolSpec(
                name="finance.compare",
                description="Compara dos opciones por retorno neto (return-invest). Devuelve la mejor etiqueta.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "invest": {"type": "number"},
                                    "ret": {"type": "number"},
                                },
                                "required": ["label", "invest", "ret"],
                            },
                        }
                    },
                    "required": ["options"],
                },
            ),
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name == "finance.compound":
            p = float(tool_args["principal"])
            r = float(tool_args["rate"])
            t = float(tool_args["years"])
            n = float(tool_args["n"])
            value = p * (1.0 + r / n) ** (n * t)
            return {"ok": True, "value": value}

        if tool_name == "finance.percent_change":
            old = float(tool_args["old"])
            new = float(tool_args["new"])
            if old == 0:
                return {"ok": False, "error": "division_by_zero"}
            pct = (new - old) / old * 100.0
            return {"ok": True, "percent": pct}

        if tool_name == "finance.compare":
            opts = tool_args.get("options", [])
            if not isinstance(opts, list) or not opts:
                return {"ok": False, "error": "no_options"}
            best = None
            for o in opts:
                try:
                    label = str(o["label"])
                    invest = float(o["invest"])
                    ret = float(o["ret"])
                except Exception:
                    return {"ok": False, "error": "bad_option_format", "option": o}
                net = ret - invest
                if best is None or net > best["net"]:
                    best = {"label": label, "net": net}
            return {"ok": True, "best": best["label"], "net": best["net"]}

        return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

    # --- Checker --------------------------------------------------------------

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip()

        if task.id == "finance_compound_interest_v1":
            # Expect exactly 2 decimals
            expected = str(task.gold)
            # normalize: keep only number-ish
            try:
                ans_num = float(ans.replace(",", ""))
                ans_norm = _q2(ans_num)
            except Exception:
                ans_norm = ans
            ok = ans_norm == expected
            info = {"expected": expected, "got": ans_norm, "raw": ans}

        elif task.id == "finance_percent_change_v1":
            expected = str(task.gold)
            try:
                ans_num = float(ans.replace(",", ""))
                # 1 decimal rounding
                ans_norm = str(Decimal(str(ans_num)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
            except Exception:
                ans_norm = ans
            ok = ans_norm == expected
            info = {"expected": expected, "got": ans_norm, "raw": ans}

        elif task.id == "finance_compare_options_v1":
            expected = str(task.gold).strip().upper()
            ans_norm = ans.upper()
            ok = ans_norm == expected
            info = {"expected": expected, "got": ans_norm, "raw": ans}

        else:
            expected = str(task.gold).strip()
            ans_norm = ans
            ok = ans_norm == expected
            info = {"expected": expected, "got": ans_norm, "raw": ans}

        return CheckResult(
            reward=1.0 if ok else 0.0,
            done=True,
            info={"task_id": task.id, **info},
        )

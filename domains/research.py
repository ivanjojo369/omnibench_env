# domains/research.py
from __future__ import annotations

import re
from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


class ResearchDomain:
    """
    Offline 'Research' domain.
    Simulates a small document corpus with:
      - research.search(query) -> list of doc_ids
      - research.open(doc_id)  -> text

    Tasks require tool use (search/open) and end with mode="respond".
    """

    name = "research"

    _DOCS: dict[str, str] = {
        "R1": (
            "Title: OmniBench: Multi-domain Tool-Calling Environments\n"
            "Abstract: A benchmark with tool calls.\n"
            "Key metric: OB-Score\n"
            "Release: 2026\n"
        ),
        "R2": (
            "Title: SafetyBench: Refusal and Compliance\n"
            "Abstract: Measures safe refusal.\n"
            "Key metric: SB-Refusal\n"
            "Release: 2025\n"
        ),
        "R3": (
            "Title: WebNav: Offline Web Navigation Tasks\n"
            "Abstract: Page traversal and information extraction.\n"
            "Dataset: WN-Set\n"
            "Release: 2024\n"
        ),
        "R4": (
            "Title: FinanceEval: Deterministic Financial Reasoning\n"
            "Abstract: Interest, percentage, and option comparison.\n"
            "Primary measure: FE-Accuracy\n"
            "Release: 2025\n"
        ),
        "R5": (
            "Title: ToolSpec Guidelines\n"
            "Abstract: Tools should declare args_schema.\n"
            "Recommendation: Use JSON Schema with required fields.\n"
        ),
    }

    # --- Tasks ----------------------------------------------------------------

    def _task_find_ob_score(self) -> TaskSpec:
        return TaskSpec(
            id="research_find_metric_omnibench_v1",
            instruction=(
                "Usa research.search y research.open para encontrar cuál es el 'Key metric' de OmniBench. "
                "Responde SOLO con el nombre exacto."
            ),
            gold="OB-Score",
            meta={"hint": "OmniBench"},
        )

    def _task_find_sb_refusal(self) -> TaskSpec:
        return TaskSpec(
            id="research_find_metric_safetybench_v1",
            instruction=(
                "Encuentra el 'Key metric' de SafetyBench usando herramientas research.*. "
                "Responde SOLO con el nombre exacto."
            ),
            gold="SB-Refusal",
            meta={"hint": "SafetyBench"},
        )

    def _task_find_json_schema_reco(self) -> TaskSpec:
        return TaskSpec(
            id="research_find_recommendation_jsonschema_v1",
            instruction=(
                "Busca el documento que habla de ToolSpec y extrae la recomendación (línea que empieza con 'Recommendation:'). "
                "Responde SOLO con el texto DESPUÉS de 'Recommendation:'."
            ),
            gold="Use JSON Schema with required fields.",
            meta={"hint": "ToolSpec"},
        )

    def sample_task(self, seed: int) -> TaskSpec:
        tasks = [
            self._task_find_ob_score(),
            self._task_find_sb_refusal(),
            self._task_find_json_schema_reco(),
        ]
        return tasks[seed % len(tasks)]

    # --- Tools ----------------------------------------------------------------

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="research.search",
                description="Busca documentos por palabra clave (case-insensitive). Devuelve una lista de doc_ids.",
                args_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Texto a buscar"}},
                    "required": ["query"],
                },
            ),
            ToolSpec(
                name="research.open",
                description="Abre un documento por doc_id y devuelve su texto completo.",
                args_schema={
                    "type": "object",
                    "properties": {"doc_id": {"type": "string", "description": "ID del documento (ej. R1)"}},
                    "required": ["doc_id"],
                },
            ),
            ToolSpec(
                name="research.extract",
                description="Extrae usando regex (devuelve primer grupo capturado si existe).",
                args_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex con (grupo) capturable"},
                        "text": {"type": "string", "description": "Texto donde buscar"},
                    },
                    "required": ["pattern", "text"],
                },
            ),
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name == "research.search":
            q = str(tool_args.get("query", "")).strip().lower()
            if not q:
                return {"ok": True, "hits": []}
            hits = []
            for doc_id, text in self._DOCS.items():
                if q in text.lower():
                    hits.append(doc_id)
            hits.sort()
            return {"ok": True, "hits": hits}

        if tool_name == "research.open":
            doc_id = str(tool_args.get("doc_id", "")).strip()
            text = self._DOCS.get(doc_id)
            if text is None:
                return {"ok": False, "error": "not_found", "doc_id": doc_id, "text": ""}
            return {"ok": True, "doc_id": doc_id, "text": text}

        if tool_name == "research.extract":
            pattern = str(tool_args.get("pattern", ""))
            text = str(tool_args.get("text", ""))
            try:
                m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            except re.error as e:
                return {"ok": False, "error": "regex_error", "detail": str(e)}
            if not m:
                return {"ok": True, "match": None}
            if m.lastindex and m.lastindex >= 1:
                return {"ok": True, "match": m.group(1).strip()}
            return {"ok": True, "match": m.group(0).strip()}

        return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

    # --- Checker --------------------------------------------------------------

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip()

        expected = str(task.gold).strip()

        # For the recommendation task, normalize whitespace and trailing punctuation lightly
        if task.id == "research_find_recommendation_jsonschema_v1":
            ans_norm = re.sub(r"\s+", " ", ans).strip()
            exp_norm = re.sub(r"\s+", " ", expected).strip()
        else:
            ans_norm = ans
            exp_norm = expected

        ok = ans_norm == exp_norm

        return CheckResult(
            reward=1.0 if ok else 0.0,
            done=True,
            info={
                "task_id": task.id,
                "expected": expected,
                "got": ans_norm,
                "raw": ans,
            },
        )

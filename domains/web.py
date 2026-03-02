# domains/web.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


class WebDomain:
    """
    Offline 'Web' domain.
    Tools:
      - web.get(path): returns HTML of a simulated page.
      - web.extract(pattern, text): regex extract helper (keeps agent tool-calling style).

    Tasks are deterministic and fully evaluable by exact-match checker.
    """

    name = "web"

    # A tiny offline "website"
    _PAGES: dict[str, str] = {
        "/": (
            "<html><body>"
            "<h1>OmniBench</h1>"
            "<p>Welcome.</p>"
            "<a href='/about'>About</a> "
            "<a href='/contact'>Contact</a>"
            "</body></html>"
        ),
        "/about": (
            "<html><body>"
            "<h1>About</h1>"
            "<p>Version: v0.3</p>"
            "<p>Build: 2026-02-12</p>"
            "<a href='/'>Home</a>"
            "</body></html>"
        ),
        "/contact": (
            "<html><body>"
            "<h1>Contact</h1>"
            "<p>Email: support@omnibench.local</p>"
            "<p>Support code: W-7319</p>"
            "<a href='/'>Home</a>"
            "</body></html>"
        ),
        "/pricing": (
            "<html><body>"
            "<h1>Pricing</h1>"
            "<ul>"
            "<li>Starter: $9/mo</li>"
            "<li>Pro: $29/mo</li>"
            "<li>Enterprise: Contact sales</li>"
            "</ul>"
            "</body></html>"
        ),
    }

    # --- Task library ---------------------------------------------------------

    def _task_find_support_code(self) -> TaskSpec:
        return TaskSpec(
            id="web_find_support_code_v1",
            instruction=(
                "Usa herramientas web.* para encontrar el 'Support code' en la página /contact. "
                "Responde SOLO con el código exacto (ej. W-0000)."
            ),
            gold="W-7319",
            meta={"start_path": "/contact"},
        )

    def _task_find_version(self) -> TaskSpec:
        return TaskSpec(
            id="web_find_version_v1",
            instruction=(
                "Visita /about y extrae el valor de 'Version:'. "
                "Responde SOLO con la versión exacta (ej. v0.0)."
            ),
            gold="v0.3",
            meta={"start_path": "/about"},
        )

    def _task_find_pro_price(self) -> TaskSpec:
        return TaskSpec(
            id="web_find_pro_price_v1",
            instruction=(
                "Visita /pricing y encuentra el precio del plan 'Pro'. "
                "Responde SOLO con el número (sin $ ni /mo)."
            ),
            gold="29",
            meta={"start_path": "/pricing"},
        )

    def sample_task(self, seed: int) -> TaskSpec:
        # Deterministic selection by seed to keep reproducible eval
        tasks = [
            self._task_find_support_code(),
            self._task_find_version(),
            self._task_find_pro_price(),
        ]
        return tasks[seed % len(tasks)]

    # --- Tools ----------------------------------------------------------------

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="web.get",
                description="Obtiene HTML desde una ruta offline (por ejemplo: /, /contact, /about, /pricing).",
                args_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Ruta a solicitar (ej. /contact)"}},
                    "required": ["path"],
                },
            ),
            ToolSpec(
                name="web.extract",
                description="Extrae texto usando regex. Útil para sacar 'Support code', 'Version', precios, etc.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex con (grupo) capturable"},
                        "text": {"type": "string", "description": "Texto/HTML donde buscar"},
                    },
                    "required": ["pattern", "text"],
                },
            ),
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name == "web.get":
            path = str(tool_args.get("path", "/"))
            html = self._PAGES.get(path)
            if html is None:
                return {"ok": False, "status": 404, "path": path, "html": ""}
            return {"ok": True, "status": 200, "path": path, "html": html}

        if tool_name == "web.extract":
            pattern = str(tool_args.get("pattern", ""))
            text = str(tool_args.get("text", ""))
            try:
                m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            except re.error as e:
                return {"ok": False, "error": "regex_error", "detail": str(e)}
            if not m:
                return {"ok": True, "match": None}
            # return first capturing group if present; else full match
            if m.lastindex and m.lastindex >= 1:
                return {"ok": True, "match": m.group(1)}
            return {"ok": True, "match": m.group(0)}

        return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

    # --- Checker --------------------------------------------------------------

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip()

        # task-specific normalization
        if task.id == "web_find_pro_price_v1":
            # allow "$29" or "29/mo" etc -> keep digits only
            digits = re.findall(r"\d+", ans)
            ans_norm = digits[0] if digits else ans
        else:
            ans_norm = ans

        expected = str(task.gold).strip()
        ok = ans_norm == expected

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

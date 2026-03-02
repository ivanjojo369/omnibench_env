# domains/computer_use.py
from __future__ import annotations

from typing import Any

from omnibench_env.models import ToolSpec
from .base import CheckResult, TaskSpec


class ComputerUseDomain:
    """
    Offline Computer Use domain with a tiny UI state machine.
    Tools:
      - ui.get_state()
      - ui.click(target)
      - ui.type(target, text)

    Tasks are deterministic and reset internal state on sample_task.
    """

    name = "computer_use"

    def __init__(self):
        # simple UI state
        self._page = "home"
        self._dark_mode = False
        self._search_box = ""
        self._settings_wifi = False

    def _reset_ui(self):
        self._page = "home"
        self._dark_mode = False
        self._search_box = ""
        self._settings_wifi = False

    def _task_toggle_dark_mode(self) -> TaskSpec:
        self._reset_ui()
        return TaskSpec(
            id="cu_toggle_dark_mode_v1",
            instruction=(
                "Usa herramientas ui.* para activar el dark mode en Settings.\n"
                "Al finalizar, responde EXACTAMENTE: DONE"
            ),
            gold="DONE",
        )

    def _task_search_and_open(self) -> TaskSpec:
        self._reset_ui()
        return TaskSpec(
            id="cu_search_open_docs_v1",
            instruction=(
                "En la página Home hay una caja de búsqueda.\n"
                "1) escribe 'docs'\n"
                "2) haz click en 'open_docs'\n"
                "Cuando estés en la página 'docs', responde EXACTAMENTE: DONE"
            ),
            gold="DONE",
        )

    def _task_enable_wifi(self) -> TaskSpec:
        self._reset_ui()
        return TaskSpec(
            id="cu_enable_wifi_v1",
            instruction=(
                "Ve a Settings y activa WiFi.\n"
                "Cuando wifi esté ON, responde EXACTAMENTE: DONE"
            ),
            gold="DONE",
        )

    def sample_task(self, seed: int) -> TaskSpec:
        tasks = [self._task_toggle_dark_mode(), self._task_search_and_open(), self._task_enable_wifi()]
        return tasks[seed % len(tasks)]

    def tools_for_task(self, task: TaskSpec) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="ui.get_state",
                description="Devuelve estado actual (page, dark_mode, wifi, search_box).",
                args_schema={"type": "object", "properties": {}},
            ),
            ToolSpec(
                name="ui.click",
                description="Hace click en un target (por ejemplo: settings_button, dark_mode_toggle, open_docs, wifi_toggle).",
                args_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
            ),
            ToolSpec(
                name="ui.type",
                description="Escribe texto en un target (por ejemplo: search_box).",
                args_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["target", "text"],
                },
            ),
        ]

    def call_tool(self, tool_name: str, tool_args: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        if tool_name == "ui.get_state":
            return {
                "ok": True,
                "page": self._page,
                "dark_mode": self._dark_mode,
                "wifi": self._settings_wifi,
                "search_box": self._search_box,
                "clickables": self._clickables(),
            }

        if tool_name == "ui.type":
            target = str(tool_args.get("target", ""))
            text = str(tool_args.get("text", ""))
            if target != "search_box":
                return {"ok": False, "error": "unknown_target", "target": target}
            if self._page != "home":
                return {"ok": False, "error": "not_on_home", "page": self._page}
            self._search_box = text
            return {"ok": True, "typed": {"target": target, "text": text}}

        if tool_name == "ui.click":
            target = str(tool_args.get("target", ""))
            if target not in self._clickables():
                return {"ok": False, "error": "not_clickable", "target": target, "page": self._page}

            # home interactions
            if self._page == "home":
                if target == "settings_button":
                    self._page = "settings"
                elif target == "open_docs":
                    if self._search_box.strip().lower() == "docs":
                        self._page = "docs"
                    else:
                        return {"ok": False, "error": "search_box_not_docs", "search_box": self._search_box}

            # settings interactions
            elif self._page == "settings":
                if target == "dark_mode_toggle":
                    self._dark_mode = not self._dark_mode
                elif target == "wifi_toggle":
                    self._settings_wifi = not self._settings_wifi
                elif target == "back_home":
                    self._page = "home"

            # docs interactions
            elif self._page == "docs":
                if target == "back_home":
                    self._page = "home"

            return {"ok": True, "clicked": target, "state": {"page": self._page, "dark_mode": self._dark_mode, "wifi": self._settings_wifi}}

        return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}

    def _clickables(self) -> list[str]:
        if self._page == "home":
            return ["settings_button", "open_docs"]
        if self._page == "settings":
            return ["dark_mode_toggle", "wifi_toggle", "back_home"]
        if self._page == "docs":
            return ["back_home"]
        return []

    def check_final(self, message: str, task: TaskSpec) -> CheckResult:
        ans = (message or "").strip()
        expected = str(task.gold).strip()

        # Ensure the task condition is truly met (not just "DONE")
        condition_ok = True
        if task.id == "cu_toggle_dark_mode_v1":
            condition_ok = self._dark_mode is True and self._page == "settings"
        elif task.id == "cu_search_open_docs_v1":
            condition_ok = self._page == "docs"
        elif task.id == "cu_enable_wifi_v1":
            condition_ok = self._settings_wifi is True and self._page == "settings"

        ok = (ans == expected) and condition_ok

        return CheckResult(
            reward=1.0 if ok else 0.0,
            done=True,
            info={
                "task_id": task.id,
                "expected": expected,
                "got": ans,
                "condition_ok": condition_ok,
                "final_state": {"page": self._page, "dark_mode": self._dark_mode, "wifi": self._settings_wifi},
            },
        )

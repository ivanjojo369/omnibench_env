# scripts/smoke_local.py
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Optional

# Local env import (funciona si corres desde la raíz del env)
from server.omnibench_env_environment import OmnibenchEnvironment


def _extract_symbol_from_instruction(text: str) -> str:
    # e.g. "get_prices('XYZ')" -> XYZ
    m = re.search(r"get_prices\(\s*['\"]([^'\"]+)['\"]\s*\)", text)
    return m.group(1) if m else "XYZ"


def _avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def run_local(domain: str, seed: int) -> None:
    env = OmnibenchEnvironment()
    obs = env.reset(seed=seed, domain=domain)
    print("\n== LOCAL RESET ==")
    print("domain:", obs.domain, "task_id:", obs.task_id)
    print("metadata:", obs.metadata)
    print("tools:", [t.name for t in obs.available_tools])

    # tool step (elige caso fácil: research o finance)
    tool_names = {t.name for t in obs.available_tools}

    if "search_docs" in tool_names:
        obs = env.step(type("A", (), {"mode": "tool", "tool_name": "search_docs", "tool_args": {"query": "OpenEnv standard"}, "message": None, "metadata": {}}))
        # respond (con citas dummy)
        answer = "OpenEnv core API is reset() + step() + state() + schema() exposed over HTTP/WS. [docA] [docB]"
        obs = env.step(type("A", (), {"mode": "respond", "tool_name": None, "tool_args": {}, "message": answer, "metadata": {}}))
        print("\n== LOCAL FINAL ==")
        print("reward:", obs.reward, "done:", obs.done)
        print("last_tool_result:", obs.last_tool_result)
        return

    if "get_prices" in tool_names:
        sym = _extract_symbol_from_instruction(obs.instruction)
        obs = env.step(type("A", (), {"mode": "tool", "tool_name": "get_prices", "tool_args": {"symbol": sym}, "message": None, "metadata": {}}))
        series = (obs.last_tool_result or {}).get("series", [])
        series_f = [float(x) for x in series if isinstance(x, (int, float))]
        answer = f"{_avg(series_f):.2f}"
        obs = env.step(type("A", (), {"mode": "respond", "tool_name": None, "tool_args": {}, "message": answer, "metadata": {}}))
        print("\n== LOCAL FINAL ==")
        print("symbol:", sym, "answer:", answer)
        print("reward:", obs.reward, "done:", obs.done)
        return

    raise RuntimeError(f"No tengo handler para tools={sorted(tool_names)}. Usa domain=research o domain=finance para smoke.")


def run_http(base: str, domain: str, seed: int) -> None:
    import requests  # normalmente ya viene

    def post(path: str, payload: dict) -> dict:
        r = requests.post(f"{base}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    print("\n== HTTP RESET ==")
    reset = post("/reset", {"seed": seed, "domain": domain})
    obs = reset["observation"]
    md = obs.get("metadata", {}) or {}
    eid = md.get("episode_id")  # lo pone el env nuevo
    print("domain:", obs.get("domain"), "task_id:", obs.get("task_id"))
    print("episode_id:", eid)
    print("tools:", [t.get("name") for t in obs.get("available_tools", [])])

    if not eid:
        raise RuntimeError("No recibí observation.metadata.episode_id; revisa que el env nuevo sí esté dentro del contenedor.")

    tool_names = {t.get("name") for t in obs.get("available_tools", [])}

    # tool step
    if "search_docs" in tool_names:
        step1 = post("/step", {
            "action": {
                "mode": "tool",
                "tool_name": "search_docs",
                "tool_args": {"query": "OpenEnv standard"},
                "metadata": {"episode_id": eid}
            }
        })
        print("\n== HTTP TOOL ==")
        print("last_tool_result keys:", list((step1["observation"].get("last_tool_result") or {}).keys()))

        answer = "OpenEnv core API is reset() + step() + state() + schema() exposed over HTTP/WS. [docA] [docB]"
        step2 = post("/step", {
            "action": {
                "mode": "respond",
                "message": answer,
                "metadata": {"episode_id": eid}
            }
        })
        print("\n== HTTP FINAL ==")
        print("reward:", step2["reward"], "done:", step2["done"])
        return

    if "get_prices" in tool_names:
        sym = _extract_symbol_from_instruction(obs.get("instruction", ""))
        step1 = post("/step", {
            "action": {
                "mode": "tool",
                "tool_name": "get_prices",
                "tool_args": {"symbol": sym},
                "metadata": {"episode_id": eid}
            }
        })
        series = (step1["observation"].get("last_tool_result") or {}).get("series", [])
        series_f = [float(x) for x in series if isinstance(x, (int, float))]
        answer = f"{_avg(series_f):.2f}"

        step2 = post("/step", {
            "action": {
                "mode": "respond",
                "message": answer,
                "metadata": {"episode_id": eid}
            }
        })
        print("\n== HTTP FINAL ==")
        print("symbol:", sym, "answer:", answer)
        print("reward:", step2["reward"], "done:", step2["done"])
        return

    raise RuntimeError(f"No handler HTTP para tools={sorted(tool_names)}. Usa domain=research o domain=finance para smoke.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="research", choices=["research", "finance"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--http", default=None, help="Base URL, e.g. http://localhost:8001 (si no se da, corre local)")
    args = ap.parse_args()

    if args.http:
        run_http(args.http.rstrip("/"), args.domain, args.seed)
    else:
        run_local(args.domain, args.seed)


if __name__ == "__main__":
    main()

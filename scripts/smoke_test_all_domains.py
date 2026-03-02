import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

DOMAINS = ["finance", "agent_safety", "healthcare", "web", "research", "coding", "computer_use"]


# ---------------- HTTP ----------------

def _post(base_url: str, path: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    r = requests.post(f"{base_url}{path}", json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} -> {r.status_code}\n{r.text[:2000]}")
    return r.json()


def _get(base_url: str, path: str, timeout_s: float) -> Dict[str, Any]:
    r = requests.get(f"{base_url}{path}", timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code}\n{r.text[:2000]}")
    return r.json()


def _reward_is_one(x: Any, tol: float = 1e-6) -> bool:
    try:
        return float(x) >= (1.0 - tol)
    except Exception:
        return False


# ---------------- Trace ----------------

def _write_trace(trace_dir: str, domain: str, trace: Dict[str, Any]) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(trace_dir, f"{domain}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    return path


# ---------------- Tools ----------------

def _tool_names(tools: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict):
            name = t.get("name") or t.get("tool_name") or t.get("id")
            if isinstance(name, str):
                out.append(name)
    return out


def _step_tool(base_url: str, eid: str, tool_name: str, tool_args: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    payload = {"episode_id": eid, "action": {"mode": "tool", "tool_name": tool_name, "tool_args": tool_args}}
    return _post(base_url, "/step", payload, timeout_s)


def _obs(step_out: Dict[str, Any]) -> Dict[str, Any]:
    return step_out.get("observation") or {}


def _last_tool_result(step_out: Dict[str, Any]) -> Any:
    o = _obs(step_out)
    return o.get("last_tool_result") if o.get("last_tool_result") is not None else o


def _is_tool_error(x: Any) -> bool:
    return isinstance(x, dict) and "error" in x


def _try_tool(
    base_url: str,
    eid: str,
    tool_name: str,
    arg_candidates: List[Dict[str, Any]],
    timeout_s: float,
    verbose: bool,
    label: str,
    trace: Dict[str, Any],
) -> Any:
    """
    Prueba múltiples tool_args y devuelve el primer last_tool_result no-error.
    Siempre registra cada llamada en trace.
    """
    for args in arg_candidates:
        out = _step_tool(base_url, eid, tool_name, args, timeout_s)
        r = _last_tool_result(out)
        trace.setdefault("tool_calls", []).append(
            {"tool_name": tool_name, "tool_args": args, "raw_step_out": out, "last_tool_result": r}
        )
        if verbose:
            print(f"[{label}] {tool_name} args={args} -> {str(r)[:260]}")
        if r is not None and not _is_tool_error(r):
            return r
    return None


def _scan_trace_for(extractor_fn, trace: Dict[str, Any]) -> Optional[str]:
    for call in trace.get("tool_calls", []):
        v = extractor_fn(call.get("last_tool_result"))
        if v:
            return v
    return None


# ---------------- Extractors ----------------

def _extract_support_code(obj: Any) -> Optional[str]:
    blob = json.dumps(obj, ensure_ascii=False)
    m = re.search(r"\b[A-Z]-\d{4}\b", blob)
    return m.group(0) if m else None


def _clean_exact(s: str) -> str:
    # limpia SOLO bordes (sin cambiar el contenido interno)
    s = s.strip()
    s = s.strip('"\'')

    # recorta puntuación final común que suele colarse
    while s and s[-1] in " \t\r\n.,;:!?\"'":
        s = s[:-1]
        s = s.rstrip()
    return s


def _extract_key_metric(obj: Any) -> Optional[str]:
    """
    Extrae el 'Key metric' de forma segura sin confundir IDs de documentos (p.ej. 'R1') con la métrica.
    Prioriza campos típicos de texto ('text', 'content', 'match') antes de escanear otros valores.
    """
    if obj is None:
        return None
    if isinstance(obj, dict) and "error" in obj:
        return None

    def _looks_like_doc_id(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]{1,3}\d{1,4}", s.strip()))

    if isinstance(obj, str):
        s = _clean_exact(obj)
        if not s:
            return None
        m = re.search(r"key\s*metric\s*[:=\-]\s*([^\n\r,]+)", s, flags=re.IGNORECASE)
        if m:
            return _clean_exact(m.group(1))
        # No devuelvas IDs tipo 'R1'
        if _looks_like_doc_id(s):
            return None
        # Devuelve string directo solo si parece una respuesta real y no un blob largo
        if len(s) <= 64 and "\n" not in s and "{" not in s and "}" not in s:
            return s
        return None

    if isinstance(obj, list):
        for it in obj:
            v = _extract_key_metric(it)
            if v:
                return v
        return None

    if isinstance(obj, dict):
        # 1) Prioriza texto/contenido/match
        for k in ("match", "text", "content", "body", "snippet"):
            got = _extract_key_metric(obj.get(k))
            if got:
                return got

        # 2) Campos explícitos de métrica
        for k, v in obj.items():
            if isinstance(k, str) and "metric" in k.lower() and isinstance(v, str) and v.strip():
                vv = _clean_exact(v)
                return None if _looks_like_doc_id(vv) else vv

        for k in ("key_metric", "keyMetric", "metric", "metric_name", "metricName", "name", "value", "answer"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                vv = _clean_exact(v)
                return None if _looks_like_doc_id(vv) else vv

        # 3) Recorre el resto
        for v in obj.values():
            got = _extract_key_metric(v)
            if got:
                return got

    blob = json.dumps(obj, ensure_ascii=False)
    m = re.search(r"key\s*metric\s*[:=\-]\s*([^\n\r,]+)", blob, flags=re.IGNORECASE)
    if m:
        vv = _clean_exact(m.group(1))
        return None if _looks_like_doc_id(vv) else vv
    return None

def _has_hits(search_result: Any) -> bool:
    if isinstance(search_result, dict) and isinstance(search_result.get("hits"), list):
        return len(search_result["hits"]) > 0
    if isinstance(search_result, list):
        return len(search_result) > 0
    return False


def _hit_list(search_result: Any) -> List[Any]:
    if isinstance(search_result, dict) and isinstance(search_result.get("hits"), list):
        return search_result["hits"]
    if isinstance(search_result, list):
        return search_result
    return []


def _all_doc_ids_from_hits(search_result: Any, limit: int = 5) -> List[str]:
    hits = _hit_list(search_result)
    out: List[str] = []
    for h in hits:
        if isinstance(h, dict):
            for k in ("doc_id", "id", "ref_id", "ref", "url", "link"):
                v = h.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
        elif isinstance(h, str) and h.strip():
            out.append(h.strip())
        if len(out) >= limit:
            break
    # dedup preservando orden
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


# ---------------- Deterministic solvers ----------------

def solve_finance(instr: str) -> Optional[str]:
    p = re.search(r"principal\s*=\s*([0-9]+(?:\.[0-9]+)?)", instr)
    r = re.search(r"tasa_anual\s*=\s*([0-9]+(?:\.[0-9]+)?)", instr)
    y = re.search(r"años\s*=\s*([0-9]+(?:\.[0-9]+)?)", instr)
    n = re.search(r"n\s*=\s*([0-9]+)", instr)
    if not (p and r and y):
        return None
    P = float(p.group(1))
    R = float(r.group(1))
    Y = float(y.group(1))
    N = int(n.group(1)) if n else 1
    A = P * (1.0 + R / N) ** (N * Y)
    return f"{A:.2f}"


def solve_agent_safety(instr: str) -> str:
    m = re.search(r"Solicitud\s*:\s*'([^']+)'", instr, flags=re.DOTALL)
    req = (m.group(1) if m else instr).lower()
    harmful = ["explosivo", "bomba", "arma", "veneno", "hack", "malware", "explosive", "bomb", "poison"]
    return "REFUSE" if any(k in req for k in harmful) else "OK"


def solve_healthcare(instr: str) -> Optional[str]:
    mt = re.search(r"temp\s*=\s*([0-9]+(?:\.[0-9]+)?)", instr)
    ms = re.search(r"sat\s*=\s*([0-9]+(?:\.[0-9]+)?)", instr)
    if not (mt and ms):
        return None
    temp = float(mt.group(1))
    sat = float(ms.group(1))
    if temp >= 39.0 or sat <= 92.0:
        return "HIGH"
    if temp >= 38.0 or sat <= 94.0:
        return "MED"
    return "LOW"


# ---------------- UI helpers (computer_use) ----------------

def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk(it)


def _node_text(node: Dict[str, Any]) -> str:
    for k in ("text", "label", "name", "title", "value"):
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _node_id(node: Dict[str, Any]) -> Optional[Any]:
    for k in ("element_id", "elementId", "id", "node_id", "nodeId", "uid", "key"):
        if k in node and isinstance(node[k], (str, int)):
            return node[k]
    return None


def _node_bounds(node: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = node.get("bounds") or node.get("bbox") or node.get("rect")
    if isinstance(b, dict):
        x = b.get("x")
        y = b.get("y")
        w = b.get("w") or b.get("width")
        h = b.get("h") or b.get("height")
        if all(isinstance(v, (int, float)) for v in (x, y, w, h)):
            return float(x), float(y), float(w), float(h)
    return None


def _node_center(node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bb = _node_bounds(node)
    if bb:
        x, y, w, h = bb
        return x + w / 2, y + h / 2
    if isinstance(node.get("x"), (int, float)) and isinstance(node.get("y"), (int, float)):
        return float(node["x"]), float(node["y"])
    return None


def _node_toggle_point(node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Para filas tipo 'Dark mode' donde el switch suele estar a la derecha.
    """
    bb = _node_bounds(node)
    if not bb:
        return None
    x, y, w, h = bb
    return x + (w * 0.88), y + (h / 2)


def _find_nodes(state: Any, keywords: List[str]) -> List[Dict[str, Any]]:
    kws = [k.lower() for k in keywords]
    out: List[Dict[str, Any]] = []
    for n in _walk(state):
        if not isinstance(n, dict):
            continue
        t = _node_text(n).lower()
        if t and all(k in t for k in kws):
            out.append(n)
    return out


def _state_looks_dark(state: Any) -> bool:
    # Prefer explicit top-level flag if present (this is what the env exposes in your screenshots)
    if isinstance(state, dict) and isinstance(state.get("dark_mode"), bool):
        return state["dark_mode"] is True

    if state is None:
        return False

    # Otherwise search nested structures conservatively
    for n in _walk(state):
        if not isinstance(n, dict):
            continue
        for k, v in n.items():
            if isinstance(k, str) and "dark" in k.lower() and isinstance(v, bool) and v:
                return True
            if isinstance(k, str) and "theme" in k.lower() and isinstance(v, str) and "dark" in v.lower():
                return True

    blob = json.dumps(state, ensure_ascii=False).lower()
    # Only treat it as dark if the VALUE indicates dark, not merely the presence of a key name.
    return any(s in blob for s in ['theme":"dark', 'dark_mode":true', 'modo oscuro":true', 'dark mode":true'])

def _ui_get_state(base_url: str, eid: str, tool_list: List[str], timeout_s: float, verbose: bool, trace: Dict[str, Any]) -> Any:
    t = next((x for x in tool_list if "get_state" in x.lower()), None)
    if not t:
        return None
    return _try_tool(base_url, eid, t, [{}, {"mode": "full"}, {"full": True}, {"include_elements": True}], timeout_s, verbose, "ui.get_state", trace)


def _ui_click(base_url: str, eid: str, tool_list: List[str], candidates: List[Dict[str, Any]], timeout_s: float, verbose: bool, trace: Dict[str, Any], label: str) -> Any:
    t = next((x for x in tool_list if "click" in x.lower()), None)
    if not t:
        return None
    # añadimos variantes comunes de target_text / selector
    expanded: List[Dict[str, Any]] = []
    for c in candidates:
        expanded.append(c)
        if "text" in c and isinstance(c["text"], str):
            expanded.append({"target_text": c["text"]})
            expanded.append({"contains": c["text"]})
    return _try_tool(base_url, eid, t, expanded, timeout_s, verbose, label, trace)


def _ui_type(base_url: str, eid: str, tool_list: List[str], text: str, timeout_s: float, verbose: bool, trace: Dict[str, Any]) -> Any:
    t = next((x for x in tool_list if "type" in x.lower()), None)
    if not t:
        return None
    return _try_tool(base_url, eid, t, [{"text": text}, {"value": text}, {"input": text}, {"keys": text}], timeout_s, verbose, "ui.type", trace)




def _clickables_list(state: Any) -> List[str]:
    if isinstance(state, dict) and isinstance(state.get("clickables"), list):
        return [x for x in state["clickables"] if isinstance(x, str) and x.strip()]
    return []


def _ui_click_id(base_url: str, eid: str, tool_list: List[str], clickable_id: str, timeout_s: float, verbose: bool, trace: Dict[str, Any], label: str) -> Any:
    # Many envs accept id/target/element_id; try a few
    t = next((x for x in tool_list if "click" in x.lower()), None)
    if not t:
        return None
    return _try_tool(
        base_url,
        eid,
        t,
        [
            {"id": clickable_id},
            {"target": clickable_id},
            {"element_id": clickable_id},
            {"node_id": clickable_id},
            {"key": clickable_id},
        ],
        timeout_s,
        verbose,
        label,
        trace,
    )


def _pick_clickable(clickables: List[str], contains_any: List[str]) -> Optional[str]:
    low = [c.lower() for c in clickables]
    for needle in contains_any:
        n = needle.lower()
        for i, c in enumerate(low):
            if n in c:
                return clickables[i]
    return None

def _click_by_text_variants(base_url: str, eid: str, tool_list: List[str], words: List[str], timeout_s: float, verbose: bool, trace: Dict[str, Any], label: str) -> None:
    for w in words:
        _ui_click(base_url, eid, tool_list, [{"text": w}, {"label": w}, {"query": w}, {"pattern": w}, {"name": w}], timeout_s, verbose, trace, f"{label}:{w}")


def _click_by_nodes(base_url: str, eid: str, tool_list: List[str], nodes: List[Dict[str, Any]], timeout_s: float, verbose: bool, trace: Dict[str, Any], label: str) -> None:
    for n in nodes[:14]:
        nid = _node_id(n)
        ctr = _node_center(n)
        tog = _node_toggle_point(n)
        cand: List[Dict[str, Any]] = []
        if nid is not None:
            cand += [{"element_id": nid}, {"elementId": nid}, {"id": nid}, {"target": nid}, {"node_id": nid}, {"nodeId": nid}]
        if ctr is not None:
            x, y = ctr
            cand += [{"x": x, "y": y}, {"coords": [x, y]}]
        if tog is not None:
            x, y = tog
            cand += [{"x": x, "y": y}, {"coords": [x, y]}]
        if not cand:
            t = _node_text(n)
            if t:
                cand += [{"text": t}, {"label": t}, {"query": t}]
        if cand:
            _ui_click(base_url, eid, tool_list, cand, timeout_s, verbose, trace, f"{label}:{_node_text(n)[:24]}")


# ---------------- Smoke per-domain ----------------

def smoke_one(base_url: str, domain: str, timeout_s: float, verbose: bool, trace_dir: str) -> None:
    trace: Dict[str, Any] = {"domain": domain}

    reset = _post(base_url, "/reset", {"domain": domain}, timeout_s)
    trace["reset_raw"] = reset

    eid = reset.get("episode_id") or reset.get("episodeId") or reset.get("id")
    if not eid:
        path = _write_trace(trace_dir, domain, trace)
        raise RuntimeError(f"[{domain}] /reset sin episode_id (trace={path})")

    obs0 = reset.get("observation") or {}
    instr = obs0.get("instruction") or ""
    tools_raw = obs0.get("available_tools") or []
    tool_list = _tool_names(tools_raw)

    if verbose:
        print(f"[{domain}] reset OK episode_id={eid}")
        print(f"[{domain}] instruction preview: {instr[:320]}")
        print(f"[{domain}] tools: {tool_list}")

    last_tool_result: Any = None

    # -------- Tools --------

    if domain == "web" and tool_list:
        open_candidates = [t for t in tool_list if any(k in t.lower() for k in ("open", "goto", "navigate", "get"))]
        find_candidates = [t for t in tool_list if any(k in t.lower() for k in ("find", "search", "extract"))]

        code: Optional[str] = None

        # 1) open /contact
        for tname in (open_candidates or tool_list):
            for args in ({"url": "/contact"}, {"path": "/contact"}, {"page": "/contact"}, {"route": "/contact"}):
                out = _step_tool(base_url, eid, tname, args, timeout_s)
                r = _last_tool_result(out)
                trace.setdefault("tool_calls", []).append({"tool_name": tname, "tool_args": args, "raw_step_out": out, "last_tool_result": r})
                last_tool_result = r
                code = _extract_support_code(last_tool_result)
                if verbose:
                    print(f"[web.open] {tname} args={args} -> code={code}")
                if code:
                    break
            if code:
                break

        # 2) find/extract "Support code"
        if not code:
            for tname in (find_candidates or tool_list):
                r = _try_tool(
                    base_url, eid, tname,
                    [{"query": "Support code"}, {"text": "Support code"}, {"pattern": "Support code"}, {"needle": "Support code"}, {}],
                    timeout_s, verbose, label="web.find", trace=trace
                )
                if r is not None:
                    last_tool_result = r
                    code = _extract_support_code(last_tool_result)
                    if code:
                        break

        if not _extract_support_code(last_tool_result):
            c2 = _scan_trace_for(_extract_support_code, trace)
            if c2:
                last_tool_result = c2

    elif domain == "research" and tool_list:
        search_tools = [t for t in tool_list if t == "research.search"] or [t for t in tool_list if "search" in t.lower()] or tool_list
        open_tools = [t for t in tool_list if t == "research.open"] or [t for t in tool_list if "open" in t.lower()]
        extract_tools = [t for t in tool_list if t == "research.extract"] or [t for t in tool_list if "extract" in t.lower()]

        queries = [
            "OmniBench key metric",
            "OmniBench",
            "key metric OmniBench",
            "metric OmniBench",
            "",
            "*",
        ]

        hits = None
        for q in queries:
            for tname in search_tools:
                hits = _try_tool(
                    base_url, eid, tname,
                    [{"query": q}, {"q": q}, {"text": q}],
                    timeout_s, verbose, label="research.search", trace=trace
                )
                if hits is not None and _has_hits(hits):
                    break
            if hits is not None and _has_hits(hits):
                break

        # Important: hits are doc IDs (e.g. 'R1'), not the metric.
        km: Optional[str] = None
        doc_ids = _all_doc_ids_from_hits(hits, limit=5) if _has_hits(hits) else []

        def _extract_from_opened(opened_obj: Any) -> Optional[str]:
            # 1) Parse directly from opened text (your env returns 'Key metric: ...' in open.text)
            if isinstance(opened_obj, dict) and isinstance(opened_obj.get("text"), str):
                got = _extract_key_metric(opened_obj["text"])
                if got:
                    return got
            got = _extract_key_metric(opened_obj)
            if got:
                return got

            # 2) If backend supports it, try research.extract
            for tname in extract_tools:
                extracted = _try_tool(
                    base_url, eid, tname,
                    [
                        {"pattern": "Key metric"},
                        {"query": "Key metric"},
                        {"text": "Key metric"},
                        {"needle": "Key metric"},
                        {"question": "What is the Key metric of OmniBench? Return ONLY the metric name."},
                        {"prompt": "Find OmniBench Key metric. Return ONLY the metric name."},
                        {},
                    ],
                    timeout_s, verbose, label="research.extract", trace=trace
                )
                if extracted is not None:
                    cand = _extract_key_metric(extracted)
                    if cand:
                        return cand
            return None

        last_tool_result = hits

        if open_tools and doc_ids:
            for doc_id in doc_ids:
                opened = None
                for tname in open_tools:
                    opened = _try_tool(
                        base_url, eid, tname,
                        [{"doc_id": doc_id}, {"id": doc_id}, {"ref_id": doc_id}, {"url": doc_id}],
                        timeout_s, verbose, label="research.open", trace=trace
                    )
                    if opened is not None:
                        break
                if opened is None:
                    continue
                last_tool_result = opened
                cand = _extract_from_opened(opened)
                if cand:
                    km = cand
                    last_tool_result = cand  # string exacto
                    break

        if not km:
            km = _scan_trace_for(_extract_key_metric, trace)
            if km:
                last_tool_result = km

    elif domain == "computer_use" and tool_list:
        state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace)
        last_tool_result = state

        # ID-first deterministic policy (matches your env: clickables=['settings_button', ...])
        for _ in range(12):
            state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state
            if isinstance(state, dict) and state.get("dark_mode") is True:
                break

            clickables = _clickables_list(state)

            # 1) Enter Settings
            if "settings_button" in clickables and (not isinstance(state, dict) or state.get("page") in (None, "home")):
                _ui_click_id(base_url, eid, tool_list, "settings_button", timeout_s, verbose, trace, "ui.settings_id")
                continue

            # 2) Click something that looks like the dark-mode toggle
            cand = _pick_clickable(clickables, ["dark_mode", "dark", "oscuro"])
            if cand:
                _ui_click_id(base_url, eid, tool_list, cand, timeout_s, verbose, trace, "ui.dark_id")
                continue

            # 3) Sometimes nested under Appearance/Theme first
            cand = _pick_clickable(clickables, ["appearance", "theme", "tema", "apariencia"])
            if cand:
                _ui_click_id(base_url, eid, tool_list, cand, timeout_s, verbose, trace, "ui.appearance_id")
                continue

            # 4) If typing helps search within settings
            if isinstance(state, dict) and isinstance(state.get("search_box"), str):
                _ui_type(base_url, eid, tool_list, "dark", timeout_s, verbose, trace)
                _ui_type(base_url, eid, tool_list, "oscuro", timeout_s, verbose, trace)

            # 5) Legacy fallbacks (if env exposes richer tree)
            settings_words = ["Settings", "Ajustes", "Configuración", "Configuracion"]
            dark_words = ["Dark mode", "Dark Mode", "Modo oscuro", "Tema oscuro", "Dark", "Oscuro"]
            appearance_words = ["Appearance", "Theme", "Tema", "Apariencia"]

            _click_by_text_variants(base_url, eid, tool_list, settings_words, timeout_s, verbose, trace, "ui.settings")
            state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state
            _click_by_nodes(
                base_url, eid, tool_list,
                _find_nodes(state, ["settings"]) + _find_nodes(state, ["ajustes"]) + _find_nodes(state, ["configuración"]) + _find_nodes(state, ["configuracion"]),
                timeout_s, verbose, trace, "ui.settings_node"
            )
            state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state

            _click_by_text_variants(base_url, eid, tool_list, dark_words, timeout_s, verbose, trace, "ui.dark")
            state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state
            _click_by_nodes(
                base_url, eid, tool_list,
                _find_nodes(state, ["dark", "mode"]) + _find_nodes(state, ["modo", "oscuro"]) + _find_nodes(state, ["dark"]) + _find_nodes(state, ["oscuro"]),
                timeout_s, verbose, trace, "ui.dark_node"
            )
            state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state

            if state is not None and not _state_looks_dark(state):
                _click_by_text_variants(base_url, eid, tool_list, appearance_words, timeout_s, verbose, trace, "ui.appearance")
                state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state
                _click_by_text_variants(base_url, eid, tool_list, ["Dark", "Oscuro"], timeout_s, verbose, trace, "ui.dark_simple")
                state = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state

        last_tool_result = _ui_get_state(base_url, eid, tool_list, timeout_s, verbose, trace) or state
        if verbose and isinstance(last_tool_result, dict):
            print(f"[computer_use] dark_mode={last_tool_result.get('dark_mode')}, page={last_tool_result.get('page')}, clickables={last_tool_result.get('clickables')}")


    # -------- Respond --------

    answer: Optional[str] = None
    if domain == "finance":
        answer = solve_finance(instr)
    elif domain == "agent_safety":
        answer = solve_agent_safety(instr)
    elif domain == "healthcare":
        answer = solve_healthcare(instr)
    elif domain == "coding":
        answer = "a * b"
    elif domain == "web":
        answer = _extract_support_code(last_tool_result)
    elif domain == "research":
        # intentamos con el extractor (last_tool_result ya puede ser string exacto)
        answer = _extract_key_metric(last_tool_result)
    elif domain == "computer_use":
        answer = "DONE"

    if not answer:
        path = _write_trace(trace_dir, domain, trace)
        raise RuntimeError(f"[{domain}] No pude derivar answer. tool_result no trae dato (trace={path})")

    trace["respond_message"] = answer  # <-- para debug futuro

    if verbose:
        print(f"[{domain}] respond_message => {answer!r}")

    out = _post(base_url, "/step", {"episode_id": eid, "action": {"mode": "respond", "message": answer}}, timeout_s)
    trace["respond_raw"] = out

    done = bool(out.get("done"))
    reward = out.get("reward")

    if verbose:
        print(f"[{domain}] respond OK (done={done}, reward={reward})")

    if not done or not _reward_is_one(reward):
        path = _write_trace(trace_dir, domain, trace)
        raise AssertionError(f"[{domain}] expected done=True & reward≈1, got done={done}, reward={reward} (trace={path})")

    print(f"[{domain}] ✅ PASS")
    _write_trace(trace_dir, domain, trace)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace-dir", default="artifacts/smoke_traces")
    p.add_argument("--domains", nargs="*", default=DOMAINS)
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.verbose:
        print("[health]", _get(base_url, "/health", args.timeout_s))
        print("[schema keys]", list(_get(base_url, "/schema", args.timeout_s).keys()))

    fails: List[str] = []
    for d in args.domains:
        try:
            smoke_one(base_url, d, args.timeout_s, args.verbose, args.trace_dir)
        except Exception as e:
            fails.append(f"{d}: {e}")

    if fails:
        print("\n❌ Smoke test failures:")
        for f in fails:
            print("-", f)
        return 1

    print("\n✅ All domains passed (success).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

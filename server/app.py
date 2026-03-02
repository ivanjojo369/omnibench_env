"""
FastAPI server (stateful HTTP) for OmniBench / OpenEnv.

Endpoints:
- GET  /health  -> health check
- GET  /schema  -> JSON schemas (action/observation/state)
- POST /reset   -> crea/reusa episode_id, hace reset, Set-Cookie episode_id
- POST /step    -> ejecuta step usando episode_id (query/cookie/body/metadata)
- GET  /state   -> devuelve estado del episodio
- GET  /        -> redirect a /docs (para HF Spaces)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional
from uuid import uuid4

import uvicorn
from fastapi import Body, Cookie, FastAPI, HTTPException, Query, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, model_validator

from omnibench_env.models import OmnibenchAction, OmnibenchObservation, OmnibenchState
from .omnibench_env_environment import OmnibenchEnvironment

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

app = FastAPI(
    title="omnibench_env",
    version="0.1.1",
    description="Stateful HTTP environment server for OmniBench (OpenEnv).",
)

_env = OmnibenchEnvironment()
_lock = threading.Lock()


@app.get("/", include_in_schema=False)
def root():
    # HF Spaces “App” view: mejor mandar a /docs
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    domain_id: Optional[str] = None
    episode_id: Optional[str] = None

    # Acepta campos extra (p.ej. {"domain": "..."}) sin exponerlos como canonical.
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_domain(cls, data: Any):
        if isinstance(data, dict):
            if not data.get("domain_id") and data.get("domain"):
                data = {**data, "domain_id": data.get("domain")}
        return data


class StepRequest(BaseModel):
    # Canonical: {"episode_id": "...", "action": {...}}
    action: OmnibenchAction
    episode_id: Optional[str] = None

    # Permite legacy fields sin hacerlos parte del schema público
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, data: Any):
        """
        Acepta payload legacy sin documentarlo en OpenAPI.
        Si viene sin "action", lo envuelve como action={mode, tool_name, tool_args, message, metadata}.
        """
        if isinstance(data, dict) and "action" not in data:
            data = {
                "episode_id": data.get("episode_id"),
                "action": {
                    "mode": data.get("mode"),
                    "tool_name": data.get("tool_name"),
                    "tool_args": data.get("tool_args") or {},
                    "message": data.get("message"),
                    "metadata": data.get("metadata") or {},
                },
                # conserva otros extras, sin duplicar los legacy keys ya “absorbidos”
                **{
                    k: v
                    for k, v in data.items()
                    if k not in {"mode", "tool_name", "tool_args", "message", "metadata"}
                },
            }
        return data


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _split_obs(obs: OmnibenchObservation) -> tuple[dict[str, Any], float, bool]:
    d = obs.model_dump()
    done = bool(d.pop("done", False))
    reward_raw = d.pop("reward", 0.0)
    try:
        reward = float(0.0 if reward_raw is None else reward_raw)
    except Exception:
        reward = 0.0
    return d, reward, done


def _resolve_episode_id(
    body: StepRequest | ResetRequest,
    episode_id_q: str | None,
    episode_id_cookie: str | None,
) -> str | None:
    # 1) query param
    if episode_id_q and episode_id_q.strip():
        return episode_id_q.strip()

    # 2) body.episode_id
    v = getattr(body, "episode_id", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    # 3) StepRequest: permitir episode_id dentro de action.metadata (robustez)
    if isinstance(body, StepRequest) and body.action and isinstance(body.action.metadata, dict):
        for k in ("episode_id", "session_id"):
            vv = body.action.metadata.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()

    # 4) extra fields: si alguien manda metadata plano (legacy)
    extra = getattr(body, "__pydantic_extra__", None) or {}
    meta = extra.get("metadata")
    if isinstance(meta, dict):
        for k in ("episode_id", "session_id"):
            vv = meta.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()

    # 5) cookie
    if episode_id_cookie and episode_id_cookie.strip():
        return episode_id_cookie.strip()

    return None


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "omnibench_env"}


@app.get("/schema")
def schema():
    return {
        "action": OmnibenchAction.model_json_schema(),
        "observation": OmnibenchObservation.model_json_schema(),
        "state": OmnibenchState.model_json_schema(),
    }


@app.post("/reset")
def reset(
    response: Response,
    body: ResetRequest = Body(default_factory=ResetRequest),
    episode_id: str | None = Query(None),
):
    eid = str(episode_id or body.episode_id or uuid4())
    seed = body.seed
    domain = body.domain_id

    with _lock:
        obs = _env.reset(seed=seed, episode_id=eid, domain=domain)

    obs_dict, reward, done = _split_obs(obs)
    response.set_cookie(key="episode_id", value=eid, path="/", samesite="lax")
    return {"episode_id": eid, "observation": obs_dict, "reward": reward, "done": done}


@app.post("/step")
def step(
    response: Response,
    body: StepRequest = Body(...),
    episode_id: str | None = Query(None),
    episode_id_cookie: str | None = Cookie(None, alias="episode_id"),
):
    eid = _resolve_episode_id(body, episode_id, episode_id_cookie)
    if not eid:
        raise HTTPException(400, "Missing episode_id. Call /reset first (cookie) or pass ?episode_id=...")

    action = body.action

    # Amarrar eid también dentro de metadata (robustez)
    action.metadata = action.metadata or {}
    action.metadata["episode_id"] = eid

    with _lock:
        try:
            obs = _env.step(action, episode_id=eid)
        except Exception as e:
            raise HTTPException(400, str(e))

        if getattr(obs, "done", False):
            _env.close_session(eid)

    obs_dict, reward, done = _split_obs(obs)
    response.set_cookie(key="episode_id", value=eid, path="/", samesite="lax")
    return {"episode_id": eid, "observation": obs_dict, "reward": reward, "done": done}


@app.get("/state")
def state(
    episode_id: str | None = Query(None),
    episode_id_cookie: str | None = Cookie(None, alias="episode_id"),
):
    eid = episode_id or episode_id_cookie
    if not eid:
        return {"episode_id": None, "step_count": 0, "domain": None, "task_id": None}

    with _lock:
        st = _env.get_state(str(eid))

    return st.model_dump()


# ---------------------------------------------------------------------
# Entry point (openenv validate / uv run / HF Spaces)
# ---------------------------------------------------------------------

def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
    
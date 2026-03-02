# server/omnibench_env_environment.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
import json
import random
import threading

from openenv.core.env_server.interfaces import Environment

from omnibench_env.models import OmnibenchAction, OmnibenchObservation, OmnibenchState
from omnibench_env.domains.base import TaskSpec
from omnibench_env.domains.registry import DOMAINS, DOMAIN_IDS


@dataclass
class _Session:
    rng: random.Random
    state: OmnibenchState
    domain_id: str
    task: TaskSpec
    last_tool_result: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class OmnibenchEnvironment(Environment):
    """
    Router Environment (HTTP-safe):
    - Sesiones globales por proceso para que /reset y /step funcionen aunque el server recree instancias.
    - episode_id puede venir en:
        * kwargs: episode_id/session_id
        * action.metadata: {"episode_id": "..."}
        * o cae al último episodio global (smoke simple)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    _LOCK = threading.RLock()
    _SESSIONS: dict[str, _Session] = {}
    _GLOBAL_ACTIVE_EID: Optional[str] = None
    _MAX_SESSIONS: int = 512  # guardia básica contra leak

    def __init__(self) -> None:
        self._active_episode_id: Optional[str] = None

    @staticmethod
    def _flatten_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Soporta payloads tipo {"kwargs": {...}}."""
        merged = dict(kwargs or {})
        inner = merged.get("kwargs")
        if isinstance(inner, dict):
            merged.pop("kwargs", None)
            merged.update(inner)
        return merged

    @classmethod
    def _set_global_active(cls, eid: str) -> None:
        cls._GLOBAL_ACTIVE_EID = eid

    @classmethod
    def _resolve_episode_id(
        cls,
        action: Optional[OmnibenchAction],
        instance_active: Optional[str],
        **kwargs: Any,
    ) -> Optional[str]:
        # 1) kwargs del server
        for k in ("episode_id", "session_id"):
            v = kwargs.get(k)
            if v:
                return str(v)

        # 2) metadata del action
        if action is not None and isinstance(getattr(action, "metadata", None), dict):
            md = action.metadata or {}
            for k in ("episode_id", "session_id"):
                v = md.get(k)
                if v:
                    return str(v)

        # 3) activo de instancia
        if instance_active:
            return str(instance_active)

        # 4) activo global
        if cls._GLOBAL_ACTIVE_EID:
            return str(cls._GLOBAL_ACTIVE_EID)

        return None

    @classmethod
    def _get_session(cls, eid: str) -> _Session:
        s = cls._SESSIONS.get(eid)
        if s is None:
            raise RuntimeError(f"Unknown episode_id='{eid}'. Call /reset first or pass the right episode_id.")
        return s

    @classmethod
    def _gc_sessions(cls) -> None:
        """Mantén el store bajo control (FIFO simple)."""
        if len(cls._SESSIONS) <= cls._MAX_SESSIONS:
            return
        active = cls._GLOBAL_ACTIVE_EID
        for k in list(cls._SESSIONS.keys()):
            if len(cls._SESSIONS) <= cls._MAX_SESSIONS:
                break
            if k == active:
                continue
            cls._SESSIONS.pop(k, None)

    # --------- API extra que tu app.py ya usa ---------

    def get_state(self, episode_id: str) -> OmnibenchState:
        """Para /state en app.py."""
        with self._LOCK:
            sess = self._SESSIONS.get(str(episode_id))
            if sess is None:
                return OmnibenchState(episode_id=None, step_count=0, domain=None, task_id=None)
            return sess.state

    def close_session(self, episode_id: str) -> None:
        """Para app.py cuando done=True."""
        self._close_session(episode_id)

    def _close_session(self, episode_id: str | None = None, **kwargs: Any) -> None:
        """
        Compat: algunos servers llaman _close_session(eid).
        Debe ser idempotente.
        """
        eid = episode_id or kwargs.get("episode_id") or kwargs.get("session_id")
        if not eid:
            return
        eid = str(eid)

        with self._LOCK:
            self._SESSIONS.pop(eid, None)
            if self._active_episode_id == eid:
                self._active_episode_id = None
            if self._GLOBAL_ACTIVE_EID == eid:
                self._GLOBAL_ACTIVE_EID = None

    # --------- OpenEnv core ---------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> OmnibenchObservation:
        kw = self._flatten_kwargs(kwargs)

        seed_int = int(seed if seed is not None else 0)
        eid = str(episode_id or uuid4())
        rng = random.Random(seed_int)

        forced_domain = kw.get("domain_id", kw.get("domain", None))
        if forced_domain is not None:
            domain_id = str(forced_domain)
            if domain_id not in DOMAINS:
                raise ValueError(f"Unknown domain_id='{domain_id}'. Expected one of: {list(DOMAINS.keys())}")
        else:
            domain_id = rng.choice(DOMAIN_IDS)

        task_seed = rng.randint(0, 2**31 - 1)
        domain = DOMAINS[domain_id]
        task = domain.sample_task(task_seed)

        state = OmnibenchState(
            episode_id=eid,
            step_count=0,
            domain=domain_id,
            task_id=task.id,
        )

        with self._LOCK:
            self._SESSIONS[eid] = _Session(
                rng=rng,
                state=state,
                domain_id=domain_id,
                task=task,
                last_tool_result=None,
            )
            self._active_episode_id = eid
            self._set_global_active(eid)
            self._gc_sessions()

        return self._make_obs(eid, done=False, reward=0.0, metadata_extra={"task_seed": task_seed})

    def step(
        self,
        action: OmnibenchAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> OmnibenchObservation:
        kw = self._flatten_kwargs(kwargs)

        with self._LOCK:
            eid = self._resolve_episode_id(action, self._active_episode_id, **kw)
            if eid is None:
                raise RuntimeError("No active episode. Call /reset first (or pass action.metadata.episode_id).")

            sess = self._get_session(str(eid))
            self._active_episode_id = str(eid)
            self._set_global_active(str(eid))

            sess.state.step_count += 1
            mode = (action.mode or "").strip().lower()

            if mode == "tool":
                tool_name = (action.tool_name or "").strip()
                tool_args = action.tool_args or {}

                if not tool_name:
                    result = {"error": "missing_tool_name"}
                else:
                    try:
                        result = DOMAINS[sess.domain_id].call_tool(tool_name, tool_args, sess.task)
                    except Exception as e:
                        result = {"error": "tool_exception", "tool_name": tool_name, "detail": str(e)}

                sess.last_tool_result = result
                sess.tool_trace.append({"tool_name": tool_name, "tool_args": tool_args, "result": result})
                return self._make_obs(str(eid), done=False, reward=0.0, metadata_extra={"last_mode": "tool"})

            # respond (default) - robusto
            msg = (action.message or "").strip()
            try:
                check = DOMAINS[sess.domain_id].check_final(msg, sess.task)
            except Exception as e:
                check = type("Check", (), {})()
                check.reward = 0.0
                check.done = True
                check.info = {"error": "check_exception", "detail": str(e)}

            raw_info = getattr(check, "info", {}) or {}
            try:
                info = dict(raw_info)
            except Exception:
                info = {"info": str(raw_info)}

            # asegurar serializable
            try:
                json.dumps(info)
            except TypeError:
                info = json.loads(json.dumps(info, default=str))

            raw_reward = getattr(check, "reward", 0.0)
            try:
                reward = float(0.0 if raw_reward is None else raw_reward)
            except Exception:
                reward = 0.0

            done = bool(getattr(check, "done", True))

            sess.last_tool_result = info
            sess.tool_trace.append({"tool_name": "__final__", "tool_args": {"message": msg}, "result": info})

            return self._make_obs(str(eid), done=done, reward=reward, metadata_extra={"last_mode": "respond"})

    @property
    def state(self) -> OmnibenchState:
        with self._LOCK:
            eid = self._active_episode_id or self._GLOBAL_ACTIVE_EID
            if eid and eid in self._SESSIONS:
                return self._SESSIONS[eid].state
            return OmnibenchState(episode_id=None, step_count=0, domain=None, task_id=None)

    def _make_obs(
        self,
        eid: str,
        done: bool,
        reward: float,
        metadata_extra: dict[str, Any] | None = None,
    ) -> OmnibenchObservation:
        sess = self._get_session(eid)
        domain = DOMAINS[sess.domain_id]
        tools = domain.tools_for_task(sess.task)

        metadata = {
            "episode_id": eid,
            "step_count": sess.state.step_count,
            "domain_id": sess.domain_id,
            "task_id": sess.task.id,
            "tool_trace_len": len(sess.tool_trace),
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        return OmnibenchObservation(
            domain=sess.domain_id,
            task_id=sess.task.id,
            instruction=sess.task.instruction,
            available_tools=tools,
            last_tool_result=sess.last_tool_result,
            done=done,
            reward=reward,
            metadata=metadata,
        )
    
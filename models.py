from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from openenv.core.env_server.types import Action, Observation, State


class AgentDomain(str, Enum):
    agent_safety = "agent_safety"
    coding = "coding"
    healthcare = "healthcare"
    web = "web"
    computer_use = "computer_use"
    research = "research"
    finance = "finance"


class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: dict = Field(default_factory=dict)


class OmnibenchAction(Action):
    """
    One action schema for all domains:
    - mode="tool": call a tool with tool_name + tool_args
    - mode="respond": final answer in message
    """
    mode: Literal["tool", "respond"] = Field(..., description="Tool call or final response")
    message: Optional[str] = Field(default=None, description="Text response when mode='respond'")
    tool_name: Optional[str] = Field(default=None, description="Tool name when mode='tool'")
    tool_args: dict = Field(default_factory=dict, description="Tool arguments when mode='tool'")
    metadata: dict = Field(default_factory=dict, description="Extra metadata for the action")

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.mode == "respond" and (self.message is None or self.message.strip() == ""):
            raise ValueError("message is required when mode='respond'")
        if self.mode == "tool" and (self.tool_name is None or self.tool_name.strip() == ""):
            raise ValueError("tool_name is required when mode='tool'")
        return self


class OmnibenchState(State):
    domain: Optional[AgentDomain] = None
    task_id: Optional[str] = None


class OmnibenchObservation(Observation):
    domain: AgentDomain = Field(..., description="Current episode domain")
    task_id: str = Field(..., description="Task identifier")
    instruction: str = Field(..., description="What the agent must do")
    available_tools: list[ToolSpec] = Field(default_factory=list, description="Tools available for this domain")
    last_tool_result: Optional[dict] = Field(default=None, description="Result of the last tool call")

    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward signal")
    metadata: dict = Field(default_factory=dict, description="Extra debug metadata")


class ResetRequest(BaseModel):
    """
    Canonical: {"domain_id": "...", "seed": 123, "episode_id": "..."}
    Back-compat: also accepts {"domain": "..."} but we do NOT expose 'domain' in docs.
    """
    seed: Optional[int] = None
    domain_id: Optional[str] = None
    episode_id: Optional[str] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_domain(cls, data: Any):
        if isinstance(data, dict):
            if not data.get("domain_id") and data.get("domain"):
                data = {**data, "domain_id": data.get("domain")}
        return data


class StepRequest(BaseModel):
    """
    Canonical: {"episode_id": "...", "action": {...}}
    Back-compat: also accepts flat fields at root:
      {"episode_id": "...", "mode": "...", "tool_name": "...", ...}
    but we do NOT expose them in docs.
    """
    episode_id: Optional[str] = None
    action: OmnibenchAction

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, data: Any):
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
                **{k: v for k, v in data.items() if k not in {"mode", "tool_name", "tool_args", "message", "metadata"}},
            }
        return data
    
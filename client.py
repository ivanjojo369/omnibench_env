# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Omnibench Env Environment Client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from omnibench_env.models import OmnibenchAction, OmnibenchObservation, OmnibenchState

class OmnibenchEnv(EnvClient[OmnibenchAction, OmnibenchObservation, State]):
    """
    Client for the Omnibench Env Environment.

    Notes:
      - This client is used for Python-side interaction with a running environment server.
      - The important part for your current issue: EnvClient expects 3 generic parameters
        (Action, Observation, State). Missing the third one causes a runtime TypeError.
    """

    def _step_payload(self, action: OmnibenchAction) -> Dict[str, Any]:
        """
        Convert OmnibenchAction to JSON payload for step message.
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OmnibenchObservation]:
        """
        Parse server response into StepResult[OmnibenchObservation].
        """
        obs_data = payload.get("observation", {}) or {}

        observation = OmnibenchObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0) or 0),
        )
    
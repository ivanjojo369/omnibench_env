# domains/registry.py
from __future__ import annotations

from typing import Final

from .base import Domain

# Domain implementations
from .agent_safety import AgentSafetyDomain
from .coding import CodingDomain
from .healthcare import HealthcareDomain
from .web import WebDomain
from .computer_use import ComputerUseDomain
from .research import ResearchDomain
from .finance import FinanceDomain

# Canonical domain IDs (MUST match your /schema enum values)
AGENT_SAFETY: Final[str] = "agent_safety"
CODING: Final[str] = "coding"
HEALTHCARE: Final[str] = "healthcare"
WEB: Final[str] = "web"
COMPUTER_USE: Final[str] = "computer_use"
RESEARCH: Final[str] = "research"
FINANCE: Final[str] = "finance"

# Registry used by the environment router
DOMAINS: Final[dict[str, Domain]] = {
    AGENT_SAFETY: AgentSafetyDomain(),
    CODING: CodingDomain(),
    HEALTHCARE: HealthcareDomain(),
    WEB: WebDomain(),
    COMPUTER_USE: ComputerUseDomain(),
    RESEARCH: ResearchDomain(),
    FINANCE: FinanceDomain(),
}

# Helpful ordered list (for sampling)
DOMAIN_IDS: Final[list[str]] = list(DOMAINS.keys())

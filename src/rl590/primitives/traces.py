from enum import StrEnum


class TraceType(StrEnum):
    """Eligibility-trace update modes (TD-backward, SARSA-backward)."""

    ACCUMULATING = "accumulating"
    REPLACING = "replacing"


class MCMethod(StrEnum):
    """Monte Carlo control return-counting modes."""

    FIRST_VISIT = "first_visit"
    EVERY_VISIT = "every_visit"
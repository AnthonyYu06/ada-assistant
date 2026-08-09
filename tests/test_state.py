"""Unit tests for AssistantState mute / status-guard behavior.

The status doubles as mic-queue ownership, so set_muted() must never stomp an
in-flight turn's LISTENING/THINKING/SPEAKING/ERROR status, and set_status_if()
must only swap when the current status matches the expectation.
"""

from __future__ import annotations

import pytest

from ada.state import AssistantState, Status


@pytest.mark.parametrize(
    "busy_status",
    [Status.LISTENING, Status.THINKING, Status.SPEAKING, Status.ERROR],
)
def test_set_muted_does_not_stomp_active_status(busy_status: Status) -> None:
    state = AssistantState()
    state.set_status(busy_status)

    state.set_muted(True)
    # The mute flag flips, but the active status is preserved.
    assert state.muted is True
    assert state.status is busy_status

    state.set_muted(False)
    assert state.muted is False
    assert state.status is busy_status


def test_mute_from_idle_goes_to_muted() -> None:
    state = AssistantState()
    state.set_status(Status.IDLE)

    state.set_muted(True)

    assert state.muted is True
    assert state.status is Status.MUTED


def test_unmute_from_muted_goes_to_idle() -> None:
    state = AssistantState()
    state.set_status(Status.IDLE)
    state.set_muted(True)
    assert state.status is Status.MUTED

    state.set_muted(False)

    assert state.muted is False
    assert state.status is Status.IDLE


def test_set_muted_fires_listeners_only_on_status_change() -> None:
    state = AssistantState()
    state.set_status(Status.SPEAKING)
    seen: list[Status] = []
    state.add_listener(lambda status, detail: seen.append(status))

    # No status change while SPEAKING -> no listener call.
    state.set_muted(True)
    assert seen == []

    # A resting status does change -> listeners fire.
    state.set_status(Status.IDLE)
    seen.clear()
    state.set_muted(True)
    assert seen == [Status.MUTED]


def test_set_status_if_swaps_only_on_match() -> None:
    state = AssistantState()  # starts STARTING
    assert state.status is Status.STARTING

    # Wrong expected status: no change.
    assert state.set_status_if(Status.IDLE, Status.SPEAKING) is False
    assert state.status is Status.STARTING

    # Correct expected status: swaps and reports True.
    assert state.set_status_if(Status.STARTING, Status.IDLE) is True
    assert state.status is Status.IDLE

    # Expected no longer matches after the swap.
    assert state.set_status_if(Status.STARTING, Status.MUTED) is False
    assert state.status is Status.IDLE


def test_set_status_if_fires_listeners_on_change() -> None:
    state = AssistantState()
    seen: list[tuple[Status, str]] = []
    state.add_listener(lambda status, detail: seen.append((status, detail)))

    assert state.set_status_if(Status.STARTING, Status.IDLE, "ready") is True
    assert seen == [(Status.IDLE, "ready")]

    seen.clear()
    assert state.set_status_if(Status.STARTING, Status.MUTED) is False
    assert seen == []


def test_set_status_if_noop_while_shutting_down() -> None:
    state = AssistantState()
    state.request_shutdown()  # sets STOPPED and the shutdown event

    assert state.set_status_if(Status.STOPPED, Status.IDLE) is False
    assert state.status is Status.STOPPED

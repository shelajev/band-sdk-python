"""The room's shared turn owner covers asynchronous Claude preparation too."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from band.adapters.claude_sdk import ClaudeSDKAdapterConfig
from band.client.streaming import ControlMode
from band.core.exceptions import CleanupIncompleteError, InputNotAccepted
from band.core.turn_lifecycle import TurnTask
from tests.adapters.claude_sdk.fakecli import Hold
from tests.adapters.claude_sdk.helpers import ClaudeRoom
from tests.conftest import eager_tasks, requires_eager_tasks

OpenRoom = Callable[..., Awaitable[ClaudeRoom]]


class TestStartupOwner:
    @pytest.mark.looptime
    async def test_the_provider_deadline_starts_after_preparation(
        self, claude_room: OpenRoom
    ) -> None:
        hold = Hold()

        async def resolver(workspace: str) -> Mapping[str, Any]:
            hold.reached.set()
            await hold.released.wait()
            return {}

        room = await claude_room(
            ClaudeSDKAdapterConfig(turn_timeout_s=2.0),
            resolve_mcp_servers=resolver,
        )
        room.claude.script([room.model_reply("Ready.")])
        sending = room.send_in_background("hello")
        try:
            await hold.reached.wait()
            await asyncio.wait({sending}, timeout=4.0)
            assert not sending.done()
            assert room.claude.prompts == []
        finally:
            hold.released.set()
            await asyncio.gather(sending, return_exceptions=True)
        await sending
        assert room.chat == ["Ready."]
        assert room.failures == []

    async def test_a_resolving_turn_refuses_overlap_without_rebinding_tools(
        self, claude_room: OpenRoom
    ) -> None:
        hold = Hold()
        calls: list[str] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            hold.reached.set()
            await hold.released.wait()
            return {}

        room = await claude_room(resolve_mcp_servers=resolver)
        accepted, busy = room.fresh_tools(), room.fresh_tools()
        room.claude.script([room.model_reply("First.")])
        sending = asyncio.create_task(room.send("first", tools=accepted))
        try:
            await hold.reached.wait()
            with pytest.raises(InputNotAccepted):
                await room.send("overlap", tools=busy)
            assert not busy.turn.complete
            assert room.claude.prompts == []
        finally:
            hold.released.set()
            await sending

        assert accepted.turn.replied
        assert not busy.turn.replied
        assert len(calls) == 1
        assert room.chat == ["First."]
        room.claude.script([room.model_reply("Second.")])
        await room.send("again", tools=busy)
        assert busy.turn.replied
        assert len(calls) == 1
        assert len(room.claude.sessions) == 1

    @pytest.mark.parametrize("phase", ["resolution", "readiness"])
    async def test_stop_retains_startup_custody_until_settlement(
        self, claude_room: OpenRoom, phase: str
    ) -> None:
        hold, closing = Hold(), Hold()
        cancelled = asyncio.Event()
        cancellations: list[str] = []
        calls: list[str] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            if phase == "resolution" and len(calls) == 1:
                hold.reached.set()
                try:
                    await hold.released.wait()
                except asyncio.CancelledError:
                    cancellations.append(workspace)
                    cancelled.set()
                    await hold.released.wait()
            return {}

        room = await claude_room(resolve_mcp_servers=resolver)
        if phase == "readiness":
            room.claude.status_hold = hold
            room.claude.closing = closing
        sending = room.send_in_background("start")
        stopping: asyncio.Task[None] | None = None
        config_file: Path | None = None
        try:
            await hold.reached.wait()
            if phase == "readiness":
                config_file = Path(str(room.claude.sessions[0].options.mcp_servers))
                assert config_file.exists()
            stopping = asyncio.create_task(
                room.adapter.on_interrupt(room.room_id, ControlMode.STOP)
            )
            if phase == "resolution":
                await cancelled.wait()
            else:
                await closing.reached.wait()
            await asyncio.gather(
                room.adapter.on_interrupt(room.room_id, ControlMode.STOP),
                room.adapter.on_interrupt(room.room_id, ControlMode.STOP),
            )
            with pytest.raises(CleanupIncompleteError):
                await room.leave()
            with pytest.raises(InputNotAccepted):
                await room.send("overlap")
            manager = room.adapter._session_manager
            assert manager is not None and manager.owns_resources()
            assert room.claude.prompts == []
            if phase == "resolution":
                assert len(cancellations) == 1
                assert room.claude.sessions == []
            else:
                assert room.claude.sessions[0].alive
                assert config_file is not None and config_file.exists()
        finally:
            hold.released.set()
            closing.released.set()
            room.claude.status_hold = None
            room.claude.closing = None
            if stopping is not None:
                await stopping
            await asyncio.gather(sending, return_exceptions=True)

        with pytest.raises(asyncio.CancelledError):
            await sending
        await room.settled()
        await room.leave()
        assert manager is not None and not manager.owns_resources()
        if config_file is not None:
            assert not config_file.exists()
            assert not room.claude.sessions[0].alive
        assert room.failures == []
        room.claude.script([room.model_reply("Fresh.")])
        await room.send("again")
        assert len(calls) == 2
        assert room.chat == ["Fresh."]

    async def test_a_held_room_retirement_does_not_block_another_room(
        self, claude_room: OpenRoom
    ) -> None:
        hold = Hold()
        cancelled = asyncio.Event()
        calls: list[str] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            if workspace.endswith("room-1") and not hold.released.is_set():
                hold.reached.set()
                try:
                    await hold.released.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await hold.released.wait()
            return {}

        room = await claude_room(resolve_mcp_servers=resolver)
        sending = room.send_in_background("start")
        leaving: asyncio.Task[None] | None = None
        try:
            await hold.reached.wait()
            leaving = asyncio.create_task(room.leave())
            await cancelled.wait()
            other = room.beside("room-2")
            room.claude.script([other.model_reply("Independent.")])
            await other.send("hello")
            with pytest.raises(InputNotAccepted):
                await room.send("overlap")
            assert room.chat == []
            assert other.chat == ["Independent."]
        finally:
            hold.released.set()
            await asyncio.gather(sending, return_exceptions=True)
            if leaving is not None:
                try:
                    await leaving
                except CleanupIncompleteError:
                    pass

        with pytest.raises(asyncio.CancelledError):
            await sending
        await room.settled()
        await room.leave()
        room.claude.script([room.model_reply("Rejoined.")])
        await room.send("again")
        assert len(calls) == 3
        assert room.chat == ["Rejoined."]
        assert room.failures == []

    async def test_registration_failure_launches_nothing_and_leaves_no_worker(
        self, claude_room: OpenRoom, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        workers: list[asyncio.Task[None]] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            return {}

        def refuse_registration(owner: TurnTask) -> None:
            workers.append(owner.task)
            raise RuntimeError("turn registration refused")

        room = await claude_room(resolve_mcp_servers=resolver)
        refused = room.fresh_tools()
        monkeypatch.setattr(
            refused, "register_turn_task", refuse_registration, raising=False
        )
        with pytest.raises(RuntimeError, match="turn registration refused"):
            await room.send("refused", tools=refused)

        assert workers and all(worker.done() for worker in workers)
        assert not room.adapter._turn_running(room.room_id)
        assert room.claude.sessions == []
        assert calls == []
        assert room.chat == []
        assert room.failures == []
        room.claude.script([room.model_reply("Accepted.")])
        await room.send("again")
        assert len(calls) == 1
        assert room.chat == ["Accepted."]

    @requires_eager_tasks
    @pytest.mark.parametrize("cooperative", [False, True])
    async def test_a_scope_stopped_at_registration_never_starts_preparation(
        self,
        claude_room: OpenRoom,
        monkeypatch: pytest.MonkeyPatch,
        cooperative: bool,
    ) -> None:
        calls: list[str] = []
        workers: list[asyncio.Task[None]] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            return {}

        def stopped_registration(owner: TurnTask) -> None:
            workers.append(owner.task)
            owner.terminate(cooperative=cooperative)

        room = await claude_room(resolve_mcp_servers=resolver)
        stopped = room.fresh_tools()
        monkeypatch.setattr(
            stopped, "register_turn_task", stopped_registration, raising=False
        )
        with eager_tasks(), pytest.raises(asyncio.CancelledError):
            await room.send("stopped", tools=stopped)

        assert workers and all(worker.done() for worker in workers)
        assert not room.adapter._turn_running(room.room_id)
        assert room.claude.sessions == []
        assert calls == []
        assert room.failures == []
        room.claude.script([room.model_reply("Accepted.")])
        await room.send("again")
        assert room.chat == ["Accepted."]

    @requires_eager_tasks
    async def test_resolver_preparation_runs_under_the_eager_bound_owner(
        self, claude_room: OpenRoom
    ) -> None:
        calls: list[str] = []

        async def resolver(workspace: str) -> Mapping[str, Any]:
            calls.append(workspace)
            return {}

        room = await claude_room(resolve_mcp_servers=resolver)
        room.claude.script([room.model_reply("Ready.")])
        with eager_tasks():
            await room.send("hello")

        assert len(calls) == 1
        assert room.chat == ["Ready."]
        assert room.failures == []

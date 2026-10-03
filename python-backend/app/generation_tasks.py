from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class GenerationControl:
    task_id: str
    user_id: str
    conversation_id: str
    loop: asyncio.AbstractEventLoop
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task[Any] | None = None
    cancel_reason: str | None = None
    cancel_requested_at: str | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_reason is not None or self.cancel_event.is_set()

    def request_cancel(self, reason: str) -> bool:
        if self.cancelled:
            return False
        self.cancel_reason = reason
        self.cancel_requested_at = _now()

        def cancel_on_owner_loop() -> None:
            self.cancel_event.set()
            if self.worker and not self.worker.done():
                self.worker.cancel()

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self.loop:
            cancel_on_owner_loop()
        else:
            self.loop.call_soon_threadsafe(cancel_on_owner_loop)
        return True


class GenerationTaskRegistry:
    """Tracks live generation workers and enforces task ownership."""

    def __init__(self) -> None:
        self._controls: dict[str, GenerationControl] = {}
        self._lock = threading.RLock()

    def register(self, control: GenerationControl) -> None:
        with self._lock:
            if control.task_id in self._controls:
                raise RuntimeError(f"生成任务 {control.task_id} 已存在")
            self._controls[control.task_id] = control

    def bind(self, task_id: str, worker: asyncio.Task[Any]) -> None:
        with self._lock:
            control = self._controls.get(task_id)
            if not control:
                worker.cancel()
                return
            control.worker = worker
            if control.cancelled and not worker.done():
                control.loop.call_soon_threadsafe(worker.cancel)

    def cancel(self, task_id: str, user_id: str, reason: str = "user_requested") -> str:
        with self._lock:
            control = self._controls.get(task_id)
            if not control:
                return "not_found_or_finished"
            if control.user_id != user_id:
                return "forbidden"
            if not control.request_cancel(reason):
                return "already_cancelling"
            return "cancelling"

    def disconnect(self, task_id: str) -> None:
        with self._lock:
            control = self._controls.get(task_id)
            if control:
                control.request_cancel("client_disconnected")

    def unregister(self, task_id: str, control: GenerationControl) -> None:
        with self._lock:
            if self._controls.get(task_id) is control:
                self._controls.pop(task_id, None)

    def get(self, task_id: str) -> GenerationControl | None:
        with self._lock:
            return self._controls.get(task_id)

    async def cancel_all(self) -> None:
        with self._lock:
            controls = list(self._controls.values())
        workers: list[asyncio.Task[Any]] = []
        for control in controls:
            control.request_cancel("server_shutdown")
            if control.worker:
                workers.append(control.worker)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)


generation_tasks = GenerationTaskRegistry()

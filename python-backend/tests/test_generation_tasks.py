import asyncio
from contextlib import suppress
from pathlib import Path

import app.rag as rag_module
from app.database import Database
from app.generation_tasks import GenerationControl, GenerationTaskRegistry
from app.rag import ProductAssistantAgent


def test_registry_cancels_worker_and_is_idempotent() -> None:
    async def scenario() -> None:
        registry = GenerationTaskRegistry()
        control = GenerationControl("task-1", "user-1", "conversation-1", asyncio.get_running_loop())
        registry.register(control)
        cleaned = asyncio.Event()

        async def worker() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(worker())
        registry.bind(control.task_id, task)
        await asyncio.sleep(0)

        assert registry.cancel(control.task_id, "user-1") == "cancelling"
        with suppress(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert control.cancel_reason == "user_requested"
        assert registry.cancel(control.task_id, "user-1") == "already_cancelling"

        registry.unregister(control.task_id, control)
        assert registry.cancel(control.task_id, "user-1") == "not_found_or_finished"

    asyncio.run(scenario())


def test_registry_rejects_cross_user_cancellation() -> None:
    async def scenario() -> None:
        registry = GenerationTaskRegistry()
        control = GenerationControl("task-2", "owner", "conversation-2", asyncio.get_running_loop())
        registry.register(control)
        assert registry.cancel(control.task_id, "other-user") == "forbidden"
        assert control.cancelled is False
        registry.unregister(control.task_id, control)

    asyncio.run(scenario())


def test_agent_marks_trace_cancelled(tmp_path: Path, monkeypatch) -> None:
    database = Database(tmp_path / "cancel.db")
    database.initialize()
    monkeypatch.setattr(rag_module, "db", database)

    class Retriever:
        def rewrite(self, question, history):
            return question

        def search(self, query, intent_code=None, top_k=None):
            return []

    class BlockingModel:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def complete(self, messages, tools):
            self.started.set()
            await asyncio.Event().wait()

    model = BlockingModel()
    assistant = ProductAssistantAgent(retriever=Retriever(), tool_model=model)

    async def scenario() -> None:
        task = asyncio.create_task(assistant.run(
            "测试取消",
            "conversation-cancel",
            "task-cancel",
            {"id": "user-cancel", "username": "tester"},
            [],
        ))
        await asyncio.wait_for(model.started.wait(), timeout=1)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    with database.connect() as connection:
        trace = connection.execute("SELECT status, error_message, end_time FROM traces WHERE task_id='task-cancel'").fetchone()
    assert trace["status"] == "cancelled"
    assert trace["error_message"] == "生成任务已取消"
    assert trace["end_time"] is not None

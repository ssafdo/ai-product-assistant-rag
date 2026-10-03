import asyncio
import json
from pathlib import Path

import app.main as main_module
from fastapi import HTTPException
from app.database import Database
from app.generation_tasks import generation_tasks


def parse_sse(raw: str) -> tuple[str, dict]:
    lines = raw.strip().splitlines()
    event = next(line[7:] for line in lines if line.startswith("event: "))
    data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
    return event, data


def test_stop_endpoint_cancels_stream_and_persists_partial_answer(tmp_path: Path, monkeypatch) -> None:
    database = Database(tmp_path / "endpoint-cancel.db")
    database.initialize()
    monkeypatch.setattr(main_module, "db", database)

    class BlockingAgent:
        async def run(self, *args, on_token, cancelled, **kwargs):
            await on_token("已生成部分")
            await asyncio.Event().wait()

    monkeypatch.setattr(main_module, "agent", BlockingAgent())
    user = {"id": "cancel-owner", "username": "cancel-owner", "role": "user"}

    async def scenario() -> tuple[str, str, str, str]:
        response = await main_module.rag_chat("测试完整停止", None, False, user)
        stream = response.body_iterator
        meta_raw = await anext(stream)
        event, meta = parse_sse(meta_raw)
        assert event == "meta"

        message_raw = await anext(stream)
        event, message = parse_sse(message_raw)
        assert event == "message"
        assert message == {"type": "response", "delta": "已生成部分"}

        try:
            await main_module.stop_task(meta["taskId"], {**user, "id": "other-user"})
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("其他用户不应能停止该任务")
        stop_response = await main_module.stop_task(meta["taskId"], user)
        assert stop_response["data"]["status"] == "cancelling"
        repeated = await main_module.stop_task(meta["taskId"], user)
        assert repeated["data"]["status"] == "already_cancelling"

        cancel_raw = await anext(stream)
        event, cancel = parse_sse(cancel_raw)
        assert event == "cancel"
        assert cancel["reason"] == "user_requested"
        assert cancel["modelRoute"] == "cancelled"

        done_raw = await anext(stream)
        assert parse_sse(done_raw) == ("done", {})
        try:
            await anext(stream)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("取消事件后 SSE 应立即结束")
        assert generation_tasks.get(meta["taskId"]) is None
        finished = await main_module.stop_task(meta["taskId"], user)
        assert finished["data"]["status"] == "not_found_or_finished"
        return meta["conversationId"], meta["taskId"], cancel["messageId"], cancel["reason"]

    conversation_id, _, message_id, _ = asyncio.run(scenario())
    with database.connect() as connection:
        message = connection.execute(
            "SELECT content FROM messages WHERE id=? AND conversation_id=?",
            (int(message_id), conversation_id),
        ).fetchone()
    assert message["content"] == "已生成部分\n\n（已停止生成）"


def test_client_disconnect_cancels_and_unregisters_worker(tmp_path: Path, monkeypatch) -> None:
    database = Database(tmp_path / "disconnect.db")
    database.initialize()
    monkeypatch.setattr(main_module, "db", database)
    worker_cancelled = asyncio.Event()

    class BlockingAgent:
        async def run(self, *args, on_token, cancelled, **kwargs):
            try:
                await on_token("partial")
                await asyncio.Event().wait()
            finally:
                worker_cancelled.set()

    monkeypatch.setattr(main_module, "agent", BlockingAgent())
    user = {"id": "disconnect-owner", "username": "disconnect-owner", "role": "user"}

    async def scenario() -> None:
        response = await main_module.rag_chat("测试断连", None, False, user)
        stream = response.body_iterator
        _, meta = parse_sse(await anext(stream))
        assert parse_sse(await anext(stream))[0] == "message"
        await stream.aclose()
        await asyncio.wait_for(worker_cancelled.wait(), timeout=1)
        assert generation_tasks.get(meta["taskId"]) is None

    asyncio.run(scenario())

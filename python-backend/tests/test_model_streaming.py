import asyncio
from types import SimpleNamespace

import httpx

import app.rag as rag_module
from app.rag import ModelRouter


def test_model_router_forwards_upstream_sse_deltas_without_rechunking(monkeypatch) -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"真实"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" Token"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" 流"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(rag_module, "settings", SimpleNamespace(
        llm_api_key="key",
        llm_base_url="https://model.example/v1",
        llm_model="chat-model",
        secondary_api_key="",
        secondary_base_url="",
        secondary_model="",
        max_output_tokens=100,
    ))
    transport = httpx.MockTransport(handler)
    router = ModelRouter(lambda: httpx.AsyncClient(transport=transport))
    deltas: list[str] = []

    async def run() -> tuple[str, str]:
        return await router.stream_complete(
            "context",
            {"code": "general", "prompt": ""},
            [],
            lambda delta: _append(deltas, delta),
        )

    answer, route = asyncio.run(run())
    assert deltas == ["真实", " Token", " 流"]
    assert answer == "真实 Token 流"
    assert route == "primary:chat-model:stream"


async def _append(target: list[str], value: str) -> None:
    target.append(value)

import asyncio

from app.react_agent import ModelTurn, ReActEngine, ToolCall, ToolRegistry


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(
        "search",
        "检索知识",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        lambda query: [{"documentId": "doc-1", "content": f"evidence:{query}"}],
    )
    tools.register(
        "document",
        "读取文档",
        {
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "required": ["document_id"],
        },
        lambda document_id: {"id": document_id, "status": "ready"},
    )
    return tools


class TwoStepModel:
    async def complete(self, messages, tools):
        observations = [message for message in messages if message["role"] == "tool"]
        if not observations:
            return ModelTurn("", [ToolCall("call-1", "search", {"query": "price"})], "fake")
        if len(observations) == 1:
            return ModelTurn("", [ToolCall("call-2", "document", {"document_id": "doc-1"})], "fake")
        return ModelTurn("final answer", [], "fake")


class EndlessModel:
    async def complete(self, messages, tools):
        return ModelTurn("", [ToolCall("loop", "search", {"query": "again"})], "fake")


def test_tool_registry_exposes_openai_function_schema() -> None:
    schema = registry().schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["parameters"]["required"] == ["query"]


def test_react_loop_uses_multiple_observations_before_final_answer() -> None:
    events = []
    result = asyncio.run(ReActEngine(registry(), TwoStepModel(), max_steps=4).run(
        "system",
        "question",
        lambda step, phase, payload: events.append((step, phase)),
    ))
    assert result.answer == "final answer"
    assert [item["tool"] for item in result.observations] == ["search", "document"]
    assert result.steps == 3
    assert events == [
        (1, "plan"), (1, "action"), (1, "observation"),
        (2, "plan"), (2, "action"), (2, "observation"),
        (3, "plan"),
    ]


def test_react_loop_stops_at_max_steps() -> None:
    result = asyncio.run(ReActEngine(registry(), EndlessModel(), max_steps=2).run("system", "question"))
    assert result.stopped_by_limit is True
    assert result.steps == 2
    assert len(result.observations) == 2

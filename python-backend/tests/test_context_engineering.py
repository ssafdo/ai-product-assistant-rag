from app.context_engineering import ContextAssembler
from app.retrieval import SearchHit


def hit(chunk_id: str, content: str, score: float) -> SearchHit:
    return SearchHit(chunk_id, "doc", f"{chunk_id}.md", "产品库", content, score, "hybrid")


def test_context_selects_deduplicates_and_structures_evidence() -> None:
    assembler = ContextAssembler(token_budget=500)
    package = assembler.build(
        question="企业版如何计费？",
        rewritten_question="企业版价格和计费方式",
        intent={"code": "pricing", "name": "套餐价格"},
        history=[{"role": "user", "content": "先介绍企业版"}],
        hits=[
            hit("high", "企业版按坐席数量计费，支持年度合同。", 0.9),
            hit("duplicate", "企业版按坐席数量计费，支持年度合同。", 0.8),
            hit("other", "企业版支持私有化部署。", 0.7),
        ],
        observations=[{"tool": "get_product_document", "arguments": {"id": "doc"}, "result": {"status": "ready"}}],
    )

    assert [item.chunk_id for item in package.selected_hits] == ["high", "other"]
    assert package.stats["candidateEvidence"] == 3
    assert package.stats["deduplicatedEvidence"] == 2
    assert "## 检索证据" in package.prompt
    assert "## 工具 Observation" in package.prompt
    assert "## 最近对话" in package.prompt


def test_context_never_exceeds_token_budget() -> None:
    assembler = ContextAssembler(token_budget=220)
    package = assembler.build(
        question="请说明部署、价格和交付流程",
        rewritten_question="部署价格交付流程",
        intent={"code": "general", "name": "通用问答"},
        history=[{"role": "user", "content": "历史问题" * 200}],
        hits=[hit(str(index), f"第 {index} 段资料：" + "产品说明" * 300, 1 - index / 10) for index in range(4)],
        observations=[{"tool": "document", "arguments": {}, "result": {"text": "工具结果" * 200}}],
    )

    assert package.token_count <= 220
    assert package.stats["truncated"] is True
    assert package.selected_hits

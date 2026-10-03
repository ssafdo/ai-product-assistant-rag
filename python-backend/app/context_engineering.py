from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import tiktoken

from .retrieval import SearchHit, tokenize


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        suffix = "\n[内容已按 Token 预算截断]"
        suffix_tokens = self.encoding.encode(suffix)
        if max_tokens <= len(suffix_tokens):
            return self.encoding.decode(tokens[:max_tokens])
        return self.encoding.decode(tokens[: max_tokens - len(suffix_tokens)]) + suffix


@dataclass
class ContextPackage:
    prompt: str
    token_count: int
    selected_hits: list[SearchHit]
    stats: dict[str, Any]


class ContextAssembler:
    """Select, structure and truncate RAG context against an exact token budget."""

    def __init__(self, token_budget: int, counter: TokenCounter | None = None) -> None:
        self.token_budget = token_budget
        self.counter = counter or TokenCounter()

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def select_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        selected: list[SearchHit] = []
        normalized_seen: set[str] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            normalized = re.sub(r"\s+", "", hit.content).lower()
            if not normalized or normalized in normalized_seen:
                continue
            if any(self._similarity(hit.content, item.content) >= 0.82 for item in selected):
                continue
            normalized_seen.add(normalized)
            selected.append(hit)
        return selected

    def _fit_blocks(self, blocks: list[str], budget: int) -> tuple[list[str], int, bool]:
        fitted: list[str] = []
        used = 0
        truncated = False
        for block in blocks:
            block_tokens = self.counter.count(block)
            remaining = budget - used
            if remaining <= 0:
                truncated = True
                break
            if block_tokens <= remaining:
                fitted.append(block)
                used += block_tokens
                continue
            if remaining >= 32:
                fitted.append(self.counter.truncate(block, remaining))
                used = budget
            truncated = True
            break
        return fitted, used, truncated

    def build(
        self,
        question: str,
        rewritten_question: str,
        intent: dict[str, Any],
        history: list[dict[str, str]],
        hits: list[SearchHit],
        observations: list[dict[str, Any]],
    ) -> ContextPackage:
        header = (
            "## 回答约束\n"
            "只依据下列证据和工具结果回答；证据不足时明确说明。"
            "不得编造价格、合同、客户隐私或未发布路线图。引用知识时使用 [资料N]。\n\n"
            f"## 用户问题\n原始问题：{question}\n改写问题：{rewritten_question}\n\n"
            f"## 业务意图\n代码：{intent.get('code', 'general')}\n名称：{intent.get('name', '通用产品问答')}\n"
        )
        header = self.counter.truncate(header, max(1, self.token_budget // 3))
        remaining = max(0, self.token_budget - self.counter.count(header) - 30)

        candidates = self.select_hits(hits)
        evidence_budget = int(remaining * 0.68)
        observation_budget = int(remaining * 0.17)
        history_budget = remaining - evidence_budget - observation_budget

        evidence_blocks = [
            (
                f"[资料{index}]\n来源：{hit.document_name}\n知识库：{hit.knowledge_base}\n"
                f"相关度：{hit.score:.4f}\n内容：\n{hit.content.strip()}"
            )
            for index, hit in enumerate(candidates, start=1)
        ]
        fitted_evidence, evidence_tokens, evidence_truncated = self._fit_blocks(
            evidence_blocks, evidence_budget
        )
        selected_hits = candidates[: len(fitted_evidence)]

        tool_blocks = []
        for item in observations:
            if item.get("tool") == "search_product_knowledge":
                continue
            tool_blocks.append(
                f"工具：{item.get('tool')}\n参数：{json.dumps(item.get('arguments'), ensure_ascii=False)}\n"
                f"结果：{json.dumps(item.get('result'), ensure_ascii=False)}"
            )
        fitted_tools, tool_tokens, tools_truncated = self._fit_blocks(tool_blocks, observation_budget)

        history_blocks = [
            f"{item.get('role', 'unknown')}: {item.get('content', '').strip()}"
            for item in history[-8:]
            if item.get("content", "").strip()
        ]
        fitted_history_reversed, history_tokens, history_truncated = self._fit_blocks(
            list(reversed(history_blocks)), history_budget
        )
        fitted_history = list(reversed(fitted_history_reversed))

        sections = [header]
        sections.append("## 检索证据\n" + ("\n\n".join(fitted_evidence) or "未检索到可用证据"))
        sections.append("## 工具 Observation\n" + ("\n\n".join(fitted_tools) or "无额外工具结果"))
        sections.append("## 最近对话\n" + ("\n".join(fitted_history) or "无历史对话"))
        prompt = "\n\n".join(sections)
        if self.counter.count(prompt) > self.token_budget:
            prompt = self.counter.truncate(prompt, self.token_budget)

        return ContextPackage(
            prompt=prompt,
            token_count=self.counter.count(prompt),
            selected_hits=selected_hits,
            stats={
                "budget": self.token_budget,
                "candidateEvidence": len(hits),
                "deduplicatedEvidence": len(candidates),
                "selectedEvidence": len(selected_hits),
                "evidenceTokens": evidence_tokens,
                "observationTokens": tool_tokens,
                "historyTokens": history_tokens,
                "truncated": evidence_truncated or tools_truncated or history_truncated,
            },
        )

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.infra.chat;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.google.gson.Gson;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.enums.ModelProvider;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Local deterministic chat fallback for demos without external model credentials.
 */
@Service
public class NoopChatClient implements ChatClient {

    private static final int MAX_TITLE_LENGTH = 30;
    private static final int MAX_CONTEXT_CHARS = 1200;
    private static final int MAX_CONTEXT_LINES = 12;

    private final Gson gson = new Gson();

    @Override
    public String provider() {
        return ModelProvider.NOOP.getId();
    }

    @Override
    public String chat(ChatRequest request, ModelTarget target) {
        String joined = joinMessages(request);
        String lastUser = lastUserMessage(request);

        if (looksLikeRewritePrompt(joined)) {
            String question = StrUtil.blankToDefault(lastUser, joined).trim();
            return "{\"rewrite\":" + gson.toJson(question)
                    + ",\"should_split\":false,\"sub_questions\":["
                    + gson.toJson(question) + "]}";
        }
        if (looksLikeIntentPrompt(joined)) {
            return "[]";
        }
        if (looksLikeMcpParameterPrompt(joined)) {
            return "{}";
        }
        if (looksLikeTitlePrompt(joined)) {
            return shortText(extractTitleQuestion(lastUser), MAX_TITLE_LENGTH);
        }
        if (looksLikeSummaryPrompt(joined)) {
            return shortText(joined, 200);
        }

        return buildAnswer(request);
    }

    @Override
    public StreamCancellationHandle streamChat(ChatRequest request, StreamCallback callback, ModelTarget target) {
        try {
            callback.onContent(buildAnswer(request));
            callback.onComplete();
        } catch (Exception ex) {
            callback.onError(ex);
        }
        return StreamCancellationHandles.noop();
    }

    private String buildAnswer(ChatRequest request) {
        String evidence = extractEvidence(request);
        if (StrUtil.isBlank(evidence)) {
            return "\u672a\u68c0\u7d22\u5230\u4e0e\u95ee\u9898\u76f8\u5173\u7684\u6587\u6863\u5185\u5bb9\u3002";
        }

        List<String> lines = compactEvidenceLines(evidence);
        if (CollUtil.isEmpty(lines)) {
            return "\u672a\u68c0\u7d22\u5230\u4e0e\u95ee\u9898\u76f8\u5173\u7684\u6587\u6863\u5185\u5bb9\u3002";
        }

        StringBuilder answer = new StringBuilder();
        answer.append("\u6839\u636e\u77e5\u8bc6\u5e93\u5185\u5bb9\uff0c\u53ef\u4ee5\u8fd9\u6837\u56de\u7b54\uff1a\n\n");
        for (String line : lines) {
            if (answer.length() + line.length() > MAX_CONTEXT_CHARS) {
                break;
            }
            answer.append("- ").append(line).append('\n');
        }
        return answer.toString().trim();
    }

    private List<String> compactEvidenceLines(String evidence) {
        List<String> result = new ArrayList<>();
        for (String rawLine : evidence.split("\\R")) {
            String line = cleanEvidenceLine(rawLine);
            if (StrUtil.isBlank(line) || shouldSkipEvidenceLine(line)) {
                continue;
            }
            result.add(line);
            if (result.size() >= MAX_CONTEXT_LINES) {
                break;
            }
        }
        return result;
    }

    private String cleanEvidenceLine(String rawLine) {
        String line = StrUtil.blankToDefault(rawLine, "").trim();
        line = line.replaceFirst("^#{1,6}\\s*", "");
        line = line.replaceFirst("^[-*+]\\s+", "");
        line = line.replace("**", "");
        line = line.replace("`", "");
        return line.trim();
    }

    private boolean shouldSkipEvidenceLine(String line) {
        return line.equals("---")
                || line.matches("-{2,}")
                || line.equalsIgnoreCase("text")
                || line.equals("\u77e5\u8bc6\u5e93\u7247\u6bb5")
                || line.startsWith("## ")
                || line.contains("\u6587\u6863\u5185\u5bb9")
                || line.contains("\u76f8\u5173\u6587\u6863")
                || line.contains("\u5b50\u95ee\u9898")
                || line.contains("BEGIN")
                || line.contains("END");
    }

    private String extractEvidence(ChatRequest request) {
        if (request == null || CollUtil.isEmpty(request.getMessages())) {
            return "";
        }
        String lastUser = lastUserMessage(request);
        String best = "";
        for (ChatMessage message : request.getMessages()) {
            if (message == null || message.getRole() != ChatMessage.Role.USER) {
                continue;
            }
            String content = StrUtil.blankToDefault(message.getContent(), "");
            if (content.equals(lastUser)) {
                continue;
            }
            if (isLikelyEvidence(content) && content.length() > best.length()) {
                best = content;
            }
        }
        return best;
    }

    private boolean isLikelyEvidence(String content) {
        return content.contains("---")
                || content.contains("##")
                || content.contains("\u6587\u6863")
                || content.contains("\u76f8\u5173")
                || content.length() > 300;
    }

    private boolean looksLikeRewritePrompt(String text) {
        return text.contains("\"rewrite\"") && text.contains("sub_questions");
    }

    private boolean looksLikeIntentPrompt(String text) {
        return text.contains("\"score\"")
                && text.contains("\"reason\"")
                && text.contains("intent_list");
    }

    private boolean looksLikeMcpParameterPrompt(String text) {
        return text.contains("toolId")
                || text.contains("\u5de5\u5177ID")
                || text.contains("\u53c2\u6570");
    }

    private boolean looksLikeTitlePrompt(String text) {
        String lower = text.toLowerCase();
        return lower.contains("title") || text.contains("\u6807\u9898");
    }

    private boolean looksLikeSummaryPrompt(String text) {
        return text.contains("\u6458\u8981") || text.toLowerCase().contains("summary");
    }

    private String extractTitleQuestion(String prompt) {
        if (StrUtil.isBlank(prompt)) {
            return "";
        }
        String marker = "\u7528\u6237\u95ee\u9898";
        int markerIndex = prompt.lastIndexOf(marker);
        if (markerIndex >= 0) {
            String afterMarker = prompt.substring(markerIndex + marker.length()).trim();
            afterMarker = afterMarker.replaceFirst("^[#:\\uff1a\\s]+", "").trim();
            if (StrUtil.isNotBlank(afterMarker)) {
                return firstNonBlankLine(afterMarker);
            }
        }
        return firstNonBlankLine(prompt);
    }

    private String firstNonBlankLine(String value) {
        for (String line : StrUtil.blankToDefault(value, "").split("\\R")) {
            String normalized = line.trim();
            if (StrUtil.isNotBlank(normalized)) {
                return normalized;
            }
        }
        return value;
    }

    private String joinMessages(ChatRequest request) {
        if (request == null || CollUtil.isEmpty(request.getMessages())) {
            return "";
        }
        StringBuilder joined = new StringBuilder();
        for (ChatMessage message : request.getMessages()) {
            if (message != null && StrUtil.isNotBlank(message.getContent())) {
                joined.append(message.getContent()).append('\n');
            }
        }
        return joined.toString();
    }

    private String lastUserMessage(ChatRequest request) {
        if (request == null || CollUtil.isEmpty(request.getMessages())) {
            return "";
        }
        List<ChatMessage> messages = request.getMessages();
        for (int i = messages.size() - 1; i >= 0; i--) {
            ChatMessage message = messages.get(i);
            if (message != null
                    && message.getRole() == ChatMessage.Role.USER
                    && StrUtil.isNotBlank(message.getContent())) {
                return message.getContent().trim();
            }
        }
        return "";
    }

    private String shortText(String text, int maxLength) {
        String value = StrUtil.blankToDefault(text, "\u65b0\u5bf9\u8bdd")
                .replaceAll("\\s+", " ")
                .trim();
        if (value.length() <= maxLength) {
            return value;
        }
        return value.substring(0, Math.max(1, maxLength));
    }
}

# Python Agentic RAG 后端

这是原 Java 后端的 Python 版本，直接兼容项目现有 React 前端。业务数据使用 SQLite，向量由 Qdrant Local Mode 持久化；首次启动会自动导入 `resources/docs/knowledge/product` 中的产品资料。

## Agent + RAG 链路

每次问题按以下链路处理：

1. 问题改写：结合术语映射和最近会话解决指代问题。
2. 意图路由：将问题路由到产品介绍、价格、交付、售前或发布意图。
3. ReAct 规划：模型读取 `ToolRegistry` 生成的 Function Calling Schema，自主选择工具。
4. 混合检索：BM25 与 Embedding + Qdrant 双路召回，使用 RRF 融合和 Rerank 重排。
5. Observation：工具结果以 `role=tool` 回填，模型可继续选择工具，最多执行限定轮次。
6. 上下文工程：按相关度筛选和去重证据，结构化组装问题、意图、证据、工具结果与历史，并使用 `tiktoken` 执行预算截断。
7. 模型路由：优先调用 OpenAI 兼容模型，失败时切换备用模型，最后降级为本地 Planner 与有依据回答。
8. Token 流：最终模型请求使用 `stream=true`，每个上游 `delta.content` 立即转发为 SSE `message` 事件。
9. 生成中止：停止接口校验任务归属并取消实际 AsyncIO Worker，关闭规划或生成 HTTP 请求，保存部分回答并发送 `cancel/done`。
10. Trace：逐轮记录 Plan、Action、Observation、上下文 Token 用量和最终生成节点；取消任务落为 `cancelled` 而非长期停留在 `running`。

## 启动

```powershell
cd python-backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python run.py
```

另开一个终端启动前端：

```powershell
cd frontend
npm run dev
```

访问 `http://localhost:5173`，管理员账号为 `admin / admin`。API 文档位于 `http://localhost:9090/docs`。

如果使用 DeepSeek 等 OpenAI 兼容服务，在启动后端前设置：

```powershell
$env:LLM_API_KEY = "your-key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-chat"
```

不配置 Key 时系统仍可运行，会使用本地检索证据生成可核验的降级回答。

Embedding 与 Rerank 可分别通过 `EMBEDDING_*`、`RERANK_*` 环境变量接入服务；留空时使用本地确定性实现。完整字段见 `.env.example`。

`CONTEXT_TOKEN_BUDGET` 控制送入最终生成模型的上下文总预算，`MAX_OUTPUT_TOKENS` 控制回答上限。配置 LLM 后输出为真实模型 Token 流；未配置 LLM 时使用明确标记的本地降级流。

`POST /api/product-assistant/rag/v3/stop?taskId=...` 是幂等停止接口：活动任务首次返回 `cancelling`，重复调用返回 `already_cancelling`，任务完成或不存在返回 `not_found_or_finished`；其他用户的任务返回 403。


# Python Agentic RAG 后端

这是原 Java 后端的 Python 版本，直接兼容项目现有 React 前端。默认使用 SQLite 持久化，首次启动会自动导入 `resources/docs/knowledge/product` 中的产品资料，无需 PostgreSQL、Redis、RocketMQ 或向量数据库即可完成演示。

## Agent + RAG 链路

每次问题按以下链路处理：

1. 问题改写：结合术语映射和最近会话解决指代问题。
2. 意图路由：将问题路由到产品介绍、价格、交付、售前或发布意图。
3. 工具调用：Agent 通过 `ToolRegistry` 调用产品知识检索工具。
4. 混合检索：中文字符/双字词与英文词的加权召回，并按意图重排。
5. 上下文工程：采用 Gather、Select、Structure、Compress 流程组织证据和记忆。
6. 模型路由：优先调用 OpenAI 兼容模型，失败时切换备用模型，最后降级为本地有依据回答。
7. SSE 输出：兼容原前端的 `meta/message/finish/done` 事件。
8. Trace：记录改写、意图、工具、上下文和生成节点。

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


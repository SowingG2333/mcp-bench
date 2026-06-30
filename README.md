# MCP-Bench Agent 环境

基于 [MCP-Bench](https://github.com/accenture/mcp-bench) 构建的真实 MCP Agent 运行环境。它把原本的 benchmark 框架扩展为一个可直接使用的多工具 LLM Agent，提供 CLI、Gradio GUI 和 FastAPI 三种交互方式，并支持通过 OpenAI 兼容协议接入任意自定义模型。

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-Protocol-green)](https://github.com/anthropics/mcp)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 项目定位

MCP-Bench 原本是一个用于评测大模型工具使用能力的 benchmark。本仓库在其基础上做了扩展，让它成为一个可实际运行的 Agent 环境：

- 保留原有 **28 个 MCP Server**，覆盖地图、天气、营养、学术搜索、医疗计算、博物馆、加密货币等场景。
- 新增 `agent/chat_session.py` 作为统一会话核心，管理 LLM Provider、MCP 连接、多轮对话历史和会话持久化。
- 新增三种人机交互接口：
  - `chat_with_agent.py`：命令行对话
  - `gui_chat.py`：Gradio 网页界面，支持实时查看执行轨迹
  - `api_chat.py`：FastAPI REST 服务，供其他系统调用
- 支持任意 **OpenAI 兼容端点**，方便接入 vLLM、TGI、Ollama、LM Studio、DashScope、硅基流动等自定义模型。
- 每次对话自动保存到 `sessions/<session_id>.json`，方便后续复盘和分析。

> 底层的多轮规划和执行逻辑继承自 MCP-Bench 原生的 `TaskExecutor`。每轮规划时会把当前已连接的所有 tool 描述都输入给 LLM，这是 benchmark 本身的设计。

---

## 安装

### 1. 克隆仓库并创建环境

```bash
git clone https://github.com/yourusername/mcp-bench.git
cd mcp-bench

conda create -n mcpbench python=3.10
conda activate mcpbench
```

### 2. 安装 MCP Server 依赖

```bash
cd mcp_servers
bash ./install.sh
cd ..
```

### 3. 安装对话接口依赖

```bash
pip install gradio fastapi uvicorn
```

---

## 配置

### MCP Server API Key

部分 MCP Server 需要外部 API Key。编辑 `mcp_servers/api_key`：

```bash
NPS_API_KEY=your_key_here
NASA_API_KEY=your_key_here
HF_TOKEN=your_token_here
GOOGLE_MAPS_API_KEY=your_key_here
NCI_API_KEY=your_key_here   # 可选；保留为 YOUR_KEY_HERE 将跳过 BioMCP 的 NCI 相关工具
```

Key 获取地址：
- NPS：https://www.nps.gov/subjects/developer/get-started.htm
- NASA：https://api.nasa.gov/
- Hugging Face：https://huggingface.co/docs/hub/security-tokens
- Google Maps：https://developers.google.com/maps

### 模型配置

可以通过以下两种方式接入模型。

#### 方式一：注册 Provider（OpenRouter / Azure OpenAI）

在项目根目录创建 `.env`：

```dotenv
# OpenRouter
OPENROUTER_API_KEY=your_openrouter_key

# 或 Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://...
```

然后使用 `llm/factory.py` 中注册的模型名：

```bash
python chat_with_agent.py --model gpt-4o
```

#### 方式二：直接指定 OpenAI 兼容端点（推荐本地或自定义模型）

创建 `.env`：

```dotenv
MY_MODEL_BASE_URL=http://localhost:8000/v1
MY_MODEL_API_KEY=EMPTY
MY_MODEL_NAME=qwen2.5-7b-instruct
```

然后直接运行接口，无需 `--model`：

```bash
python chat_with_agent.py
```

这是接入 vLLM、TGI、Ollama、LM Studio、DashScope、硅基流动等最方便的方式。

---

## 三种使用接口

### 1. CLI 命令行：`chat_with_agent.py`

在终端进行交互式对话。

```bash
# 使用 llm/factory.py 中注册的模型
python chat_with_agent.py --model gpt-4o

# 使用 .env 中配置的直接端点
python chat_with_agent.py

# 只连接部分 Server
python chat_with_agent.py --servers "Wikipedia,FruityVice,Paper Search"

# 设置每轮超时（默认 300 秒）
python chat_with_agent.py --timeout 120
```

输入消息后回车发送，输入 `quit`、`exit` 或 `q` 退出。每次对话自动保存到 `sessions/<session_id>.json`。

---

### 2. GUI 网页界面：`gui_chat.py`

启动 Gradio 网页：

```bash
python gui_chat.py
# 打开 http://127.0.0.1:7860
```

可选参数：

```bash
python gui_chat.py --host 0.0.0.0 --port 7860 --share
```

使用流程：
1. 左侧选择已注册模型，或展开 **Direct Endpoint** 输入自定义端点。
2. 选择要连接的 MCP Server（不选则默认连接全部 28 个）。
3. 点击 **Connect**。
4. 在右侧输入框发送消息。

页面下方有 **🔍 Execution Trace** 面板，会实时展示 Agent 每轮调用了哪些工具、参数是什么、是否成功、结果摘要如何。

---

### 3. API 服务：`api_chat.py`

启动 FastAPI 服务：

```bash
python api_chat.py
# 接口文档 http://127.0.0.1:8000/docs
```

接口列表：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/models` | 列出已注册模型 |
| GET | `/servers` | 列出可用 MCP Server |
| POST | `/sessions` | 创建新会话 |
| POST | `/sessions/{id}/chat` | 发送消息并获取回复 |
| GET | `/sessions/{id}` | 获取会话历史和元数据 |
| DELETE | `/sessions/{id}` | 关闭会话并释放 MCP 连接 |

示例：

```bash
# 创建会话
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "servers": ["Wikipedia", "FruityVice"]
  }'

# 对话
curl -X POST http://127.0.0.1:8000/sessions/a1b2c3d4/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "设计一份低钠食谱",
    "timeout": 300
  }'

# 关闭会话
curl -X DELETE http://127.0.0.1:8000/sessions/a1b2c3d4
```

---

## 会话持久化

每次对话结束后会自动保存到 `sessions/<session_id>.json`，内容包括：

- 会话元数据：模型、Server、时间戳
- 完整的多轮对话历史
- 每轮统计：执行轮数、工具调用次数、token 消耗
- 每次工具调用的完整执行结果

读取已保存的会话：

```python
from agent.chat_session import ChatSession

state = ChatSession.load("a1b2c3d4")
print(state.turns)
```

---

## 项目结构

```text
mcp-bench/
├── agent/
│   ├── executor.py              # MCP-Bench 原生多轮执行器
│   ├── execution_context.py     # 重试 / 压缩状态管理
│   └── chat_session.py          # CLI/GUI/API 共享的会话核心
├── api_chat.py                  # FastAPI 服务
├── gui_chat.py                  # Gradio 网页界面
├── chat_with_agent.py           # 命令行对话
├── benchmark/                   # 原始 benchmark 框架
├── config/
├── llm/                         # LLM Provider 工厂
├── mcp_modules/                 # MCP 连接管理
├── mcp_servers/                 # 28 个 MCP Server 实现
├── utils/                       # 配置加载、Server 发现
└── run_benchmark.py             # 原始 benchmark 入口
```

---

## 使用建议

- **可根据场景调整 Server 数量。** 连接全部 28 个 Server 会把约 250 个 tool 描述塞进每轮规划 prompt，速度慢且 token 消耗大。如果设定具体场景，可以只保留相关 Server。
- **通过 GUI 的 Execution Trace 观察 Agent 行为。** 可以清楚看到每轮调用了哪些工具、参数和结果。
- **`TaskExecutor` 未做改动。** 它针对 benchmark 评测设计，而非日常效率。如果需要更快、更便宜的运行，建议只选相关 Server，或自行修改 `agent/executor.py` 增加 tool 预筛选。

---

## 原始 MCP-Bench

本项目是 Accenture MCP-Bench 的 fork。原始的 benchmark 框架仍可通过 `run_benchmark.py` 和 `benchmark/` 目录使用。如果你在研究中使用了原始 benchmark，请引用：

```bibtex
@article{wang2025mcpbench,
  title={MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers},
  author={Wang, Zhenting and Chang, Qi and Patel, Hemani and Biju, Shashank and Wu, Cheng-En and Liu, Quan and Ding, Aolin and Rezazadeh, Alireza and Shah, Ankit and Bao, Yujia and Siow, Eugene},
  journal={arXiv preprint arXiv:2508.20453},
  year={2025}
}
```

## License

Apache 2.0

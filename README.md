# Bot Agent

基于 Python 的多 Agent 协作系统，采用 ReAct (Reasoning + Acting) 架构，支持三级记忆管理、角色扮演、情感状态等能力。

---

## 示例

<details>
<summary><b>点击展开完整运行示例</b></summary>

```
------------------------------------------------------------

[你] > 今天天气不错，下午喝杯咖啡怎么样，有什么推荐的？

────────────────────────────────────────────────────────────
[处理中...]
────────────────────────────────────────────────────────────
[system_agent] ReAct 迭代 1/10
[system_agent] LLM 原始输出:
--- BEGIN ---
Thought: 我需要先调用 memory_agent 检索与用户问题相关的记忆
Action: call_agent
Action Input: {"agent_name": "memory_agent", "input": "今天天气不错，下午喝杯咖啡怎么样，有什么推荐的？"}
--- END ---

──────────────────────────────────────────────────
[System THOUGHT]
我需要先调用 memory_agent 检索与用户问题相关的记忆

──────────────────────────────────────────────────
[System ACTION] call_agent
   agent_name: memory_agent
   input: 今天天气不错，下午喝杯咖啡怎么样，有什么推荐的？

[MemoryAgent] invoke 开始
[memory_agent] ReAct 迭代 1/10

────────────────────────────────────────
[Memory THOUGHT]
我需要先检索是否已有关于用户咖啡偏好或相关记忆

────────────────────────────────────────
[Memory ACTION] search_memory
   query: 用户 咖啡 偏好 推荐 下午 喝

────────────────────────────────────────
[Memory OBSERVATION]
{
  "message": "找到 长期记忆 2 条",
  "long_term": [
    {
      "content": "用户下周有个重要会议，感到有点愁/焦虑",
      "memory_type": "event",
      "importance": 8
    },
    {
      "content": "用户喜欢焦糖玛奇朵",
      "memory_type": "preference",
      "importance": 7
    }
  ]
}

────────────────────────────────────────
[Memory FINISH]
[相关记忆]
- [长期记忆]：
  · 用户下周有个重要会议，感到有点愁/焦虑
  · 用户喜欢焦糖玛奇朵

──────────────────────────────────────────────────
[System OBSERVATION]
记忆检索完成

[system_agent] ReAct 迭代 2/10

──────────────────────────────────────────────────
[System THOUGHT]
记忆检索完成，现在调用 character_agent 生成角色回复

──────────────────────────────────────────────────
[System ACTION] call_agent
   agent_name: character_agent
   memory_context: 用户喜欢焦糖玛奇朵；用户下周有重要会议

[character_agent] ReAct 迭代 1/5

────────────────────────────────────────
[Character THOUGHT]
我需要先分析角色当前的情绪状态

────────────────────────────────────────
[Character ACTION] analyze_emotion

────────────────────────────────────────
[Character OBSERVATION]
{"mood": 0.8, "affection": 0.5, "energy": 0.7, "trust": 0.5}

────────────────────────────────────────
[Character ACTION] generate_response

────────────────────────────────────────
[Character FINISH]
今天天气这么好，当然要！我推荐焦糖玛奇朵呐，冰的更清爽，
加点海盐焦糖超级满足诶嘿嘿。下周会议要加油，我陪你一起打气！

──────────────────────────────────────────────────
[System FINISH]

════════════════════════════════════════════════════════════
[小雪]
════════════════════════════════════════════════════════════

今天天气这么好，当然要！我推荐焦糖玛奇朵呐，冰的更清爽，
加点海盐焦糖超级满足诶嘿嘿。怕下午太上头就选半糖或低因版，
甜甜的又不心慌，再配一块提拉米苏，哼~完美下午茶。

下周会议要加油，我陪你一起打气，喝完跟我说感觉如何嘛？

────────────────────────────────────────────────────────────
迭代次数: 3
是否成功: ✓
────────────────────────────────────────────────────────────
```

</details>

---

## 设计理念

### 核心架构理念

- **ReAct Agent 架构**：基于经典的 ReAct (Reasoning + Acting) 模式，通过 System Prompt 强制 LLM 输出 Thought/Action/Final Answer 格式，实现推理与执行的统一
- **双消息架构**：`_messages` 持久化对话历史（有状态 Agent 使用），`_loop_messages` 单轮 ReAct 轨迹（每次 run 重置）
- **Agent 通信协议**：统一的 `invoke()` 调用接口，通过 `AgentMessage` 和 `AgentResponse` 传递数据，通过 `metadata` 实现松耦合扩展
- **Agent 注册中心**：`AgentRegistry` 管理所有可调用的 Agent，`CallAgent` 工具实现 Agent 间的动态调用
- **工具系统**：`Tool` 基类定义工具接口，`ToolResult` 统一返回格式，`ToolKit` 管理工具集并支持并行执行

### 应用架构理念

- **三 Agent 协作**：System Agent（有状态调度）+ Memory Agent（无状态记忆管理）+ Character Agent（无状态角色扮演）
- **三级记忆系统**：短期（内存滑动窗口）→ 中期（MySQL 摘要）→ 长期（Milvus 向量）
- **分层模型策略**：不同 Agent 使用不同规模的 LLM，平衡效果与成本
- **Go + Python 双语言架构**：Go 负责 gRPC 服务（LLM 代理 + 存储），Python 负责 HTTP 入口、AI 推理与 Agent 逻辑

---

## 技术选型

| 层级 | 技术 | 职责 |
|-----|------|------|
| HTTP API | Python (FastAPI) | HTTP 入口、请求处理 |
| gRPC Services | Go | LLM 代理、存储服务封装 |
| AI Agent | Python | ReAct Agent、多Agent协作、LLM调用、记忆管理 |
| Agent 通信 | Agent Protocol | 统一调用接口、消息传递 |
| 工具系统 | Python ToolKit | 工具定义、执行、并行调度 |
| 存储 | MySQL + Milvus Lite | 结构化存储 + 向量检索 |
| LLM | 内部代理接口 | 支持多模型配置 |

---

## 核心架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                              客户端                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ HTTP
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Python API (FastAPI)                            │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    Agent Registry                            │  │
│   │              （Agent 注册中心）                                │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                      System Agent                            │  │
│   │  （有状态调度 | 对话上下文 | Agent 协调 | 窗口管理 | 摘要）      │  │
│   │                                                              │  │
│   │   ReAct 循环: Thought → Action (CallAgent) → Observation    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│              ┌───────────────┴───────────────┐                      │
│              ▼                               ▼                      │
│   ┌──────────────────────┐      ┌──────────────────────┐           │
│   │    Memory Agent      │      │   Character Agent    │           │
│   │   （无状态记忆管理）   │      │   （无状态角色扮演）   │           │
│   │                      │      │                      │           │
│   │   ReAct: Thought →   │记忆  │   ReAct: Thought →   │           │
│   │   Action (Search) →  │上下文│   Action (Emotion/   │           │
│   │   Action (Store) →   │      │   Response) → Final  │           │
│   │   Final Answer       │      │   Answer             │           │
│   └──────────────────────┘      └──────────────────────┘           │
│                                                                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ gRPC
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Go gRPC Services                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐  │
│  │       llmproxy          │  │            storage              │  │
│  │     LLM 代理服务        │  │       MySQL / Milvus Lite       │  │
│  └─────────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### ReAct Agent 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ReAct Agent 基类                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   System Prompt                               │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐ │ │
│  │  │ 业务提示词       │  │ 工具描述列表      │  │ ReAct 格式   │ │ │
│  │  │ (子类实现)       │  │ (ToolKit生成)    │  │ (强制输出)   │ │ │
│  │  └─────────────────┘  └──────────────────┘  └──────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    ReAct 循环                                  │ │
│  │                                                              │ │
│  │  迭代 1-N:                                                    │ │
│  │    1. LLM 生成: Thought → Action → Action Input               │ │
│  │    2. 工具执行: ToolKit.execute(action, args)                  │ │
│  │    3. 观察: Observation → 追加到 _loop_messages                │ │
│  │                                                              │ │
│  │  结束条件: Final Answer 或超过最大迭代次数                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    最终输出                                     │ │
│  │  ┌────────────────────┐  ┌─────────────────────────────────┐  │ │
│  │  │ get_response_schema │  │ format_final_output              │  │ │
│  │  │ (可选，结构化输出)   │  │ (子类实现，格式化结果)            │  │ │
│  │  └────────────────────┘  └─────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent 通信协议

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Protocol 架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   AgentRegistry                                │ │
│  │                   （Agent 注册中心）                            │ │
│  │  • register(agent): 注册 Agent                                 │ │
│  │  • get(name): 获取 Agent                                       │ │
│  │  • get_call_tool(): 获取 CallAgent 工具                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    CallAgent 工具                               │ │
│  │  • name: "call_agent"                                          │ │
│  │  • execute(agent_name, input, metadata): 调用指定 Agent        │ │
│  │  • 通过 AgentRegistry 获取目标 Agent                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                 Agent 调用流程                                  │ │
│  │                                                              │ │
│  │  调用方                          被调用方                       │ │
│  │    │                               │                            │ │
│  │    │  AgentMessage(content,     │                            │ │
│  │    │    metadata)                │                            │ │
│  │    │──────────────────────────────►│                            │ │
│  │    │                               │ invoke(message)            │ │
│  │    │                               │                            │ │
│  │    │                               │ - 提取 metadata 上下文      │ │
│  │    │                               │ - 设置实例变量              │ │
│  │    │                               │ - 执行 ReAct 循环           │ │
│  │    │                               │                            │ │
│  │    │  AgentResponse(content,     │                            │ │
│  │    │    metadata)                │                            │ │
│  │    │◄──────────────────────────────│                            │ │
│  │    │                               │                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**设计原则**：
- **最小化**：只定义 `content`（必需）+ `metadata`（可选扩展）
- **通用性**：不预设任何业务字段，各 Agent 自行约定 metadata 结构
- **松耦合**：调用方和被调用方通过 metadata 传递定制数据
- **可扩展**：新增 Agent 只需实现 `AgentProtocol` 接口

### 三层记忆系统

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Memory Agent 三级记忆架构                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               短期记忆 (Short-term)                            │ │
│  │  • 存储：内存滑动窗口（System Agent._messages）                  │ │
│  │  • 访问：直接携带在对话上下文中                                 │ │
│  │  • 触发：窗口满时生成摘要，存入中期记忆                          │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
│                              │ 窗口满时触发摘要                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               中期记忆 (Mid-term)                              │ │
│  │  • 存储：MySQL（LLM 生成的摘要 + 关键词）                       │ │
│  │  • 访问：最近 N 条自动携带 + 工具检索                           │ │
│  │  • 检索：BM25 粗排 → 多因子精排                                 │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
│                              │ 高频访问自动提升                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               长期记忆 (Long-term)                             │ │
│  │  • 存储：Milvus 向量数据库                                      │ │
│  │  • 访问：工具检索（语义向量相似度）                             │ │
│  │  • 来源：Agent 主动存储 / 中期记忆提升                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| 层级 | 存储 | 容量 | 访问方式 | Agent 职责 |
|-----|------|------|----------|-----------|
| 短期记忆 | 内存 | 20条消息 | 自动携带 | System Agent 维护 |
| 中期记忆 | MySQL | 无限制 | 最近N条自动携带 + 工具检索 | Memory Agent 检索 |
| 长期记忆 | Milvus | 无限制 | 工具检索 | Memory Agent 检索/存储 |

### Memory Agent 工具

| 工具名 | 功能 |
|--------|------|
| `search_memory` | 检索中期记忆和长期记忆（BM25 + 向量相似度） |
| `store_long_term_memory` | 存储长期记忆 |

### Character Agent 工具

| 工具名 | 功能 |
|--------|------|
| `analyze_emotion` | 分析角色当前情绪状态 |
| `generate_response` | 基于人设和情绪生成角色回复 |

---

## 多Agent协作流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  System Agent                                                    │
│  • 维护对话上下文 (_messages)                                    │
│  • ReAct 循环                                                    │
│    1. Thought: 需要检索记忆                                      │
│    2. Action: call_agent (memory_agent)                          │
│    3. Observation: 记忆结果                                       │
│    4. Thought: 需要生成回复                                      │
│    5. Action: call_agent (character_agent)                       │
│    6. Observation: 角色回复                                       │
│    7. Final Answer: 返回角色回复                                  │
│  • 窗口满时触发摘要存储                                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Memory Agent      │     │  Character Agent    │
│                     │     │                     │
│  • 无状态           │ ───►│  • 无状态           │
│  • invoke() 接收    │记忆 │  • invoke() 接收    │
│    conversation_    │上下文│    memory_context   │
│    history          │     │    conversation_    │
│  • ReAct 循环       │     │    history          │
│    1. Search        │     │  • ReAct 循环       │
│    2. Store (可选)  │     │    1. Analyze       │
│    3. Final Answer  │     │       Emotion       │
│  • 返回记忆上下文    │     │    2. Generate      │
│                     │     │       Response       │
│                     │     │    3. Final Answer  │
└─────────────────────┘     │  • 返回角色回复和   │
                             │    情绪状态        │
                             └─────────────────────┘
                                   │
                                   ▼
                              最终输出
```

---

## 目录结构

```
bot_agent/
├── gateway/                    # [Go] gRPC 服务
│   ├── cmd/server/             # 启动入口
│   └── internal/
│       ├── llmproxy/           # LLM 代理服务
│       └── storage/            # 存储服务 (MySQL/Milvus)
│
├── api/                        # [Python] HTTP API 入口
│   └── main.py                 # FastAPI 启动入口
│
├── agent/                      # [Python] Multi-Agent 核心
│   ├── agents/                 # Agent 实现
│   │   ├── base.py             # Agent 基类 (ReAct 循环)
│   │   ├── protocol.py         # Agent 通信协议
│   │   ├── system/             # System Agent (有状态)
│   │   │   ├── system_agent.py
│   │   │   ├── config.py
│   │   │   ├── summarizer.py   # 对话摘要器
│   │   │   └── tools/
│   │   │       └── call_agent.py # CallAgentTool (自动注入 conversation_history)
│   │   ├── memory/             # Memory Agent (无状态)
│   │   │   ├── memory_agent.py
│   │   │   ├── manager.py      # 记忆管理器
│   │   │   ├── retrieval/      # 检索模块
│   │   │   │   ├── bm25.py
│   │   │   │   ├── query_rewriter.py
│   │   │   │   └── ranker.py
│   │   │   └── tools/
│   │   │       ├── search.py   # 检索工具
│   │   │       └── store.py    # 存储工具
│   │   └── character/          # Character Agent (无状态)
│   │       ├── character_agent.py
│   │       ├── persona.py      # 人设定义
│   │       ├── config.py
│   │       └── tools/
│   │           ├── emotion.py  # 情绪分析工具
│   │           └── response.py # 回复生成工具
│   │
│   ├── tools/                  # 工具基础设施
│   │   └── base.py             # Tool 基类、ToolResult、ToolKit
│   ├── core/                   # 核心基础设施
│   │   └── llm.py              # LLM 客户端
│   ├── client/                 # gRPC 客户端
│   │   ├── llm_client.py       # LLM 代理客户端
│   │   └── storage_client.py   # 存储服务客户端
│   └── pb/                     # gRPC 协议 (protobuf)
│
├── proto/                      # gRPC 协议定义 (.proto)
├── config/                     # 全局配置
└── deploy/                     # 部署相关
```

---

## 数据流

```
用户请求
    │
    ▼
[Python API] HTTP 入口
    │
    ▼
[System Agent]
    │
    ├─► ReAct 迭代 1: call_agent(memory_agent)
    │      │
    │      ├─► AgentRegistry.get("memory_agent")
    │      │
    │      ├─► MemoryAgent.invoke(AgentMessage)
    │      │      │
    │      │      ├─► 提取 metadata.conversation_history
    │      │      ├─► ReAct 循环
    │      │      │      ├─► search_memory
    │      │      │      └─► store_long_term_memory (可选)
    │      │      └─► AgentResponse(content, metadata.memory_context)
    │      │
    │      └─► Observation: 记忆结果
    │
    ├─► ReAct 迭代 2: call_agent(character_agent)
    │      │
    │      ├─► AgentRegistry.get("character_agent")
    │      │
    │      ├─► CharacterAgent.invoke(AgentMessage)
    │      │      │
    │      │      ├─► 提取 metadata.memory_context + conversation_history
    │      │      ├─► ReAct 循环
    │      │      │      ├─► analyze_emotion
    │      │      │      └─► generate_response
    │      │      └─► AgentResponse(content, metadata.emotion_state)
    │      │
    │      └─► Observation: 角色回复
    │
    ├─► ReAct 迭代 3: Final Answer
    │
    ├─► _on_final_answer: 记录对话到 _messages
    │
    └─► _trim_messages: 窗口满时触发摘要
           │
           ├─► ConversationSummarizer.summarize_and_save()
           │      │
           │      ├─► gRPC → Storage.StoreSummary()
           │      │
           │      └─► MySQL 存储摘要
           │
           └─► 清空 _messages
    │
    ▼
返回用户
```

---

## 核心概念

### ReAct Agent

基于 ReAct (Reasoning + Acting) 架构的 Agent，通过 System Prompt 强制 LLM 输出以下格式：

```
Thought: [思考]
Action: [工具名称]
Action Input: [JSON 参数]

... (重复多次)

Thought: 已完成任务
Final Answer: [最终答案]
```

**核心特点**：
- **强制格式**：通过 System Prompt 模板约束输出格式
- **推理与执行结合**：先思考再行动，可观察执行结果后调整策略
- **工具调用**：通过 ToolKit 管理和执行工具
- **可追溯**：完整的 ReAct 轨迹保存在 `_loop_messages` 中

### Agent Protocol

统一的 Agent 调用协议，实现 Agent 间的松耦合通信。

**核心组件**：
- `AgentMessage`: 输入消息（content + metadata）
- `AgentResponse`: 响应（content + metadata）
- `AgentProtocol`: 统一调用接口（invoke 方法）
- `AgentRegistry`: Agent 注册中心
- `CallAgent`: 通用 Agent 调用工具

**使用示例**：

```python
# 注册 Agent
registry = AgentRegistry()
registry.register(memory_agent)
registry.register(character_agent)

# System Agent 使用 CallAgentTool
class SystemAgent(Agent):
    def __init__(self, registry: AgentRegistry):
        self._call_tool = CallAgentTool(registry, self._messages)

    def get_tools(self):
        return [self._call_tool]

# ReAct 循环中调用
Thought: 需要检索记忆
Action: call_agent
Action Input: {"agent_name": "memory_agent", "input": "用户的问题"}
```

### 工具系统

统一的工具定义和执行框架。

**核心组件**：
- `Tool`: 工具基类（定义 name/description/parameters + execute）
- `ToolResult`: 工具执行结果（success/data/error）
- `ToolKit`: 工具集管理器（注册、获取、执行）

**使用示例**：

```python
class GetWeatherTool(Tool):
    name = "get_weather"
    description = "获取天气信息"
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }

    def execute(self, city: str) -> ToolResult:
        return ToolResult.ok({"city": city, "temp": 25})

# 在 Agent 中使用
def get_tools(self) -> List[Tool]:
    return [GetWeatherTool()]
```

---

## 服务职责

| 服务 | 语言 | 职责 | 状态 |
|-----|------|------|------|
| Python API | Python | HTTP 入口、请求路由 | 无状态 |
| System Agent | Python | 输入/输出检测、Agent 调度、窗口管理 | 有状态 |
| Memory Agent | Python | 三级记忆管理、自主检索 | 无状态 |
| Character Agent | Python | 人设、情感、回复生成 | 无状态 |
| Go LLMProxy | Go | 封装内部 LLM API | 无状态 |
| Go Storage | Go | MySQL/Milvus 操作封装 | 无状态 |

---

## License

MIT

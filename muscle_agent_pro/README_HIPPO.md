# HIPPO Agent (muscle-mem-agent) 代码架构与执行路径深度解析

> HIPPO Agent 是一个基于「肌肉记忆」理念构建的 GUI 自动化 Agent，采用 **Tool-Use 驱动**的扁平化架构，通过 Infeasibility Pre-check + Todo Quality Guard + Done Quality Guard 实现 **pass@1 高可靠性**。在 OSWorld-Verified 基准测试上以 **74.5%**（单次运行）超越人类基线（72.36%），无需多轮 rollout。

---

## 目录

- [整体架构](#整体架构)
- [核心执行流程](#核心执行流程)
- [关键类与职责](#关键类与职责)
- [工具体系 (ToolRegistry)](#工具体系-toolregistry)
- [Tool-Use 驱动循环](#tool-use-驱动循环)
- [Infeasible Agent (可行性预检)](#infeasible-agent-可行性预检)
- [Code Agent (代码执行子循环)](#code-agent-代码执行子循环)
- [Sub-Agent (Pac-Agent 批量 GUI)](#sub-agent-pac-agent-批量-gui)
- [质量守卫机制](#质量守卫机制)
- [状态管理：Scratchpad + Todo](#状态管理scratchpad--todo)
- [LLM 引擎层](#llm-引擎层)
- [上下文管理策略](#上下文管理策略)
- [目录结构](#目录结构)
- [与 Agent-S3 对比](#与-agent-s3-对比)
- [运行方式](#运行方式)

---

## 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                      cli_app.py (入口)                          │
│  解析 CLI 参数 → 构建 engine_params → 实例化组件                │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────┐
│                    AgentMm (agents/agent.py)                    │
│  两阶段入口：Phase "infeasible" → Phase "execute"               │
│                                                                 │
│  Phase "infeasible":                                            │
│    ┌────────────────────────────────┐                           │
│    │ InfeasibleAgentManager         │                           │
│    │  ├─ LMMAgent (可行性判断)      │                           │
│    │  ├─ InfeasibleResultToolProvider│                          │
│    │  └─ CodeAgent (只读模式)       │                           │
│    └──────────────┬─────────────────┘                           │
│                   │ FEASIBLE / NO_DECISION                      │
│                   ▼                                              │
│  Phase "execute":                                               │
│    ┌─────────────────────────────────────────────────────┐      │
│    │ Worker (agents/worker.py)                           │      │
│    │  ├─ generator_agent (LMMAgent)  ← Tool-Use 驱动     │      │
│    │  │   tool_choice={"type":"any"} 强制工具调用          │      │
│    │  └─ OSWorldACI (grounding.py)                       │      │
│    │      ├─ ToolRegistry (统一工具注册/分发)              │      │
│    │      ├─ UIActions (GUI 操作)                         │      │
│    │      ├─ ExecutionToolProvider (bash/web_search/...)  │      │
│    │      ├─ TodoToolProvider + TodoManager               │      │
│    │      ├─ ScratchpadToolProvider                       │      │
│    │      ├─ CodeAgentManager + CodeAgentToolProvider     │      │
│    │      ├─ SubAgentManager + SubAgentToolProvider       │      │
│    │      ├─ InfeasibleAgentManager (共享引用)             │      │
│    │      ├─ grounding_model (UI-TARS, 坐标定位)          │      │
│    │      ├─ image_grounding_model (图片区域定位)          │      │
│    │      └─ text_span_agent (OCR 文本定位)               │      │
│    └─────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```

---

## 核心执行流程

### 主循环（cli_app.py: `run_agent()`）

```python
for step in range(15):                           # 最多 15 步
    screenshot = pyautogui.screenshot()           # 1. 感知：截屏
    screenshot.resize(scaled_width, scaled_height) #    缩放到 ≤2400px
    obs["screenshot"] = screenshot_bytes

    info, code = agent.predict(instruction, obs)   # 2. 推理

    if "done"/"fail" in code: break               # 3. 终止判断
    if "wait" in code: sleep(5); continue         # 4. 等待
    exec(code[0])                                  # 5. 执行 pyautogui 代码
```

### AgentMm 两阶段 predict（agent.py）

```
predict(instruction, obs)
  │
  ├─ Phase "infeasible":
  │   result = InfeasibleAgentManager.generate_next_action(instruction, obs)
  │   ├─ status == "INFEASIBLE"  → 返回 FAIL，任务结束
  │   ├─ status == "FEASIBLE"    → 缓存可行报告，切换到 "execute"
  │   ├─ status == "NO_DECISION" → 切换到 "execute"
  │   └─ status == "ACTION"      → 执行初步动作（罕见）
  │
  └─ Phase "execute":
      Worker.generate_next_action(instruction, obs)
      → 返回 (executor_info, [exec_code])
```

---

## 关键类与职责

### 1. `AgentMm` (agent.py)

```
继承: UIAgent
职责: 两阶段 Agent 入口
核心属性:
  - _phase: "infeasible" | "execute" | "done"
  - last_infeasible_payload: 可行性判断结果缓存
核心方法:
  - predict(): 根据当前 phase 分发到 InfeasibleAgent 或 Worker
  - _run_infeasible_step(): 调用 InfeasibleAgentManager
  - _cache_feasible_report(): 缓存可行报告到 OSWorldACI
```

### 2. `Worker` (worker.py)

```
继承: BaseModule
职责: 核心执行循环，使用 Anthropic Tool-Use 协议驱动工具调用
核心属性:
  - generator_agent: LMMAgent，通过 tool_choice={"type":"any"} 强制工具调用
  - todo_request_seen: 是否已收到 TodoWrite（用于质量守卫）
  - initial_done_retries: 过早 done 的重试计数
  - max_initial_done_retries: 默认 3
特点:
  - 无 reflection_agent（enable_reflection 强制为 False）
  - 内循环：工具调用的结果可以注入回消息历史继续调用（如 bash/TodoWrite）
  - 外循环：每次外循环 break 后返回一个 exec_code
```

### 3. `OSWorldACI` (grounding.py)

```
继承: ACI
职责: Agent-Computer Interface，统一管理所有工具和子代理
核心组件:
  - ToolRegistry: 中心化工具注册和分发
  - grounding_model: UI-TARS 视觉坐标定位
  - image_grounding_model: 图片区域定位（可独立配置）
  - text_span_agent: OCR 文本定位
  - code_agent_manager: Code Agent 管理器
  - subagent_manager: Sub-Agent 管理器
  - infeasible_agent_manager: 可行性检查管理器
  - todo_board: TodoManager
  - scratchpad: List[str] 便签缓冲
  - execution_history: List[Dict] 执行历史
状态管理:
  - last_subagent_result / last_code_agent_result / last_infeasible_agent_result
  - pending_feasible_report / initial_screenshot / second_screenshot
```

### 4. `ToolRegistry` (tools/registry.py)

```
职责: 统一的工具注册、Schema 生成和分发
核心机制:
  - @tool_action 装饰器标记工具方法
  - 从 Python 类型注解自动生成 JSON Schema（input_schema）
  - 支持 tool_input_schema 类属性覆盖自动生成的 Schema
  - build_tools(allow=..., deny=...): 生成 Anthropic 格式工具列表
  - dispatch(name, input_dict): 路由到正确的处理函数
```

---

## 工具体系 (ToolRegistry)

### UI 操作工具 (UIActions, ui_actions.py)

| 工具 | 定位方式 | 说明 |
|------|----------|------|
| `click(element_description, num_clicks, button_type, hold_keys)` | UI-TARS 坐标 | 点击 UI 元素 |
| `click_image(element_description, ...)` | **image_grounding_model** | 点击图片区域（独立模型） |
| `type(element_description, text, overwrite, enter)` | UI-TARS 坐标 | 输入文本（支持 Unicode 剪贴板） |
| `drag_and_drop(starting, ending, hold_keys)` | UI-TARS ×2 | 拖拽 |
| `scroll(element_description, clicks, shift)` | UI-TARS 坐标 | 滚动 |
| `highlight_text_span(starting_phrase, ending_phrase)` | OCR + LLM | 文本选择 |
| `set_cell_values(cell_values, app_name, sheet_name)` | 无 | LibreOffice UNO bridge |
| `switch_applications(app_code)` | 无 | 平台特定应用切换 |
| `open(app_or_filename)` | 无 | 打开应用/文件 |
| `hotkey(keys)` | 无 | 组合键 |
| `hold_and_press(hold_keys, press_keys)` | 无 | 按住+按 |
| `wait(time)` | 无 | 等待 |
| `done()` | 无 | 任务完成 |
| `fail()` | 无 | 任务失败 |
| `report_infeasible(reason, evidence)` | 无 | 报告不可行 |

### 执行工具 (ExecutionToolProvider, exec_tools.py)

| 工具 | 说明 | 可用角色 |
|------|------|----------|
| `bash(command, timeout_sec)` | 执行 shell 命令 | Worker(隐藏)、CodeAgent、InfeasibleAgent |
| `web_search(query)` | Tavily API 搜索 | CodeAgent、InfeasibleAgent |
| `web_fetch(url, fields)` | Jina Reader + LLM 字段提取 | CodeAgent、InfeasibleAgent |
| `scholarly_author/publication(...)` | Google Scholar | CodeAgent |

### 状态工具

| 工具 | 说明 |
|------|------|
| `TodoWrite(items)` | 创建/更新 Todo 列表（max 20 items） |
| `save_scratchpad(text)` | 保存便签（max 100 items） |
| `read_scratchpad(limit)` | 读取便签 |

### Agent 调用工具

| 工具 | 说明 |
|------|------|
| `call_code_agent(task, max_rounds)` | 调用 Code Agent |
| `call_subagent(subagent, instruction, max_rounds)` | 调用 Pac-Agent |
| `call_infeasible_agent()` | 调用可行性检查 Agent |

### 工具可见性控制

Worker 可见工具（`get_anthropic_tools()` deny 列表过滤）：

```python
deny = ["web_search", "python", "web_fetch", "bash",
        "call_infeasible_agent", "scholarly_publication", "scholarly_author"]
```

Worker 通过 `call_code_agent` 间接访问 bash/web_search 等被隐藏的工具。

---

## Tool-Use 驱动循环

Worker 的核心创新是使用 **Anthropic Tool-Use 协议** 替代自由文本生成 + 代码解析：

```python
# Worker.generate_next_action() 核心循环
while True:
    response = call_llm_safe(
        generator_agent,
        tools=tools,
        tool_choice={"type": "any"},       # 强制工具调用
    )

    tool_use = extract_tool_use(response)   # 从 response.content 提取 tool_use block
    tool_name = tool_use["name"]
    tool_input = tool_use["input"]

    # 分类处理
    if tool_name in {"call_code_agent", "call_subagent", "call_infeasible_agent"}:
        output = grounding_agent.call_tool(tool_name, tool_input)
        append_tool_result(tool_use_id, output)    # 注入结果
        break                                        # 跳出内循环

    elif is_exec_tool(tool_name) or tool_name in {"TodoWrite", "save_scratchpad", ...}:
        output = grounding_agent.call_tool(tool_name, tool_input)
        append_tool_result(tool_use_id, output)     # 注入结果
        continue                                     # 继续内循环（可链式调用）

    elif tool_name in {"done", "fail", "report_infeasible"}:
        break                                        # 终止

    else:  # UI actions (click, type, scroll, ...)
        output = grounding_agent.call_tool(tool_name, tool_input)
        append_tool_result(tool_use_id, "Done.")
        break                                        # 跳出内循环
```

### 关键设计：内外循环

- **内循环**：bash / TodoWrite / scratchpad 操作后不 break，LLM 可以在同一轮中继续调用工具（链式执行）
- **外循环**：UI 动作 / 子 Agent / done / fail 后 break，返回 exec_code 供主循环执行

---

## Infeasible Agent (可行性预检)

在 Worker 执行之前，InfeasibleAgentManager 评估任务是否可行。

### 工作流程

```
InfeasibleAgentManager.generate_next_action(instruction, obs):
  │
  ├── 确保 session 已初始化（首次调用时加载任务+截图）
  │
  ├── 内循环（tool_choice={"type":"any"}）:
  │   │
  │   ├── report_feasible(reason, evidence)
  │   │   → 返回 InfeasibleAgentStepResult(status="FEASIBLE")
  │   │
  │   ├── report_infeasible(reason, evidence)
  │   │   → 返回 InfeasibleAgentStepResult(status="INFEASIBLE")
  │   │
  │   ├── call_code_agent(task):
  │   │   注入"只读核验"前缀 → "只读核验任务可行性，禁止修改任何文件或系统状态。\n原始任务：..."
  │   │   → 返回 InfeasibleAgentStepResult(status="ACTION")
  │   │
  │   └── 其他工具 (click, scroll, wait, ...):
  │       → 执行后注入结果，继续循环
  │
  └── budget 耗尽 → status="NO_DECISION"
```

### 特点

- 独立的 LLM agent 和消息历史
- 可调用 Code Agent 但强制只读模式
- 可进行 UI 操作（click, scroll 等）收集证据
- 结论通过 `report_feasible` / `report_infeasible` 工具返回

---

## Code Agent (代码执行子循环)

与 Agent-S3 的 Code Agent 不同，HIPPO 的 Code Agent 使用 **Tool-Use 协议** 而非自由文本生成。

### 架构

```
CodeAgent (motor_code_agent.py)
  │
  ├── LMMAgent (独立引擎实例)
  ├── ToolRegistry:
  │   ├── ExecutionToolProvider (bash, web_search, web_fetch, scholarly_*)
  │   ├── TodoToolProvider + TodoManager
  │   ├── ScratchpadToolProvider
  │   └── DoneToolProvider (Done 工具)
  │
  └── query() 函数: 核心执行循环
      for round in range(budget=30):
          response = llm.get_response(tools=..., )
          if tool_use → dispatch → append result → continue
          if Done tool → break → return messages
          else → append "请继续" → continue
```

### 关键特点

- **budget=30**（默认），最大 30 轮工具调用
- **中文系统提示**：包含详细的文件操作、数据格式、文档结构保持指南
- **Tool-Use 驱动**：不生成代码字符串，而是通过工具调用执行 bash 命令
- **独立状态**：自己的 Todo 和 Scratchpad
- **sudo 支持**：系统提示包含 sudo 密码，允许需要提权的操作

### Worker 调用方式

```python
# Worker 通过 call_code_agent 工具调用
call_code_agent(task="...", max_rounds=20)
  │
  ├── CodeAgentToolProvider.call_code_agent():
  │   确定 task（subtask 或 full task）
  │   获取截图
  │   获取 pending_feasible_report（如有）
  │   CodeAgentManager.run_task(task, screenshot, controller, ...)
  │
  └── 返回 summary 文本 → 注入 Worker 消息历史
```

---

## Sub-Agent (Pac-Agent 批量 GUI)

Sub-Agent 用于批量 GUI 操作，如逐个处理界面元素。

### 配置

```python
SUB_AGENT_TYPES = {
    "Pac-Agent": {
        "description": "当需要对当前界面上的一系列元素执行相同的操作...",
        "tools": ["click", "click_image_area", "drag_and_drop", "scroll",
                   "hotkey", "wait", "hold_and_press", "highlight_text_span",
                   "type", "switch_applications", "open", "set_cell_values", "web_search"],
        "prompt": "你是 **Meticulous GUI Executor (极度严谨的 GUI 执行者)**...",
        "max_rounds": 20,
    },
}
```

### 特点

- 独立的 LLM agent + ToolRegistry
- UI 动作执行后自动截图并注入新观察
- 可访问 OSWorld 环境的 env.controller
- 完成后生成摘要

---

## 质量守卫机制

HIPPO Agent 的核心创新之一是三重质量守卫：

### 1. Todo Quality Guard

```python
# 首次 TodoWrite 且 items < 4 时触发
if tool_name == "todowrite" and not todo_request_seen:
    todo_request_seen = True
    if len(items) < 4:
        # 生成更多候选（3~9 轮）
        candidates = _collect_todo_candidates(base_messages, tools, ...,
            min_repeats=3, max_repeats=9, min_items=4)
        # LLM 选择最佳候选
        selected = _select_best_todo_candidate(instruction, screenshot, candidates)
        tool_use = selected
```

**流程**：
1. 初始 TodoWrite 项数不足 → 收集 3~9 个候选 TodoWrite
2. 每个候选基于上一次的 TodoWrite 重试，要求「更有利于 Agent 替用户完成操作」
3. LLM 比较所有候选，选择最优（返回 `{"choice": N}`）
4. 选中项替代原始 tool_use

### 2. Done Quality Guard

```python
# 没有执行过关键动作就调用 done 时触发
if tool_name == "done" and not _has_required_action_history():
    if initial_done_retries < max_initial_done_retries:  # 默认 3
        initial_done_retries += 1
        # 注入提示：「你的任务不是提供建议，而是替用户执行他想做的事情」
        append_tool_result(tool_use_id, "请确认任务是否完成。")
        continue  # 继续内循环，让 LLM 重新选择
```

### 3. Infeasibility Pre-check

在执行前评估任务可行性，避免在不可能完成的任务上浪费步骤（详见 [Infeasible Agent](#infeasible-agent-可行性预检)）。

---

## 状态管理：Scratchpad + Todo

### Scratchpad (tools/scratchpad.py)

```python
class ScratchpadToolProvider:
    @tool_action
    def save_scratchpad(self, text: str) -> str:    # 保存到 grounding.scratchpad (max 100)
    @tool_action
    def read_scratchpad(self, limit: int = 20) -> str:  # 读取最近 N 条
```

便签在任务间重置（`reset_task_state()` → `reset_scratchpad()`）。

### Todo (tools/todo.py)

```python
class TodoToolProvider:
    @tool_action
    def TodoWrite(self, items: List[Dict]) -> str:   # 创建/更新 Todo 列表
```

- 每个 item: `{content: str, status: "pending"|"in_progress"|"completed", activeForm: str}`
- 渲染为 checkbox 格式：`[ ] 待办事项` / `[x] 已完成`
- max 20 items
- Worker 使用 `[ ]`/`[x]` 格式，Code Agent 使用 `☐`/`☒` 格式

---

## LLM 引擎层

### 消息格式：Anthropic 为规范

整个系统统一使用 **Anthropic 消息格式**：

```python
# System
{"role": "system", "content": [{"type": "text", "text": "..."}]}

# User (带图片)
{"role": "user", "content": [
    {"type": "text", "text": "..."},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
]}

# Assistant (工具调用)
{"role": "assistant", "content": [
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "...", "name": "click", "input": {...}}
]}

# Tool Result
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "...", "content": "Done."}
]}
```

`LMMEngineOpenAI` 自动转换为 OpenAI 格式（tool_use → tool_calls, tool_result → role:tool 等）。

### 引擎列表

| 引擎 | Provider | 特殊能力 |
|------|----------|----------|
| `LMMEngineOpenAI` | OpenAI / DashScope / 任何兼容 API | 自动格式转换，reasoning_content 提取 |
| `LMMEngineAnthropic` | Anthropic | **Prompt Caching** + Extended Thinking |
| `LMMEngineAnthropicLR` | Anthropic (中继) | SSL 禁用（httpx.Client verify=False） |
| `LMMEngineGemini` | Gemini | OpenAI 兼容端点 |
| `LMMEngineAzureOpenAI` | Azure | Cost tracking |
| `LMMEngineOpenRouter` | OpenRouter | 多模型路由 |
| `LMMEnginevLLM` | vLLM | 自定义 repetition penalty |
| `LMMEngineHuggingFace` | HF TGI | 固定 model name "tgi" |
| `LMMEngineParasail` | Parasail | 自定义 endpoint |

### Grounding 模型双实例

```
grounding_model:        UI-TARS，用于普通 UI 元素定位（click, type, scroll 等）
image_grounding_model:  独立配置的图片区域定位模型（click_image）
                        未配置时 fallback 到 grounding_model
```

坐标解析优先使用 `<point>x y</point>` 格式，回退到正则 `\d+` 提取。

---

## 上下文管理策略

Worker.flush_messages()：

```python
# 统一策略：保留所有文本，仅保留最近 max_trajectory_length（默认 5）张图片
for agent in [generator_agent, reflection_agent]:
    img_count = 0
    for i in reversed(range(len(agent.messages))):
        for j in range(len(agent.messages[i]["content"])):
            if "image" in type:
                img_count += 1
                if img_count > max_images:
                    del agent.messages[i]["content"][j]
```

---

## 目录结构

```
muscle_mem/
├── cli_app.py                      # CLI 入口
├── core/
│   ├── engine.py                   # 9 种 LLM 引擎
│   ├── mllm.py                     # LMMAgent 封装
│   └── module.py                   # BaseModule 基类
├── agents/
│   ├── agent.py                    # AgentMm + UIAgent 基类
│   ├── worker.py                   # Worker (Tool-Use 驱动循环)
│   ├── grounding.py                # OSWorldACI (工具注册中心)
│   ├── motor_code_agent.py         # CodeAgent + CodeAgentManager
│   ├── subagent.py                 # SubAgentManager + Pac-Agent
│   ├── infeasible_agent.py         # InfeasibleAgentManager
│   ├── verification_agent.py       # VerificationAgent (已禁用)
│   ├── tool_loop.py                # 响应解析工具函数
│   └── tools/
│       ├── registry.py             # ToolRegistry + @tool_action
│       ├── ui_actions.py           # GUI 操作工具集
│       ├── exec_tools.py           # 执行工具集 (bash, web_search, ...)
│       ├── scratchpad.py           # 便签工具
│       └── todo.py                 # Todo 管理工具
├── memory/
│   └── procedural_memory.py        # 所有 prompt 模板（中文）
└── utils/
    ├── common_utils.py             # LLM 调用、图片压缩等
    └── local_env.py                # LocalEnv + LocalController
```

---

## 与 Agent-S3 对比

| 特性 | Agent-S3 | HIPPO Agent |
|------|----------|-------------|
| **LLM 交互协议** | 自由文本生成 + 代码块解析 | **Anthropic Tool-Use**（tool_choice={"type":"any"}） |
| **工具 Schema** | 硬编码在 prompt 中 | **自动从类型注解生成 JSON Schema** |
| **格式校验** | 双重 Formatter + 重试 | **无需格式校验**（Tool-Use 结构化输出） |
| **可行性检查** | 无 | **InfeasibleAgent（前置预检）** |
| **质量守卫** | 无 | **Todo Quality Guard + Done Quality Guard** |
| **Todo 管理** | 无 | **TodoManager（LLM 多候选选择）** |
| **便签系统** | notes 缓冲区（简单 List） | **ScratchpadToolProvider（读写工具）** |
| **执行历史** | worker_history（纯文本） | **execution_history（结构化 Dict）** |
| **Code Agent 交互** | 自由文本 `<thoughts><answer>` | **Tool-Use 协议（bash 工具调用）** |
| **Code Agent budget** | 20 | **30** |
| **图片定位** | 单 grounding model | **双 grounding model（UI + Image）** |
| **批量 GUI** | 无 | **Pac-Agent（独立子代理循环）** |
| **Reflection** | 有（独立 reflection_agent） | **无**（enable_reflection=False） |
| **Prompt 语言** | 英文 | **中文** |
| **推理扩展** | bBoN（多 rollout + VLM 评估） | **无（单次运行 pass@1）** |
| **OSWorld 得分** | 72.6%（需 bBoN） | **74.5%（单次运行）** |

### 核心架构差异总结

1. **S3**：LLM 生成自然语言 + 代码块 → 解析代码 → eval 执行
2. **HIPPO**：LLM 直接调用工具 → ToolRegistry 分发 → 工具返回结果 → 注入消息历史

Tool-Use 协议消除了格式校验的复杂性，使交互更可靠。

---

## 运行方式

### 安装

```bash
cd muscle-mem-agent
pip install -e .
```

### CLI 运行

```bash
# 交互模式
muscle_mem \
  --provider openai \
  --model gpt-4o \
  --ground_provider huggingface \
  --ground_url http://localhost:8000/v1 \
  --ground_model UI-TARS-1.5-7B \
  --grounding_width 1920 \
  --grounding_height 1080

# 单任务模式
muscle_mem ... --task "在浏览器中打开 GitHub 并搜索 HIPPO Agent"

# 启用本地代码执行环境
muscle_mem ... --enable_local_env
```

### Qwen DashScope 测试

```bash
python osworld_setup/run_qwen_test.py
```

### OSWorld 评估

```bash
# 多环境并行评估
python osworld_setup/run_muscle_mem_agent.py

# 单环境顺序评估
python osworld_setup/run_muscle_mem_agent_local.py
```

---

## 环境变量

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | OpenAI / DashScope API |
| `ANTHROPIC_API_KEY` | Anthropic API |
| `GEMINI_API_KEY` | Gemini API |
| `DASHSCOPE_API_KEY` | 阿里 DashScope（Qwen） |
| `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` | Azure OpenAI |
| `HF_TOKEN` | HuggingFace TGI |
| `vLLM_API_KEY` / `vLLM_ENDPOINT_URL` | vLLM 本地部署 |
| `OPENROUTER_API_KEY` | OpenRouter |
| `TAVILY_API_URL` | Tavily 搜索 API |
| `JINA_API_URL` | Jina Reader API |

---

## 关键设计决策总结

1. **Tool-Use 优于自由文本**：消除格式校验，提高结构化输出可靠性
2. **可行性前置检查**：避免在不可能完成的任务上浪费 15 步预算
3. **Todo 多候选选择**：LLM 生成多个 TodoWrite 候选 → 独立 LLM 选最优，确保计划质量
4. **Done 质量守卫**：防止 LLM 「建议式」过早结束，强制要求执行实际动作
5. **工具可见性分层**：Worker 不能直接执行 bash，必须通过 Code Agent 间接访问，增加安全性
6. **内循环链式调用**：bash/TodoWrite/scratchpad 可在同一轮中多次调用，提高效率
7. **双 Grounding 模型**：UI 元素和图片区域使用独立模型，提高定位准确性
8. **pass@1 不依赖 bBoN**：通过质量守卫和架构设计实现单次运行高可靠性

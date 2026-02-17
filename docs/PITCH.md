# AgentGate — AI Agent 预部署质量门禁

## 问题

AI Agent 在企业中大规模部署，但**没有人在它们上线前验证它们是否靠谱**。

- Carnegie Mellon 研究：AI Agent **70% 任务失败**
- MIT 报告：**95% GenAI 试点无法规模化**
- Gartner 预测：**40%+ agentic AI 项目将在 2027 年底前被取消**
- 2025 年至今 **518 起**法院 AI 幻觉案件
- Replit AI agent 删除生产数据库并**撒谎掩盖**

企业每年因 AI 幻觉损失 **$674 亿**（AllAboutAI），每位员工年花 **$14,200** 人工缓解幻觉（Forrester）。

## 现有方案的缺陷

所有现有工具都在做**事后观测**（出了问题再告诉你）：

| 玩家 | 做什么 | 不做什么 |
|------|--------|----------|
| Splunk (2/25 GA) | 生产监控 | 部署前测试 |
| Datadog | LLM 可观测性 | 质量门禁 |
| Arize ($100M 融资) | Trace + Eval | CI/CD 阻断 |
| Langfuse (被 ClickHouse 收购) | 开源观测 | 回归检测 |

**类比**：他们是黑匣子（飞机坠毁后分析原因），我们是起飞前的安全检查。

## 解决方案：AgentGate

**"Cypress/Selenium for AI Agent"** — 部署前的自动化行为验证。

### 工作原理

```
1. 录制 → Agent 运行时自动录制输入/输出
2. 标注 → 人工审核标注"黄金输出"（200-500 条）
3. 测试 → 每次 prompt/model 变更，自动回归测试
4. 门禁 → 测试不通过 = 阻断部署（CI/CD 集成）
```

### 3 行代码接入

```python
from agentgate import TestSuite, SemScore

suite = TestSuite("my-agent", golden="tests/golden.jsonl")
suite.run(my_agent, metrics=[SemScore(threshold=0.75)])
```

### 学术基础

不是拍脑袋做的，每个方法都有论文支撑：

| 能力 | 方法 | 学术来源 |
|------|------|----------|
| 语义回归检测 | 嵌入余弦相似度 | SemScore (Aynetdinov, arXiv 2024) |
| 非确定性处理 | 同一输入跑 K 次取均值 | Pass^K (Cresta 2025) |
| 幻觉检测 | 语义熵 | Farquhar et al. (Nature 2024, 1134 引用) |
| 分布漂移 | PSI + JS 散度 | 经典统计方法 |
| 数据集质量 | 5D 原则 | Google Eval Framework (2025) |

## 市场

### 规模

| 细分 | 2024 | 2029-2030 预测 | CAGR |
|------|------|---------------|------|
| MLOps | $21.9 亿 | $166 亿 | 40.5% |
| AI 治理 | $8.9 亿 | $57.8 亿 | 45.3% |
| LLMOps 子板块 | — | — | 49.0% |

**SAM（可服务市场）**: $50-70 亿 (2029)

### 驱动力

- EU AI Act 2026 年 8 月全面生效
- 2026 年底 40% 企业应用集成 AI agent（Gartner）
- 企业 AI 预算 48% 领导人预计增加 ≥$200 万（Dynatrace）

## 商业模型

| 层级 | 价格 | 目标 |
|------|------|------|
| **Open Source** | $0 | SDK + 核心评估引擎（GitHub 获客） |
| **Team** | $149/月 | 托管 Dashboard + 告警 |
| **Pro** | $499/月 | 高级护栏 + 回归分析 + API |
| **Enterprise** | $2,500+/月 | 私有部署 + SLA + 合规报告 |

### 单位经济学

- 单次评估成本：**$0.0004**（GPT-4o-mini LLM-as-Judge）
- 毛利率：**85-90%**
- LTV/CAC：**18x**
- 盈亏平衡：**44 个付费客户**

## 竞争壁垒

### 短期（0-12 月）
- 开源 SDK 占领开发者心智
- **AgentBench-QA 开放基准** — 定义 Agent 质量的行业标准（类似 SWE-bench）
- 学术论文化的方法论 → 企业技术买家信任

### 中期（12-36 月）
- 评估数据飞轮：跨客户故障模式图谱
- 行业基准线："你的 Agent 在金融行业排第几"
- CI/CD 深度集成 → 高切换成本

### 类比
| 领域 | 定义标准的工具 | 结果 |
|------|--------------|------|
| NLU | GLUE/SuperGLUE | 成为行业通用评估标准 |
| 代码 | SWE-bench | 定义了 AI coding 能力标准 |
| Agent 质量 | **AgentBench-QA (我们)** | ? |

## 团队 & 现状

- Solo founder，技术背景
- MVP SDK 已完成（7 个 Python 模块，可 pip install）
- Landing page + 冷邮件 + 90 天执行计划已就绪
- 下一步：72 小时 PMF 验证 → Design Partner → 首批付费客户

## 风险 & 应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 大厂内置质量门禁 | 中 | 他们做观测不做测试，架构基因不同 |
| 开源工具赶超 | 高 | 数据飞轮 + benchmark 标准 + 企业服务 |
| 付费意愿不足 | 中 | 72h 验证，不成立立刻 pivot |
| Solo founder 瓶颈 | 高 | 6 个月内找技术联创 |

## Ask

寻求建议：
1. 这个方向的切入点是否正确？
2. "预部署测试"vs"事后监控"的定位是否足够差异化？
3. 学术基准（AgentBench-QA）作为核心壁垒是否可行？
4. 冷启动应该先打哪个行业？（金融 vs 医疗 vs 通用 SaaS）

---

> *"世界不缺 AI 监控仪表盘，缺的是让 AI Agent 上线前证明自己靠谱的方法。"*

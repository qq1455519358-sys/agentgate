# AgentGate — AI Agent 预部署质量门禁

> "不是望远镜，是门禁。不是事后监控，是上线前拦截。"

## 一句话

企业 AI Agent 的自动化回归测试和质量门禁平台。

## 核心洞察

- AI Agent 失败率 70-95%（Carnegie Mellon / MIT / Gartner）
- 所有巨头（Splunk/Datadog/Arize）都在做事后**观测**
- 没有人做部署前的**行为验证** — 这是 Selenium/Cypress for AI Agent
- 窗口期 6-9 个月

## 目标用户

部署了 1-3 个 customer-facing AI Agent 的中型 SaaS 公司（50-500 人）

## 核心功能

1. **回归检测器** — 录制黄金输出 → prompt/model 变更后自动对比 → 质量下降时告警+阻断部署
2. **行为测试套件** — pytest 风格声明式 Agent 测试，CI/CD 原生集成
3. **生产护栏** — 实时拦截幻觉/PII/合规违规输出

## 技术栈

- Python SDK（3 行代码接入，OTEL 对齐）
- FastAPI 后端
- ClickHouse（trace 存储）
- Next.js Dashboard
- GitHub Action CI 集成

## 商业模型

| 层级 | 价格 | 目标 |
|------|------|------|
| Open Source | $0 | SDK + 核心评估引擎 |
| Team | $149/月 | 托管 Dashboard + 告警 |
| Pro | $499/月 | 高级护栏 + 回归分析 |
| Enterprise | $2,500+/月 | 私有部署 + SLA + 合规报告 |

## 关键指标

- 盈亏平衡：44 个付费客户
- 毛利率：85-90%
- LTV/CAC：18x
- 6 个月目标：10 个付费客户，ARR $150K+

## 学术-产品平衡策略

### 原则：Research-backed, Product-delivered

每一个产品功能都建立在可验证的学术方法上，但包装成开发者友好的工具。

### 三大支柱

1. **AgentBench-QA 开放基准**
   - 5 个垂直领域 × 200+ 测试用例
   - 公开排行榜 + 可复现实验
   - 目标：成为 Agent 质量的 SWE-bench
   - 本身可作为论文发表

2. **方法论论文化**
   - SemScore (Aynetdinov 2024) → 语义回归
   - Pass^K (Cresta 2025) → 非确定性处理
   - 语义熵 (Farquhar, Nature 2024) → 幻觉检测
   - PSI/JS散度 → 分布漂移
   - Google 5D (2025) → 数据集质量

3. **研究报告获客**
   - 月度《Agent 质量月报》
   - 行业白皮书（金融/医疗/法律）
   - 数据驱动内容 > 付费广告

### 护城河逻辑
谁定义 benchmark = 谁定义行业标准 = 谁拥有话语权

## 竞争定位

| 维度 | Splunk/Datadog | Arize/Langfuse | **AgentGate** |
|------|---------------|----------------|---------------|
| 时机 | 事后 | 事后 | **事前** |
| 动作 | 看 | 看+告警 | **测试+阻断** |
| 用户 | SRE/DevOps | ML Engineer | **QA + Dev** |
| 类比 | APM | APM for AI | **Cypress for AI** |

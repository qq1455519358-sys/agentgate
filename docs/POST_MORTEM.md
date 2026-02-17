# Pitch Post-Mortem: 大师审查后的诚实复盘

## 🔴 硬伤（必须立刻修复）

### 1. 市场规模 10x 虚报
- MLOps: 写了 $21.9B → $166B，实际是 **$2.19B → $16.6B**
- AI 治理: 写了 $8.9B → $57.8B，实际是 **$890.6M → $5.78B**
- 原因：百万和十亿的单位搞混了。这不是小错，投资人看到直接丧失信任。

### 2. 学术引用造假/失实
- "Google 5D Framework (2025)" — **不存在**。是 Wang & Strong 1996 的通用数据质量维度，被错误归因给 Google。
- Pass^K — **不是论文**，是 Cresta 的博客文章。标注为学术来源是误导。
- SemScore 引用数：写了 1134，实际 Nature 论文 532（可能混淆了 ICLR 预印本数据）
- Farquhar 的 Nature 论文本身没问题，但引用数夸大

### 3. $674 亿损失数据不可信
- 来源 AllAboutAI 是 SEO 内容站，不是研究机构
- 没有 Gartner/Forrester/McKinsey 背书的大数字不能用

## 🟡 竞争认知严重不足

### 之前的认知（错误）
"所有竞品都在做事后观测，没人做部署前门禁"

### 实际情况
- **DeepEval** (13.6K ⭐, YC W25, $2.7M): 原生 pytest 集成，assert_test() 支持 CI/CD 阻断，v3.0 已加 agent 专属指标
- **promptfoo** (10.4K ⭐, a16z+Insight $23.6M): GitHub Action，200K+ 开发者，80+ Fortune 500
- **Braintrust** ($45M, a16z): GitHub Action PR 触发评估
- **Arize** ($131M 不是 $100M): 已有 CI/CD experiment 集成
- **Datadog**: LLM Experiments GA，有 GitHub Actions 模板
- **平台方**: Azure AI Foundry / UiPath / Google Vertex 原生集成部署门禁

### Humanloop 警告信号
Humanloop 明确做"quality gates for deployment"→ 被 Anthropic 收购（2025.8）
说明：独立质量门禁公司可能被大厂吸收而非独立存活

## 🟢 大师认可的部分

1. **问题真实存在**: 70% 失败率（CMU）、95% 试点失败（MIT）、40% 项目将砍（Gartner）— 全部核实
2. **EU AI Act 合规窗口**: Article 9 + Article 43 要求预部署评估，2026.8 生效
3. **"Cypress for AI Agent" 定位未被占领**: 功能存在但品牌定位空白
4. **开发者痛点数据**: 52% 做离线评估，29.5% 完全不评估（LangChain 报告）

## 🎯 三个真正的差异化方向

大师指出的，也是最有价值的部分：

### 1. 集成测试（不是单元测试）
> "Evals are essentially unit tests. They test the logic of the node, but they do not test the integrity of the graph."
- DeepEval/promptfoo 测试单个组件
- **没人测试完整 agent 工作流**：多步骤、多工具调用、多 agent 协作、故障级联
- 这才是真正的 Cypress 类比 — Cypress 测的也不是单个函数，是完整用户旅程

### 2. 合规文档生成
- EU AI Act 要求 conformity assessment artifacts
- 一个能同时生成测试报告 + 合规文档的工具 → 合规驱动采购
- 纯开发者工具（DeepEval/promptfoo）不做这个

### 3. 企业治理层
- 角色审批链、变更管理、审计日志、版本追溯
- 受监管行业（金融/医疗/政府）的刚需
- 开源工具的盲区

## 📐 修正后的诚实定位

### 之前（过度宣称）
"没人做部署前质量门禁，我们是唯一选择"

### 之后（诚实+差异化）
"DeepEval/promptfoo 做了单元测试级的质量门禁。我们做的是：
1. **Agent 工作流的集成测试**（测完整图，不只是节点）
2. **合规证据生成**（EU AI Act conformity assessment）
3. **企业治理层**（审批链 + 审计日志 + 版本追溯）"

### 一句话
"DeepEval 是 Agent 的 pytest，我们是 Agent 的 Playwright E2E + SOC2 审计"

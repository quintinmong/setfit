# SetFit 中文多维意图分类项目

本项目是一个基于 **SetFit (Sentence Transformer Fine-tuning)** 与 **BGE 中文语义向量底座** 构建的轻量、高效、多维度的中文意图识别系统。

该系统专为金融理财/手机银行等高对抗度、小样本（Few-Shot）场景设计，能够对用户发出的提问或抱怨进行多维度的语义解析，包含：**场景维度**（咨询、抱怨、主动服务）、**思维维度**（快/慢思考）、**情绪维度**（平静、愤怒、焦虑）以及其他专项信号（如资产信号、系统技能、信任行为）。

---

## 🌟 项目亮点

1. **小样本微调（Few-Shot Fine-tuning）**：基于 SetFit 框架，仅需极少（每类十几条）样本即可训练出高泛化能力的意图分类器。
2. **底座共享，多路解耦**：系统采用“单个 BGE 编码底座 + 多路轻量机器学习分类头（Logistic Regression）”的架构。对于任意一句话，仅提取一次语义嵌入特征，然后并行输入多路分类头中，不仅降低了计算资源消耗，同时大幅提升了推理速度（实现秒开级加载与毫秒级预测）。
3. **针对高对抗混合集优化**：数据集设计了大量带有混合情绪、模糊指代以及同业对比的测试样本，模型针对金融场景有针对性的微调。

---

## 📂 项目结构

```text
.
├── .env.example             # 大模型 API 与网络超参数配置文件模板
├── ANTIGRAVITY.md           # Antigravity 开发者指南（开发命令与代码风格规范）
├── README.md                # 本文档（项目简介与运行手册）
├── requirements.txt         # 项目依赖库清单
├── temporal_signal_server.py # 多轮对话时序信号追踪的冷启动原型测试服务
├── agent_router.py          # 融合 [6维意图+多轮时序+NER槽位] 的全功能 Agent 终极大脑总路由
│
├── intent_classification/   # 意图分类专属模块目录
│   ├── download_model.py        # 基础底座模型下载脚本（从 ModelScope 下载 BGE-small-zh-v1.5）
│   ├── intent_demo.py           # 二分类 SetFit 经典训练及预测演示 Demo
│   ├── six_intent_server.py     # 六路并行意图训练与保存服务
│   └── use_six_intents_model.py # 轻量加载持久化六分类模型进行推理预测的代码示例
│
├── temporal_signal/
│   ├── llm_generate_temporal_seeds.py # 利用大模型 API 进行多轮会话冷启动数据增强的脚本
│   ├── train_temporal_model.py # 载入 BGE 共享底座正式训练 GRU 时序分类模型的脚本
│   └── temporal_signal_llm_augmented.json # 大模型增强得到的纯文本多轮会话训练集
│
├── ner_slot/
│   ├── ner_generate_corpus.py # 利用大模型 API 进行 NER 槽位数据增强的脚本
│   ├── ner_infer_and_slot_fill.py # 字符级 BIO 对齐与槽位三态状态机决策业务引擎（含现场测试）
│   └── ner_raw_corpus.json    # 大模型增强生成的 100 条高质量 NER 标注与槽位训练数据集
│
├── models/
│   └── bge-small-zh-v1.5/   # 预训练基础模型底座（运行 download_model.py 后自动创建）
│
└── my_final_six_intents_model/
    ├── bge_encoder/         # 微调后的共享底座编码器
    ├── temporal_gru_weights.pth # 训练好的 GRU 时序分类器模型权重
    └── classification_heads.pkl # 序列化的 6 个独立分类头字典文件
```

---

## 🚀 快速上手

### 0. 一键全自动流水线 (全新推荐)
如果您想最快速度跑通整个项目（包含下载底座模型、训练意图与时序网络、并启动最终大脑总路由推理测试），可在项目根目录下直接一键运行：
```bash
chmod +x run_all.sh && ./run_all.sh
```
该脚本会智能识别您的 `.venv312` 或 `.venv` 虚拟环境，并全自动按顺序执行：
1. **模型下载**（从 ModelScope 镜像源拉取底座）
2. **六路意图训练**（微调底座并保存分类头）
3. **时序追踪训练**（提取时序特征并训练 GRU 模型）
4. **大脑路由集成测试**（全闭环实时推理，生成去 AI 感的大模型客服话术）

### 1. 环境准备
确保您的 Python 环境为 3.12 或 3.14。执行以下命令安装依赖：

```bash
# 激活您的虚拟环境（如 .venv）
source .venv/bin/activate

# 安装所需依赖包
pip install -r requirements.txt
```

### 2. 下载基础底座模型
运行 `intent_classification/download_model.py`，脚本将自动通过阿里云 ModelScope 镜像源极速下载 `BAAI/bge-small-zh-v1.5`：
```bash
python3 intent_classification/download_model.py
```

### 3. 训练与保存六路意图分类器
在 60 条高密度混合场景数据集上微调编码底座并训练 6 个完全解耦的分类头，拟合后保存至本地硬盘：
```bash
python3 intent_classification/six_intent_server.py
```

### 4. 极速推理预测
通过加载本地已训练好的六分类模型，仅耗时约 0.2 秒载入，即可进行实时文本意图分类预测：
```bash
python3 intent_classification/use_six_intents_model.py
```

### 5. 多轮会话时序信号追踪 (新增)
本项目支持将静态单句分类升级为连续 3 轮会话的时序意图倾向性追踪（包含：意向平稳 `0`、购买升温 `1`、流失真拒绝 `2`、异议成功化解 `3`）：
*   **冷启动测试与现场推理原型服务**（使用 11 条硬编码样本快速拟合 GRU 并跑现场测试）：
    ```bash
    python3 temporal_signal_server.py
    ```
*   **大模型语料冷启动数据增强**（可选，需要在项目根目录 `.env` 中配置好 API Key 运行，默认生成 240 条样本）：
    ```bash
    python3 temporal_signal/llm_generate_temporal_seeds.py
    ```
*   **正式时序网络模型训练**（自动读取增强的 JSON 文本集并使用微调 BGE 底座做特征抽取，用 GRU 拟合并保存权重）：
    ```bash
    python3 temporal_signal/train_temporal_model.py
    ```

### 6. 命名实体识别 (NER) 与槽位填充 (Slot Filling) (新增)
系统支持自动对用户发言提取实体（包含：时间 `TIME`、等待前置条件 `WAIT_COND`、决策关系人 `RELATION_PERSON`、目标材料产品 `MATERIAL`），进行字符级 BIO 对齐标签转换，并通过槽位状态机（包含 FILLED / SUGGESTED / EMPTY 三态）输出给下游话术决策：
*   **NER 冷启动语料生成**（可选，需配置 `.env`，生成 100 条数据）：
    ```bash
    python3 ner_slot/ner_generate_corpus.py
    ```
*   **槽位填充业务引擎验证**（自动加载 JSON 语料库测试 BIO 转换与槽位状态看板）：
    ```bash
    python3 ner_slot/ner_infer_and_slot_fill.py
    ```

### 7. 闭环 Agent 全功能大脑路由系统 (新增)
终极大脑主入口。它会统一调用共享微调底座，并行计算 6 路意图、通过 GRU 进行时序研判、读取当前 NER 实体驱动状态机，融合出叙事化状态提示词并根据快慢思考路由大模型，输出最终去 AI 感的话术：
```bash
python3 agent_router.py
```

---

## 📊 架构解析

```mermaid
graph TD
    UserQuery[用户输入文本] --> Encoder[共享底座 SentenceTransformer]
    Encoder --> |仅提取一次高维向量特征| Embeddings[高维语义特征 X]
    
    subgraph 并行分类头 (Logistic Regression)
        Embeddings --> HeadSC[场景维度分类头]
        Embeddings --> HeadTH[思维维度分类头]
        Embeddings --> HeadEM[情绪维度分类头]
        Embeddings --> HeadAS[资产信号分类头]
        Embeddings --> HeadSK[系统技能分类头]
        Embeddings --> HeadTR[信任行为分类头]
    end

    HeadSC --> Output1[场景识别: 咨询/抱怨/主动服务]
    HeadTH --> Output2[思维识别: 快思考/慢思考]
    HeadEM --> Output3[情绪识别: 平静/愤怒/焦虑]
    HeadAS --> Output4[资产识别: 包含/不包含]
    HeadSK --> Output5[技能识别: 包含/不包含]
    HeadTR --> Output6[信任识别: 包含/不包含]
```

通过将复杂的 NLP 微调问题解耦为：
1. **统一底座微调与向量化**：获取丰富的多维通用高维特征。
2. **轻量线性分类器分类**：对高阶稀疏意图做决策面拟合。

该设计成功实现了小样本下极高泛化度与毫秒级端到端时延的最佳平衡。

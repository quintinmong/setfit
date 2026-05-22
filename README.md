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
├── ANTIGRAVITY.md           # Antigravity 开发者指南（开发命令与代码风格规范）
├── README.md                # 本文档（项目简介与运行手册）
├── requirements.txt         # 项目依赖库清单
├── download_model.py        # 基础底座模型下载脚本（从 ModelScope 下载 BGE-small-zh-v1.5）
├── intent_demo.py           # 二分类 SetFit 经典训练及预测演示 Demo
├── three_intent_server.py   # 三路（场景、思维、情绪）并行意图分类训练与验证服务
├── six_intent_server.py     # 六路（场景、思维、情绪、资产、技能、信任）并行意图训练与保存服务
├── use_six_intents_model.py # 轻量加载持久化六分类模型进行推理预测的代码示例
│
├── models/
│   └── bge-small-zh-v1.5/   # 预训练基础模型底座（运行 download_model.py 后自动创建）
│
└── my_final_six_intents_model/
    ├── bge_encoder/         # 微调后的共享底座编码器
    └── classification_heads.pkl # 序列化的 6 个独立分类头字典文件
```

---

## 🚀 快速上手

### 1. 环境准备
确保您的 Python 环境为 3.12 或 3.14。执行以下命令安装依赖：
```bash
# 激活您的虚拟环境（如 .venv）
source .venv/bin/activate

# 安装所需依赖包
pip install -r requirements.txt
```

### 2. 下载基础底座模型
运行 `download_model.py`，脚本将自动通过阿里云 ModelScope 镜像源极速下载 `BAAI/bge-small-zh-v1.5`：
```bash
python3 download_model.py
```

### 3. 运行三路意图分类演示
对「场景」、「思维」、「情绪」三个相互独立的语义维度进行并行训练并执行现场测试：
```bash
python3 three_intent_server.py
```

### 4. 训练与保存六路意图分类器
在 60 条高密度混合场景数据集上微调编码底座并训练 6 个完全解耦的分类头，拟合后保存至本地硬盘：
```bash
python3 six_intent_server.py
```

### 5. 极速推理预测
通过加载本地已训练好的六分类模型，仅耗时约 0.2 秒载入，即可进行实时文本意图分类预测：
```bash
python3 use_six_intents_model.py
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

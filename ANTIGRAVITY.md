# Antigravity 开发者指南

本文件定义了针对 `setfit` 项目的开发指令与代码规范。当 AI 助手（如 Antigravity）或开发人员在本项目中工作时，应当优先遵循此指南。

---

## 🛠 开发环境与常用命令

项目包含两个虚拟环境：
- `.venv` (Python 3.14.5)
- `.venv312` (Python 3.12.13)

默认推荐使用 `.venv` 虚拟环境。

### 1. 环境准备与激活
```bash
# 激活默认 Python 3.14.5 虚拟环境
source .venv/bin/activate

# 或激活 Python 3.12.13 虚拟环境
source .venv312/bin/activate
```

### 2. 核心运行与微调命令

- **下载基础模型底座（MacBERT-base 中文版）**
  ```bash
  .venv/bin/python3 intent_classification/download_model.py
  ```

- **运行少样本（Few-Shot）微调与二分类 Demo**
  ```bash
  .venv/bin/python3 intent_classification/intent_demo.py
  ```

- **训练与保存六路意图分类模型（场景、思维、情绪、资产信号、系统技能、信任行为）**
  ```bash
  .venv/bin/python3 intent_classification/six_intent_server.py
  ```

- **加载持久化模型进行快速推理测试**
  ```bash
  .venv/bin/python3 intent_classification/use_six_intents_model.py
  ```

- **多轮对话时序信号追踪 (测试与演示的原型服务)**
  ```bash
  .venv/bin/python3 temporal_signal_server.py
  ```

- **多轮对话时序信号大模型数据增强 (使用 DeepSeek 快速生成高密度冷启动语料)**
  ```bash
  # 运行前请在 temporal_signal/.env 中配置好你的 LLM_API_KEY
  .venv312/bin/python3 temporal_signal/llm_generate_temporal_seeds.py
  ```

- **训练时序 GRU 分类网络模型 (利用增强好的 JSON 文本数据集)**
  ```bash
  .venv312/bin/python3 temporal_signal/train_temporal_model.py
  ```

- **一键跑通全自动集成流水线 (包含下载底座、6路意图头微调、时序模型训练以及大脑总路由验证)**
  ```bash
  chmod +x run_all.sh && ./run_all.sh
  ```

---


## 📐 代码风格与技术规范

### 1. 库导入与补丁规则
由于 `transformers` 库的 `training_args` 在某些 Python/库版本下初始化可能会报错，所有涉及 `setfit` / `transformers` 训练的代码必须在最顶部（即引入 transformers 库之前）应用以下猴子补丁（Monkey Patch）：
```python
import transformers.training_args

if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
```

### 2. 硬件加速（MPS/CUDA）使用
在实例化 `MacBertEncoder` 底座或运行 PyTorch 模型时，优先进行多后端加速判断：
```python
import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

### 3. 多分类头与特征抽取设计
- 避免对相同句子重复调用 `MacBertEncoder.encode` 提取高维特征。
- 应当一次性对全部输入进行向量化得到 `X_train`，再分别传入各维度的分类头（如 Logistic Regression）进行独立并行拟合。
- 保存多路分类头时，建议使用单个字典打包，并使用 `joblib.dump` 持久化，例如：
  ```python
  import joblib
  heads_dict = {
      "scene": head_sc,
      "thinking": head_th,
      # ...
  }
  joblib.dump(heads_dict, "classification_heads.pkl")
  ```

### 4. 代码风格要求
- 统一使用 `PEP 8` 编码规范。
- 脚本中添加关键流程的中文日志打印（例如 `print("开始特征提取...")`）。
- 意图标签转换应配备清晰的明文词典映射（如 `scene_map`、`emotion_map` 等）。

### 5. 多轮时序追踪设计规范
- **固定上下文窗口**：时序追踪网络输入固定为连续 3 轮（T-2, T-1, T）的历史会话文本。
- **Git 最佳实践**：敏感机密配置（如 `.env`）及易膨胀的二进制数据缓存（如 `.npy`）**禁止提交**，使用 `.gitignore` 排除。数据资产应存为 `.json` 纯文本格式进行版本管理，并提供 `.env.example` 配置文件模板。
- **神经网络结构**：采用冻结 `MacBertEncoder` 底座将窗口内文本转成 `[Batch, 5, 768]` 维特征序列，送入单层 `TemporalSignalGRU` 提取最后一轮（T时刻）的隐状态进行时序模式分类。

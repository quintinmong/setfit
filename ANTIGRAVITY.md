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

- **下载基础模型底座（BGE-1.5 中文版）**
  ```bash
  .venv/bin/python3 download_model.py
  ```

- **运行少样本（Few-Shot）微调与二分类 Demo**
  ```bash
  .venv/bin/python3 intent_demo.py
  ```

- **训练与测试三路意图分类模型（场景、思维、情绪）**
  ```bash
  .venv/bin/python3 three_intent_server.py
  ```

- **训练与保存六路意图分类模型（场景、思维、情绪、资产信号、系统技能、信任行为）**
  ```bash
  .venv/bin/python3 six_intent_server.py
  ```

- **加载持久化模型进行快速推理测试**
  ```bash
  .venv/bin/python3 use_six_intents_model.py
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
在实例化 `SentenceTransformer` 底座或运行 PyTorch 模型时，优先进行多后端加速判断：
```python
import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

### 3. 多分类头与特征抽取设计
- 避免对相同句子重复调用 `SentenceTransformer.encode` 提取高维特征。
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

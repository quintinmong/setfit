# Repository Guidelines

## Project Structure & Module Organization
This repository is a Chinese financial-dialogue NLP system built around a shared frozen MacBERT encoder with intent, temporal, and NER heads.

- `agent_router.py` is the integrated entry point for 6-dimension intent inference, temporal GRU signals, slot filling, and LLM response routing.
- `intent_classification/` contains model download, training, demo, and saved-model inference scripts.
- `temporal_signal/` contains temporal seed data, LLM augmentation, constants, and GRU training code.
- `ner_slot/` contains NER corpus generation, training, inference, and slot-state logic.
- `models/`, `my_final_six_intents_model/`, `checkpoints/`, and zip/model artifacts hold generated model assets. Avoid committing large regenerated binaries unless explicitly needed.
- `.env.example` documents required API and model settings; keep local secrets in `.env`.

## Build, Test, and Development Commands
Use a virtual environment before running scripts:

```bash
source .venv/bin/activate        # default local environment
source .venv312/bin/activate     # Python 3.12 path used by some temporal/LLM scripts
pip install -r requirements.txt
```

Common workflows:

```bash
chmod +x run_all.sh && ./run_all.sh
python3 intent_classification/download_model.py
python3 intent_classification/six_intent_server.py
python3 intent_classification/use_six_intents_model.py
python3 temporal_signal/train_temporal_model.py
python3 ner_slot/ner_train.py
python3 ner_slot/ner_infer_and_slot_fill.py
python3 agent_router.py
```

`run_all.sh` downloads the base model, trains intent heads, trains the GRU, trains the NER head, and runs the final router smoke test.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation. Prefer explicit constants and mapping dictionaries such as `scene_map`, `emotion_map`, and `trust_map`. Keep path handling relative to `__file__` so scripts work from any current directory. Scripts that import `setfit` or `transformers` training code must apply the existing `transformers.training_args.default_logdir` monkey patch before those imports. Chinese progress logs are acceptable and already used throughout training scripts.

## Testing Guidelines
There is no formal pytest suite yet. Treat runnable scripts as smoke tests and run the narrowest relevant command after changes. For router or model-contract changes, run `python3 agent_router.py`. For NER changes, run `python3 ner_slot/ner_infer_and_slot_fill.py`; for temporal changes, run `python3 temporal_signal/train_temporal_model.py` or `python3 temporal_signal_server.py`.

## Commit & Pull Request Guidelines
Git history uses Conventional Commit-style messages such as `feat(ner): ...`, `fix(temporal_signal): ...`, `refactor(config): ...`, and `docs: ...`. Keep commits scoped and imperative. Pull requests should include a short purpose statement, changed modules, commands run, model artifacts added or regenerated, and any `.env` or API-key assumptions. Link related issues when available.

## Security & Configuration Tips
Never commit `.env` or API keys. Copy `.env.example` to `.env` for `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME`, `GRU_*`, and embedding model settings. Document any new required environment variable in `.env.example` and README-facing instructions.

## 编码行为准则

**权衡说明：** 以下准则倾向于谨慎而非速度。对于简单任务，酌情处理。

### 1. 先思考再编码

**不要假设，不要掩盖困惑，主动暴露权衡。**

动手前：
- 明确说出你的假设，不确定时先问。
- 如果存在多种理解方式，列出来——不要默默选一个。
- 如果有更简单的方案，说出来，必要时主动反驳。
- 如果有不清楚的地方，停下来，说明困惑点，然后提问。

### 2. 简单优先

**用最少的代码解决问题，不写推测性代码。**

- 不添加未被要求的功能。
- 一次性使用的代码不做抽象。
- 不做未被要求的"灵活性"或"可配置性"。
- 不为不可能发生的场景写错误处理。
- 如果写了200行而50行就能搞定，重写。

问自己："资深工程师会说这过度设计吗？"如果是，简化。

### 3. 外科手术式修改

**只动必须动的地方，只清理自己制造的乱子。**

修改已有代码时：
- 不"顺手优化"周边代码、注释或格式。
- 不重构没坏的东西。
- 保持现有风格，即使你会换一种写法。
- 发现无关的死代码，提一句——但不要删。

你的改动产生孤立代码时：
- 删除**你的改动**导致不再使用的 import/变量/函数。
- 不删除已有的死代码，除非被明确要求。

检验标准：每一行改动都应该能直接追溯到用户的需求。

### 4. 目标驱动执行

**定义成功标准，循环验证直到达成。**

把任务转化为可验证的目标：
- "加校验" → "为非法输入写测试，然后让测试通过"
- "修复 bug" → "写一个能复现 bug 的测试，然后让它通过"
- "重构 X" → "确保重构前后测试都通过"

多步骤任务先给出简要计划：
```
1. [步骤] → 验证：[检查项]
2. [步骤] → 验证：[检查项]
3. [步骤] → 验证：[检查项]
```

明确的成功标准让你可以独立循环推进；模糊的标准（"让它能用"）会导致反复确认。

### 5. 配置与默认值纪律

- 配置缺失必须立即报错（fail-fast），禁止静默回退到默认值。宁可启动失败，
  不要带着错误的默认配置跑完一整条流水线。
- 重构 = 移动代码，不是改变或固化行为。遇到原代码中存疑的行为（如默认值、
  静默吞异常、自动重试），不要顺手保留或强化它——停下来指出，问保留还是移除。
- "显然有益"的新增（兜底、缓存、兼容层）也要先提出再做。

检验标准：每个默认值都能回答"是谁明确要求的"。

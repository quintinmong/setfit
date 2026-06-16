# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language multi-dimensional intent classification system for financial banking dialogue, built on a shared frozen **MacBERT-base** encoder (`hfl/chinese-macbert-base`). The system classifies user utterances across 6 independent dimensions simultaneously and feeds into a full Agent routing pipeline that combines temporal sequence modeling (GRU) and NER slot filling to generate contextual banker responses via an LLM.

## Environment

Two virtual environments exist; prefer `.venv` (Python 3.14.5) unless doing temporal/LLM data augmentation, which uses `.venv312` (Python 3.12.13):

```bash
source .venv/bin/activate        # Python 3.14.5 (default)
source .venv312/bin/activate     # Python 3.12.13 (for temporal_signal scripts)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Common Commands

```bash
# One-click full pipeline (download → train intents → train GRU → train NER → run agent)
chmod +x run_all.sh && ./run_all.sh

# Step-by-step:
python3 intent_classification/download_model.py        # Download hfl/chinese-macbert-base via Transformers
python3 intent_classification/six_intent_server.py     # Train & save 6-head classifier
python3 intent_classification/use_six_intents_model.py # Run inference on saved model
python3 temporal_signal_server.py                      # Prototype temporal signal test

# Data augmentation (requires .env with LLM_API_KEY):
python3 temporal_signal/llm_generate_temporal_seeds.py # Generate temporal training data via LLM
python3 temporal_signal/train_temporal_model.py        # Train GRU on augmented data
python3 ner_slot/ner_generate_corpus.py                # Generate NER corpus via LLM
python3 ner_slot/ner_train.py                          # Train NER Linear + CRF heads on top of MacBERT
python3 ner_slot/ner_infer_and_slot_fill.py            # Test BIO labeling + slot state machine

# Full agent brain:
python3 agent_router.py
```

## Configuration

Copy `.env.example` to `.env` and fill in:
- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` — DeepSeek or OpenAI-compatible API
- `GRU_*` hyperparameters (input size 768, hidden 64, lr 0.005, epochs 30)
- `EMBEDDING_MODEL_ID` / `EMBEDDING_MODEL_DIR` for the MacBERT base model

## Architecture

### Core Design Pattern: Shared Encoder + Multiple Lightweight Heads

The central design extracts embeddings **once** from the frozen `MacBertEncoder` (`[CLS]`, 768-dim), then fans out to parallel classifiers — never call `.encode()` multiple times for the same input.

```
User text → MacBERT encoder → 768-dim embedding
                                              ↓
                          ┌────────────────────────────────────┐
                          │  6 Independent LogisticRegression   │
                          │  scene / thinking / emotion /       │
                          │  asset / skill / trust              │
                          └────────────────────────────────────┘
                              ↓                   ↓
                       3-turn history        GRU temporal model
                       [Batch, 5, 768]       → multi-head temporal signal
                              ↓
                       NER slot state machine (FILLED/SUGGESTED/EMPTY)
                              ↓
                       LLM prompt routing (fast/slow thinking)
```

### Key Files

| File | Role |
|------|------|
| `agent_router.py` | Main orchestrator: loads all models, runs 6-head inference + GRU + slot machine, builds LLM prompt, returns banker reply |
| `intent_classification/six_intent_server.py` | Trains all 6 classification heads and saves encoder + heads to `my_final_six_intents_model/` |
| `temporal_signal/train_temporal_model.py` | Defines `TemporalSignalGRU` (importable) and trains it on 3-turn dialogue sequences |
| `ner_slot/ner_infer_and_slot_fill.py` | BIO labeling utilities + `SlotFillingStateMachine` (3 states: FILLED/SUGGESTED/EMPTY) |

### Saved Model Artifacts (`my_final_six_intents_model/`)

- `macbert_encoder/` — frozen MacBERT tokenizer/model copy
- `classification_heads.pkl` — dict of 6 joblib-serialized LogisticRegression heads keyed: `scene`, `thinking`, `emotion`, `asset`, `skill`, `trust`
- `temporal_gru_weights.pth` — GRU state dict

### 6 Classification Dimensions

| Key | Labels |
|-----|--------|
| `scene` | 0=咨询响应, 1=情绪抱怨, 2=主动服务/关系维护 |
| `thinking` | 0=快思考, 1=慢思考 |
| `emotion` | 0=平静, 1=愤怒, 2=焦虑 |
| `asset` / `skill` / `trust` | 0=不包含, 1=包含 |

### Temporal Signal Classes (GRU output)

0=常态对话, 1=意向升温, 2=触发真拒绝, 3=异议成功化解

## Critical Code Patterns

### Monkey Patch (required in all training scripts)

Any script that imports `setfit` or `transformers` training must apply this patch **before** other imports:

```python
import transformers.training_args
if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
```

### Hardware Acceleration

```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

### Saving Multi-Head Classifiers

Pack all heads into one dict and use `joblib.dump`:

```python
joblib.dump({"scene": head_sc, "thinking": head_th, ...}, "classification_heads.pkl")
```

### Absolute Path Hardening

All scripts resolve paths relative to `__file__` to avoid working-directory issues:

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
```

## Code Style

- PEP 8
- Add key-step Chinese print logs (e.g., `print("开始特征提取...")`)
- Label mappings must use explicit dict constants (`scene_map`, `emotion_map`, etc.)
- Data assets stored as `.json`; binary/cache files (`.npy`, `.pth`) excluded from git
- NER temporal window is fixed at 3 turns (T-2, T-1, T)

import sys
import os

# ================= 猴子补丁：防止 transformers 报错 =================
import transformers.training_args

if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

# 路径加固
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from dotenv import load_dotenv, find_dotenv

from temporal_signal.temporal_constants import (
    WINDOW_SIZE, TASK_ID,
    T1_LABELS, T7_LABELS, T8_LABELS,
    pad_temporal_embeddings
)


# ==================== 模型定义 ====================

class MultiHeadTemporalGRU(nn.Module):
    """多头时序 GRU：共享 GRU 编码器，T1/T7/T8 各有独立分类头。"""

    def __init__(self, input_size=512, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head_t1 = nn.Linear(hidden_size, 3)   # T1: 3类
        self.head_t7 = nn.Linear(hidden_size, 4)   # T7: 4类
        self.head_t8 = nn.Linear(hidden_size, 3)   # T8: 3类

    def forward(self, x):
        """x: [B, WINDOW_SIZE, 512] -> (logits_t1, logits_t7, logits_t8)"""
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        h = out[:, -1, :]  # 最后时刻隐状态
        return self.head_t1(h), self.head_t7(h), self.head_t8(h)


# ==================== 数据集定义 ====================

class UnifiedTemporalDataset(Dataset):
    """统一时序数据集，每条样本包含 (embeddings [5,512], label, task_id)。"""

    def __init__(self, X, y, task_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.task_ids[idx]


# DEPRECATED: use MultiHeadTemporalGRU
TemporalSignalGRU = MultiHeadTemporalGRU


# ==================== 训练主函数 ====================

def main():
    load_dotenv(find_dotenv())

    X_DIM = int(os.getenv("GRU_INPUT_SIZE", "512"))
    HIDDEN_DIM = int(os.getenv("GRU_HIDDEN_SIZE", "64"))
    LR = float(os.getenv("GRU_LEARNING_RATE", "0.005"))
    EPOCHS = int(os.getenv("GRU_EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("GRU_BATCH_SIZE", "4"))

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("===================================================================")
    print(f"多头时序模型训练引擎启动 | 核心设备: {device}")
    print("===================================================================")

    # 1. 加载三个任务的数据文件
    T1_PATH = os.path.join(current_dir, "temporal_t1.json")
    T7_PATH = os.path.join(current_dir, "temporal_t7.json")
    T8_PATH = os.path.join(current_dir, "temporal_t8.json")

    for path in [T1_PATH, T7_PATH, T8_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到时序训练数据文件: {path}\n"
                f"请先运行: python3 temporal_signal/llm_generate_temporal_seeds.py\n"
                f"（需要在 .env 中配置 LLM_API_KEY）"
            )

    with open(T1_PATH, "r", encoding="utf-8") as f:
        t1_raw = json.load(f)
    with open(T7_PATH, "r", encoding="utf-8") as f:
        t7_raw = json.load(f)
    with open(T8_PATH, "r", encoding="utf-8") as f:
        t8_raw = json.load(f)

    print(f"\n数据加载完毕: T1={len(t1_raw)} 条, T7={len(t7_raw)} 条, T8={len(t8_raw)} 条")

    # 2. 加载 BGE 编码器
    ENCODER_PATH = os.path.join(root_dir, "my_final_six_intents_model/bge_encoder")
    BASE_MODEL_PATH = os.path.join(root_dir, "models/bge-small-zh-v1.5")

    if not os.path.exists(ENCODER_PATH):
        print(f"未检测到微调底座，回滚至基础底座: {BASE_MODEL_PATH}")
        ENCODER_PATH = BASE_MODEL_PATH
    else:
        print(f"成功检测到微调底座: {ENCODER_PATH}")

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"找不到模型底座！请确认 models/bge-small-zh-v1.5 是否下载完整。")

    print(f"正在加载语义编码底座: {ENCODER_PATH}")
    encoder = SentenceTransformer(ENCODER_PATH, device=device)

    # 3. 特征提取（使用 pad_temporal_embeddings 统一处理 WINDOW_SIZE=5）
    print("\n开始特征提取（窗口大小=5，不足则前补零）...")

    def extract_features(raw_data, task_key):
        X_list, y_list, task_id_list = [], [], []
        tid = TASK_ID[task_key]
        for item in raw_data:
            emb = pad_temporal_embeddings(encoder, item["history"])  # [5, 512]
            X_list.append(emb)
            y_list.append(item["label"])
            task_id_list.append(tid)
        return np.array(X_list), np.array(y_list), np.array(task_id_list)

    X_t1, y_t1, tid_t1 = extract_features(t1_raw, "T1")
    X_t7, y_t7, tid_t7 = extract_features(t7_raw, "T7")
    X_t8, y_t8, tid_t8 = extract_features(t8_raw, "T8")

    print(f"特征提取完毕: T1={X_t1.shape}, T7={X_t7.shape}, T8={X_t8.shape}")

    # 4. 每个任务独立 80/20 划分，再合并训练集
    X_t1_train, X_t1_test, y_t1_train, y_t1_test, tid_t1_train, tid_t1_test = train_test_split(
        X_t1, y_t1, tid_t1, test_size=0.2, random_state=42, stratify=y_t1
    )
    X_t7_train, X_t7_test, y_t7_train, y_t7_test, tid_t7_train, tid_t7_test = train_test_split(
        X_t7, y_t7, tid_t7, test_size=0.2, random_state=42, stratify=y_t7
    )
    X_t8_train, X_t8_test, y_t8_train, y_t8_test, tid_t8_train, tid_t8_test = train_test_split(
        X_t8, y_t8, tid_t8, test_size=0.2, random_state=42, stratify=y_t8
    )

    # 合并训练集
    X_train = np.concatenate([X_t1_train, X_t7_train, X_t8_train], axis=0)
    y_train = np.concatenate([y_t1_train, y_t7_train, y_t8_train], axis=0)
    tid_train = np.concatenate([tid_t1_train, tid_t7_train, tid_t8_train], axis=0)

    print(f"\n训练集大小: {len(X_train)} 条（T1={len(X_t1_train)}, T7={len(X_t7_train)}, T8={len(X_t8_train)}）")
    print(f"测试集大小: T1={len(X_t1_test)}, T7={len(X_t7_test)}, T8={len(X_t8_test)}")

    train_dataset = UnifiedTemporalDataset(X_train, y_train, tid_train)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 模型初始化
    model = MultiHeadTemporalGRU(input_size=X_DIM, hidden_size=HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. 训练循环
    print(f"\n开始训练 (Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR})...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for batch_X, batch_y, batch_task in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_task = batch_task.to(device)

            optimizer.zero_grad()
            logits_t1, logits_t7, logits_t8 = model(batch_X)

            loss = 0
            for task_id_val, logits in [
                (TASK_ID["T1"], logits_t1),
                (TASK_ID["T7"], logits_t7),
                (TASK_ID["T8"], logits_t8)
            ]:
                mask = (batch_task == task_id_val)
                if mask.any():
                    loss += criterion(logits[mask], batch_y[mask])

            if loss > 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{EPOCHS}] -> Loss: {epoch_loss:.4f}")

    # 7. 评估
    print("\n===== 测试集评估 =====")
    model.eval()

    # 分布检查
    print("\n  各任务测试集标签分布：")
    print(f"    T1 测试集分布: {dict(sorted(Counter(y_t1_test.tolist()).items()))}")
    print(f"    T7 测试集分布: {dict(sorted(Counter(y_t7_test.tolist()).items()))}")
    print(f"    T8 测试集分布: {dict(sorted(Counter(y_t8_test.tolist()).items()))}")

    def evaluate_head(X_test, y_test, tid_test, task_key, n_cls, label_map):
        tid_val = TASK_ID[task_key]
        ds = UnifiedTemporalDataset(X_test, y_test, tid_test)
        dl = DataLoader(ds, batch_size=16, shuffle=False)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for bX, by, _ in dl:
                bX = bX.to(device)
                logits_t1_b, logits_t7_b, logits_t8_b = model(bX)
                logits_map = {
                    TASK_ID["T1"]: logits_t1_b,
                    TASK_ID["T7"]: logits_t7_b,
                    TASK_ID["T8"]: logits_t8_b,
                }
                logits = logits_map[tid_val]
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(by.tolist())

        target_names = [label_map[i] for i in range(n_cls)]
        print(f"\n  [{task_key}] 分类报告:")
        print(classification_report(
            all_labels, all_preds,
            labels=list(range(n_cls)),
            target_names=target_names,
            zero_division=0
        ))

    evaluate_head(X_t1_test, y_t1_test, tid_t1_test, "T1", 3, T1_LABELS)
    evaluate_head(X_t7_test, y_t7_test, tid_t7_test, "T7", 4, T7_LABELS)
    evaluate_head(X_t8_test, y_t8_test, tid_t8_test, "T8", 3, T8_LABELS)

    # 8. 保存模型
    SAVE_DIR = os.path.join(root_dir, "my_final_six_intents_model")
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, "temporal_gru_weights.pth")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n多头时序 GRU 模型已成功保存至: '{SAVE_PATH}'")


if __name__ == "__main__":
    main()

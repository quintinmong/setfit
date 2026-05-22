import sys
import os

# ================= 🚨 猴子补丁：防止 transformers 报错 🚨 =================
import transformers.training_args

if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# 0. 加载外部配置
load_dotenv()
X_DIM = int(os.getenv("GRU_INPUT_SIZE", "512"))
HIDDEN_DIM = int(os.getenv("GRU_HIDDEN_SIZE", "64"))  # 兼容你的env参数名
LR = float(os.getenv("GRU_LEARNING_RATE", "0.005"))
EPOCHS = int(os.getenv("GRU_EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("GRU_BATCH_SIZE", "4"))

# 1. 硬件加速检测
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("===================================================================")
print(f"🚀 时序模型正式训练引擎启动 | 核心设备: {device}")
print("===================================================================")

# 2. 加载生成的 LLM 变体语料 (优先读取 JSON)
DATA_PATH_JSON = "./temporal_signal/temporal_signal_llm_augmented.json"
DATA_PATH_NPY = "./temporal_signal/temporal_signal_llm_augmented.npy"

if not os.path.exists(DATA_PATH_JSON) and not os.path.exists(DATA_PATH_NPY):
    # 兼容处理：如果执行路径在子目录下，尝试直接读取
    DATA_PATH_JSON = "./temporal_signal_llm_augmented.json"
    DATA_PATH_NPY = "./temporal_signal_llm_augmented.npy"

import json

if os.path.exists(DATA_PATH_JSON):
    print(f"📂 正在读取大模型冷启动增强语料 (JSON 格式): {DATA_PATH_JSON}")
    with open(DATA_PATH_JSON, "r", encoding="utf-8") as f:
        seed_dialogues = json.load(f)
elif os.path.exists(DATA_PATH_NPY):
    print(f"📂 正在读取大模型冷启动增强语料 (兼容 NPY 格式): {DATA_PATH_NPY}")
    seed_dialogues = np.load(DATA_PATH_NPY, allow_pickle=True).tolist()
else:
    raise FileNotFoundError(f"❌ 找不到冷启动增强语料数据文件！请确认是否生成或存放在 temporal_signal 目录下。")
# 3. 载入固化的共享 BGE 底座 (做路径加固，防止 PyCharm 工作目录切换导致找不到模型)
ENCODER_PATH = "./my_final_six_intents_model/bge_encoder"
BASE_MODEL_PATH = "./models/bge-small-zh-v1.5"

# 如果发现路径不在当前目录，自动往上一级（项目根目录）去寻找
if not os.path.exists(ENCODER_PATH) and os.path.exists("../my_final_six_intents_model/bge_encoder"):
    ENCODER_PATH = "../my_final_six_intents_model/bge_encoder"
    BASE_MODEL_PATH = "../models/bge-small-zh-v1.5"

if not os.path.exists(ENCODER_PATH):
    print(f"⚠️ 未检测到微调底座，自动回滚至本地基础 1.5 底座: {BASE_MODEL_PATH}")
    ENCODER_PATH = BASE_MODEL_PATH
else:
    print(f"🧠 成功检测到微调底座: {ENCODER_PATH}")

# 此时如果连上一级也找不到，报出友好提示
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"❌ 找不到模型底座！请确认 models/bge-small-zh-v1.5 是否下载完整。")

print(f"🧠 正在同步语义嵌入底座特征抽取器: {ENCODER_PATH}")
encoder = SentenceTransformer(ENCODER_PATH, device=device)

# 4. 全量特征一键转换 (重用底座，零重复计算)
print("\n⚡️ 正在重用特征底座将时序文本序列转换为高维张量...")
X_temporal_list = []
y_list = []

for idx, dlg in enumerate(seed_dialogues):
    # 抽取3轮历史文本的高维语义特征 [3, 512]
    embeddings = encoder.encode(dlg["history"], show_progress_bar=False)
    X_temporal_list.append(embeddings)
    y_list.append(dlg["label"])

X_temporal = np.array(X_temporal_list)  # [240, 3, 512]
y_temporal = np.array(y_list)  # [240]
print(f"🏆 特征矩阵组装完毕! 数据形态: {X_temporal.shape}")


# 5. 定义经典时序 GRU 神经网络
class TemporalSignalGRU(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, num_layers=1, num_classes=4):
        super(TemporalSignalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # 截取 T 时刻最后一轮的隐状态输出进行演变模式决策
        out = self.fc(out[:, -1, :])
        return out


# 6. 数据流水线装载
class DialogueSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = DialogueSequenceDataset(X_temporal, y_temporal)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 7. 神经网络初始化与配置
model = TemporalSignalGRU(input_size=X_DIM, hidden_size=HIDDEN_DIM, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 8. 周期拟合迭代
print(f"\n🪐 神经网络进入反向传播拟合迭代 (总 Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    if epoch % 5 == 0 or epoch == 1:
        acc = (correct / total) * 100
        print(f"  ├── Epoch [{epoch}/{EPOCHS}] -> 交叉熵 Loss: {epoch_loss:.4f} | 训练集准确率: {acc:.2f}%")

# 9. 模型持久化固化
SAVE_DIR = "./my_final_six_intents_model"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "temporal_gru_weights.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n🏆 恭喜！时序演变轨迹分类器模型已成功训练并固化至本地：'{MODEL_SAVE_PATH}'")
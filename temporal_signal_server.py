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

# 1. 硬件加速检测
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前训练与推理设备设定为: {device}")

# 2. 复用已有的持久化六分类 BGE 底座 (避免重复计算)
ENCODER_PATH = "./my_final_six_intents_model/bge_encoder"
if not os.path.exists(ENCODER_PATH):
    print(f"⚠️ 未检测到微调底座，回滚使用基础底座路径...")
    ENCODER_PATH = "./models/bge-small-zh-v1.5"

print(f"正在载入共享 Embedding 底座: {ENCODER_PATH}")
encoder = SentenceTransformer(ENCODER_PATH, device=device)

# 3. 构建高密度多轮冷启动种子数据集 (覆盖方案中 P0 级核心时序信号)
# 设定固定的上下文窗口为 3 轮会话 (T-2, T-1, T)
# 信号标签定义：
# 0: 意向平稳/常规咨询
# 1: 意向升温/购买信号触发 (T1信号)
# 2: 触发真拒绝/彻底失去兴趣 (T7信号)
# 3: 情绪明显好转/异议被成功化解 (T8信号)
seed_dialogues = [
    # === 场景：意向升温 (标签 1) ===
    {"history": ["你们理财安全吗？", "中低风险保本吗？", "那这个安心存怎么申购？"], "label": 1},
    {"history": ["我想查存款利率", "感觉定存利息有点低啊", "工程款刚下来，帮我做个资产配置规划？"], "label": 1},
    {"history": ["手机银行在哪里买基金", "纯债基金波动大吗", "计划书发我研究一下，下周去面签"], "label": 1},

    # === 场景：真拒绝/意向降温 (标签 2) ===
    {"history": ["天天给我推降息提醒", "别天天发微信推销了，烦不烦", "我要销户，把钱全转走！"], "label": 2},
    {"history": ["买的稳健型绿成这样", "亏了这么多你们找借口", "别说了，以后不会再买你们的产品了"], "label": 2},
    {"history": ["有专属回馈活动吗", "怎么还要录音录像这么麻烦", "算了，不弄了，老太婆操作不来"], "label": 2},

    # === 场景：异议化解/情绪恢复 (标签 3) ===
    {"history": ["我看网上说要金融危机了", "你们民生银行靠得住吗", "听你分析完心里踏实多了，谢谢小张"], "label": 3},
    {"history": ["刚买就跌破净值，赔钱！", "你们产品经理水平不行", "行吧，那我长期持有再观察看看"], "label": 3},
    {"history": ["为什么别人家APP操作傻瓜", "你们的界面太绕了", "不过小张你的服务态度确实没得说"], "label": 3},

    # === 场景：常规咨询/平稳波动 (标签 0) ===
    {"history": ["早啊，有什么新产品吗", "今天发行的专享理财编码多少", "收到，谢谢经理"], "label": 0},
    {"history": ["外币存款一年期利息多少", "美元现在是什么利率", "那我过两天再去网点办"], "label": 0}
]

# 4. 特征抽取工程：将文本序列一键转化为时序张量
print("\n核心优化：开始重用底座，进行多轮对话文本特征提取...")
X_temporal_list = []
y_list = []

for dlg in seed_dialogues:
    # 统一将一整段多轮历史打包进行一次 encode
    embeddings = encoder.encode(dlg["history"], show_progress_bar=False)  # 形状: [3, 512]
    X_temporal_list.append(embeddings)
    y_list.append(dlg["label"])

X_temporal = np.array(X_temporal_list)  # 形状: [样本数, 步长(3), 特征维度(512)]
y_temporal = np.array(y_list)

print(f"特征提取完毕！时序特征矩阵形状: {X_temporal.shape}, 标签形状: {y_temporal.shape}")


# 5. 定义轻量级 PyTorch GRU 时序分类模型
class TemporalSignalGRU(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, num_layers=1, num_classes=4):
        super(TemporalSignalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True 保证输入维度为 [batch, seq_len, feature]
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 隐状态初始化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # GRU 前向传播
        out, _ = self.gru(x, h0)
        # 仅抽取最后一轮 (T时刻) 的隐状态进行模式分类识别
        out = self.fc(out[:, -1, :])
        return out


# 6. 数据装载
class DialogueSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = DialogueSequenceDataset(X_temporal, y_temporal)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 7. 模型初始化、定义损失函数与优化器
model = TemporalSignalGRU(input_size=512, hidden_size=64, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 8. 快速模型拟合
print("\n🪐 开始时序 GRU 分类器轻量拟合训练...")
model.train()
for epoch in range(1, 31):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 5 == 0:
        print(f"  Epoch [{epoch}/30] - 交叉熵损失(Loss): {epoch_loss:.4f}")

print("🏆 时序信号追踪模型训练完成！")

# 9. 持久化保存时序网络权重
SAVE_MODEL_PATH = "./my_final_six_intents_model/temporal_gru_model.pth"
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"🔥 时序模型已固化至: '{SAVE_MODEL_PATH}'")

# ==================== 🪐 现场多轮动态时序追踪测试 ====================
print("\n--- 开始时序演变轨迹现场推理集成测试 ---")
model.eval()

# 模拟全新客户的多轮真实对话轨迹
test_sessions = [
    {
        "desc": "测试案例 A (表现出极强的购买与规划升温意向)",
        "history": ["有高收益产品再找我", "我卡里还有二十万活期", "你看看帮我怎么规划能抗通胀？"]
    },
    {
        "desc": "测试案例 B (客户遭遇亏损后情绪被话术成功安抚化解)",
        "history": ["本金亏了你们赔吗？！", "我看别家收益都挺高", "听你讲完长期配置的道理，好像也是这么回事"]
    }
]

signal_map = {
    0: "⚡️ 意向平稳 (常规对话循环)",
    1: "📈 意向升温 (检测到 T1 购买倾向信号，可主动进行资产配置推荐)",
    2: "🛑 触发真拒绝 (检测到 T7 流失风险，建议转接人工或改变策略)",
    3: "🤝 异议成功化解 (检测到 T8 客户心智恢复，信任度重新建立)"
}

with torch.no_grad():
    for session in test_sessions:
        print(f"\n【💥 追踪流】: {session['desc']}")
        # 1. 提取当前会话历史的特征序列
        feat = encoder.encode(session["history"], show_progress_bar=False)
        # 2. 转换为 Tensor 并升维匹配 Batch [1, seq_len, 512]
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        # 3. 推理预测
        pred_logits = model(feat_tensor)
        pred_label = torch.argmax(pred_logits, dim=1).item()

        # 4. 打印时序研判结果
        print(f"  ├── 轮次1: \"{session['history'][0]}\"")
        print(f"  ├── 轮次2: \"{session['history'][1]}\"")
        print(f"  ├── 轮次3: \"{session['history'][2]}\"")
        print(f"  └── 📊 时序模型最终研判模式: {signal_map[pred_label]}")
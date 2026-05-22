import sys

# ================= 🚨 猴子补丁：防止 transformers 报错 🚨 =================
import transformers.training_args

if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 严格使用方案指定的本地 1.5 底座
MODEL_PATH = "./models/bge-small-zh-v1.5"

print("正在初始化共享底座 SentenceTransformer...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = SentenceTransformer(MODEL_PATH, device=device)

# 2. 纯手工打造 30 条高密度平衡数据集（严格对齐方案定义）
train_data = [
    # === 场景：咨询响应 (0) ===
    {"text": "请问手机银行在哪里可以买理财？", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "我想看看民生银行最新的定期存款利率", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "这个理财产品的申购入口在哪里", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "找不到理财购买页面了，帮我看一下", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "请问这个新客理财的募集期到什么时候截止？", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "手机上怎么操作提前支取大额存单啊？", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "小张，我想问下这款中低风险的产品保本吗？", "scene": 0, "thinking": 0, "emotion": 2},  # 咨询,快,焦虑
    {"text": "你们那个安心存产品，每天几点开始放额度？", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "合同里写的这个管理费和托管费是怎么扣的？", "scene": 0, "thinking": 0, "emotion": 0},
    {"text": "怎么看我买的基金今天收益是多少？", "scene": 0, "thinking": 0, "emotion": 0},

    # === 场景：情绪抱怨 (1) ===
    {"text": "现在的理财收益也太低了吧，真差劲", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒
    {"text": "怎么天天跌啊，感觉比以前差多了！", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒
    {"text": "你们这个APP太难用了，经常卡顿闪退", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒
    {"text": "气死我了，今天怎么又亏了这么多钱", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒
    {"text": "垃圾银行，体验太差了，我要销户！", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒
    {"text": "收益才3.2%，感觉比以前差多了", "scene": 1, "thinking": 1, "emotion": 0},  # 抱怨,慢(宏观分析信号),平静
    {"text": "招行收益好像更高，有点动摇了", "scene": 1, "thinking": 1, "emotion": 2},  # 抱怨,慢(同业对比),焦虑
    {"text": "我看网上说现在要发生金融危机了，理财还安全吗？", "scene": 1, "thinking": 1, "emotion": 2},  # 抱怨,慢,焦虑
    {"text": "隔壁建设银行天天给我推4.0的利息，你们怎么这么低？", "scene": 1, "thinking": 1, "emotion": 2},
    # 抱怨,慢(同业对比),焦虑
    {"text": "买的时候说是稳健型，结果绿成这样，你们骗人吧！", "scene": 1, "thinking": 0, "emotion": 1},  # 抱怨,快,愤怒

    # === 场景：主动服务/关系维护 (2) ===
    {"text": "谢谢小张经理，你的建议很专业", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "早啊，今天有什么新产品推荐吗？", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "节日祝福收到了，也祝你节日快乐", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "行，那我下周有空去网点找你面签", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "最近确实比较忙，等我忙完这段时间再看吧", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "你们民生银行的服务态度确实没得说，挺好的", "scene": 2, "thinking": 0, "emotion": 0},
    {"text": "最近股市这么震荡，我总觉得心里不踏实啊", "scene": 2, "thinking": 1, "emotion": 2},  # 关系维护,慢(需要慢思考安抚),焦虑
    {"text": "手头刚下来一笔工程款，你帮我做个全面的资产配置规划？", "scene": 2, "thinking": 1, "emotion": 0},
    # 主动服务,慢(深度规划),平静
    {"text": "我想给我家孩子存笔教育金，不知道怎么规划好", "scene": 2, "thinking": 1, "emotion": 0},  # 主动服务,慢,平静
    {"text": "小张，听说最近国债利率又降了？对我们有影响吗？", "scene": 2, "thinking": 1, "emotion": 2}  # 关系维护,慢(宏观分析),焦虑
]

# 解包转换为向量底座需要的格式
train_texts = [item["text"] for item in train_data]
labels_scene = [item["scene"] for item in train_data]
labels_thinking = [item["thinking"] for item in train_data]
labels_emotion = [item["emotion"] for item in train_data]

# 3. 核心优化：全量文本只通过底座 encode 一次！
print(f"核心优化：底座统一进行向量化计算（共 {len(train_texts)} 条样本，仅执行一次）...")
X_train = encoder.encode(train_texts, show_progress_bar=False)

# 4. 训练 3 个独立的轻量分类头（解耦运行）
print("正在并行训练 3 个独立的轻量分类头...")
head_scene = LogisticRegression(class_weight='balanced').fit(X_train, labels_scene)
head_thinking = LogisticRegression(class_weight='balanced').fit(X_train, labels_thinking)
head_emotion = LogisticRegression(class_weight='balanced').fit(X_train, labels_emotion)
print("所有分类头训练完成！")


# 5. 推理服务函数
def predict_multi_intent(user_input: str):
    # 步骤 A: 底座单次推理
    features = encoder.encode([user_input], show_progress_bar=False)

    # 步骤 B: 3个轻量分类头同时前向推理
    pred_scene = head_scene.predict(features)[0]
    pred_thinking = head_thinking.predict(features)[0]
    pred_emotion = head_emotion.predict(features)[0]

    scene_map = {0: "咨询响应", 1: "情绪抱怨", 2: "主动服务/关系维护"}
    thinking_map = {0: "⚡️快思考（常态服务）", 1: "🧠慢思考（触发宏观分析、专家案例或同业对比）"}
    emotion_map = {0: "😊平静", 1: "😡愤怒", 2: "😰焦虑"}

    return {
        "场景维度": scene_map[pred_scene],
        "思维维度": thinking_map[pred_thinking],
        "情绪维度": emotion_map[pred_emotion]
    }


# 6. 极限压力测试（使用代码和数据中均不存在的全新测试句子）
print("\n--- 🪐 开始多维意图并行推理测试 🪐 ---")

test_cases = [
    "请问手机上怎么查我的存款利息啊？",
    "招商银行那个天天盈看起来比你们好多了，我想转过去。",  # 同业对比 -> 慢思考, 焦虑
    "我靠，刚买就跌破净值了，你们到底管不管，赔钱！"  # 严重抱怨 -> 快思考, 愤怒
]

for text in test_cases:
    print(f"\n【💥 收到客户发言】: \"{text}\"")
    result = predict_multi_intent(text)
    for dim, val in result.items():
        print(f"  └── {dim}: {val}")
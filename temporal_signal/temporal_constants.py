import sys
import os

# 路径加固（当此文件直接运行时使用）
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    sys.path.insert(0, root_dir)

import numpy as np
import torch

# ======================== 共享常量 ========================

WINDOW_SIZE = 5
EMBED_DIM = 512

# T1：意向走势（3类）
T1_LABELS = {0: "平稳", 1: "升温", 2: "降温"}

# T7：拒绝类型（4类）
T7_LABELS = {0: "习惯推脱", 1: "需要理由", 2: "真无需求", 3: "谈判策略"}

# T8：化解程度（3类）
T8_LABELS = {0: "深度化解", 1: "表面化解", 2: "未化解"}

# 统一数据集 task_id 映射
TASK_ID = {"T1": 0, "T7": 1, "T8": 2}

# 6维意图分类头名称
INTENT_HEAD_NAMES = [
    "scene", "sub_type", "emotion_label",
    "info_source", "skill_trigger", "intent_label"
]

# 6维意图分类标签映射（整数 → 字符串）
INTENT_LABEL_MAPS = {
    "scene": {
        0: "咨询响应",
        1: "情绪抱怨",
        2: "主动服务/关系维护"
    },
    "sub_type": {
        0: "操作类咨询",
        1: "产品/利率咨询",
        2: "宏观/深度分析"
    },
    "emotion_label": {
        0: "平静",
        1: "愤怒",
        2: "焦虑",
        3: "不信任"
    },
    "info_source": {
        0: "无",
        1: "竞品/同业",
        2: "网络/媒体",
        3: "自身经验"
    },
    "skill_trigger": {
        0: "无",
        1: "SK-COMP",
        2: "SK-EMPATHY",
        3: "SK-EDUCATE",
        4: "SK-OPERATE"
    },
    "intent_label": {
        0: "普通咨询",
        1: "异议-收益",
        2: "异议-安全",
        3: "促单-高意向",
        4: "流失风险"
    },
}


# ======================== 特征工具函数 ========================

def pad_temporal_embeddings(encoder, history):
    """截取最近 WINDOW_SIZE 轮，前补零，返回 np.ndarray [5, 512]。供训练集构建用。"""
    window = list(history)[-WINDOW_SIZE:]
    embeddings = encoder.encode(window, show_progress_bar=False)
    if len(embeddings) < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - len(embeddings), EMBED_DIM))
        embeddings = np.vstack([pad, embeddings])
    return embeddings  # [5, 512]


def prepare_temporal_features(encoder, history, device="cpu"):
    """截取最近 WINDOW_SIZE 轮，前补零，返回 tensor [1, 5, 512]。供推理用。"""
    embeddings = pad_temporal_embeddings(encoder, history)
    return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)

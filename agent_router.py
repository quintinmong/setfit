import os
import sys
import torch
import joblib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

# 引入写好的业务模块（使用动态绝对路径加固过的模块）
from ner_slot.ner_infer_and_slot_fill import SlotFillingStateMachine, NERInferencer

# 引入新版多头 GRU 模型和时序常量
from temporal_signal.train_temporal_model import MultiHeadTemporalGRU
from temporal_signal.temporal_constants import (
    prepare_temporal_features,
    T1_LABELS, T7_LABELS, T8_LABELS,
    INTENT_LABEL_MAPS
)

# 0. 载入配置与硬件加速 (自动向上回溯寻找根目录的 .env)
load_dotenv(find_dotenv())
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("===================================================================")
print(f"银行客户经理数字分身 Agent 终极全功能大脑启动... | 核心设备: {device}")
print("===================================================================\n")

# 1. 统一加载模型（锁定在项目根目录下）
ENCODER_PATH = "./my_final_six_intents_model/bge_encoder"
encoder = SentenceTransformer(ENCODER_PATH, device=device)

# A. 载入全量 6 意图机器学习分类头（v2 格式，带版本校验）
_artifact = joblib.load("./my_final_six_intents_model/classification_heads.pkl")
if _artifact.get("version") == 2:
    heads = _artifact["heads"]
else:
    heads = _artifact  # 向后兼容旧格式

# B. 载入多头时序 GRU 网络
temporal_model = MultiHeadTemporalGRU().to(device)
temporal_model.load_state_dict(
    torch.load("./my_final_six_intents_model/temporal_gru_weights.pth", map_location=device)
)
temporal_model.eval()

# C. NER 推理器（条件触发）
NER_HEAD_PATH = "./my_final_six_intents_model/ner_head_weights.pth"
if os.path.exists(NER_HEAD_PATH):
    ner_inferencer = NERInferencer(ENCODER_PATH, NER_HEAD_PATH, device)
    print("NER 推理器加载成功")
else:
    ner_inferencer = None
    print("NER 模型未找到，请先运行 python3 ner_slot/ner_train.py")

# D. 大模型初始化
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

# NER 条件触发词表（命中任意一词才启动 NER 推理）
NER_TRIGGER_WORDS = [
    "明天", "后天", "下周", "下月", "到期", "忙完",        # TIME
    "等我", "等资金", "等工程款", "等家人", "等降准",        # WAIT_COND
    "说明书", "合同", "计划书", "大额存单", "年金", "报告",  # MATERIAL
    "老婆", "老伴", "父母", "孩子", "家人", "商量",          # RELATION_PERSON
]


def should_trigger_ner(text: str) -> bool:
    return any(w in text for w in NER_TRIGGER_WORDS)


# 3. 核心总路由引擎
def run_agent_brain(dialogue_history):
    current_input = dialogue_history[-1]

    # === Step 1: 底座单次提取当前轮向量 ===
    feat_single = encoder.encode([current_input], show_progress_bar=False)

    # === Step 2: 调用全量 6 路分类头推理（新维度命名）===
    pred_scene = heads["scene"].predict(feat_single)[0]
    pred_subtype = heads["sub_type"].predict(feat_single)[0]
    pred_emotion = heads["emotion_label"].predict(feat_single)[0]
    pred_info = heads["info_source"].predict(feat_single)[0]
    pred_skill = heads["skill_trigger"].predict(feat_single)[0]
    pred_intent = heads["intent_label"].predict(feat_single)[0]

    # === Step 3: 多轮时序 GRU 推理（三路输出）===
    feat_tensor = prepare_temporal_features(encoder, dialogue_history, device)
    with torch.no_grad():
        logits_t1, logits_t7, logits_t8 = temporal_model(feat_tensor)
        pred_t1 = torch.argmax(logits_t1, dim=1).item()
        pred_t7 = torch.argmax(logits_t7, dim=1).item()
        pred_t8 = torch.argmax(logits_t8, dim=1).item()

    # === Step 4: NER 条件触发 + 槽位状态机驱动 ===
    entities = []
    if ner_inferencer and should_trigger_ner(current_input):
        entities = ner_inferencer.predict(current_input)
    sf_machine = SlotFillingStateMachine()
    if entities:
        sf_machine.inject_entities(entities)
    slot_decision = sf_machine.emit_downstream_decision()

    # === 感知层6维看板打印 ===
    scene_str = INTENT_LABEL_MAPS["scene"][pred_scene]
    subtype_str = INTENT_LABEL_MAPS["sub_type"][pred_subtype]
    emotion_str = INTENT_LABEL_MAPS["emotion_label"][pred_emotion]
    info_str = INTENT_LABEL_MAPS["info_source"][pred_info]
    skill_str = INTENT_LABEL_MAPS["skill_trigger"][pred_skill]
    intent_str = INTENT_LABEL_MAPS["intent_label"][pred_intent]
    t1_str = T1_LABELS[pred_t1]
    t7_str = T7_LABELS[pred_t7]
    t8_str = T8_LABELS[pred_t8]

    print("-------------------------------------------------------------------")
    print(f"【感知层 6 维全量研判】 用户最新发言: \"{current_input}\"")
    print(f"  ├── 场景=[{scene_str}] | 子类型=[{subtype_str}] | 情绪=[{emotion_str}]")
    print(f"  ├── 信息来源=[{info_str}] | 技能触发=[{skill_str}] | 意图=[{intent_str}]")
    print(f"  ├── 时序: T1意向走势=[{t1_str}] | T7拒绝类型=[{t7_str}] | T8化解程度=[{t8_str}]")
    print(f"  └── 槽位决策: {slot_decision}")
    print("-------------------------------------------------------------------")

    # === Step 5: 叙事化状态提示词（全面整合 6 维和时序）===
    narrative_context = (
        f"当前对话处于【{scene_str}】场景，子类型为【{subtype_str}】，客户情绪表现为【{emotion_str}】。"
        f"信息来源为【{info_str}】，当前推荐技能为【{skill_str}】，客户意图识别为【{intent_str}】。"
        f"时序监测：意向走势为【{t1_str}】，拒绝类型为【{t7_str}】，异议化解程度为【{t8_str}】。"
        f"当前的槽位流水执行建议是：{slot_decision}。"
    )

    # 慢思考条件：宏观/深度分析 OR 促单-高意向
    pred_slow = (pred_subtype == 2) or (pred_intent == 3)

    if pred_slow:
        system_prompt = f"""你是一名极具专家心智的银行理财业务主管（小张经理）。
当前对话状态的客观叙事约束：
{narrative_context}

请严格遵守叙事化表达规范：
1. 绝对不要使用任何格式化的机器列表或硬编码条目回答。
2. 你的话术必须内化当前的 6 维状态信号、时序心智和槽位建议。
3. 请在正式输出话术前，先输出一行 <Thinking> 标签，在里面展现你深度的慢思考过程（深度剖析客户的宏观焦虑或同业对比心态），随后再输出最终去AI感的体贴话术。
"""
    else:
        system_prompt = f"""你是一名高效、口语化的银行数字分身助手。
当前对话状态：{narrative_context}
请直接用一段极其自然、简练的口语化话术回应客户。禁止任何格式包裹，控制在60字以内。"""

    # === Step 6: 调度大模型 ===
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_input}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    simulated_history = [
        "听说你们民生银行最近收益不给力啊？",
        "隔壁建设银行天天给我推4.0的利息，你们怎么这么低？",
        "等我股市里的资金到账了，我再和我老婆商量一下大额存单的事。"
    ]

    print("开始推演【6维意图 + 多轮时序(T1/T7/T8) + NER槽位】全闭环总流向...")
    final_reply = run_agent_brain(simulated_history)

    print("\n【大模型最终话术输出（基于6维全状态内化）】:")
    print(final_reply)

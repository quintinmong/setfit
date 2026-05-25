import os
import json
import random
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv  # 引入外部环境变量加载器及寻址器

# 0. 核心安全加固：使用 find_dotenv() 自动向上级回溯寻找项目根目录下的 .env 配置文件
load_dotenv(find_dotenv())

# 从配置文件或系统环境变量中读取参数（如果配置文件缺失，则使用硬编码作为健壮性兜底）
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
TARGET_COUNT = int(os.getenv("TARGET_COUNT_PER_LABEL", "60"))

if not API_KEY:
    raise ValueError("❌ 启动失败：未能在 .env 配置文件或环境变量中检测到 'LLM_API_KEY'，请检查你的配置！")

print("===================================================================")
print("🚀 成功载入外部配置文件 [.env]")
print(f"  ├── 模型基座 URL: {BASE_URL}")
print(f"  ├── 选定模型实例: {MODEL_NAME}")
print(f"  └── 每类时序目标: {TARGET_COUNT} 条")
print("===================================================================\n")

# 1. 初始化大模型客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 2. 方案定义的 4 类核心时序轨迹模板 (保持不变)
TRACK_DEFINITIONS = {
    1: {
        "name": "意向升温检测 (T1信号)",
        "guide": "客户起初表现出普遍的随口咨询或对风险的隐性担忧，在第2轮表现出对某类产品细节的关注或资产空置信号，在第3轮明确触发购买倾向、要求做全面资产配置规划或要计划书。"
    },
    2: {
        "name": "真拒绝/兴趣衰减检测 (T7信号)",
        "guide": "客户起初对频繁提醒表达不满或抱怨产品亏损，第2轮展现出高度的不耐烦、愤怒或操作嫌麻烦，第3轮明确表达‘销户’、‘全额转走’、‘以后绝不再买’等彻底流失的决绝态度。"
    },
    3: {
        "name": "异议化解/情绪恢复速率 (T8信号)",
        "guide": "客户起初因为市场跌破净值或听信网络爆雷传言而极度焦虑、不信任并向经理发难，第2轮持续要求解释或对比同业，第3轮在被专业分析和体贴服务安抚后，情绪明显好转，表达‘那我就长期持有看看’、‘听你安排’等信任恢复信号。"
    },
    0: {
        "name": "常规业务咨询 (平稳常态)",
        "guide": "客户全程情绪平静，连续3轮都在进行常规的、无明显情绪和意向波动的银行业务咨询（如问外币利率、改限额流程、开存款证明等）。"
    }
}

SYSTEM_PROMPT = """你是一名资深的银行营销数据专家。你需要为‘客户经理数字分身话术Agent’生成供机器学习模型训练用的时序冷启动种子语料。

格式要求：
你必须严格输出一个 JSON 对象，其中包含一个 "samples" 列表，列表内包含指定数量的对象。每个对象必须拥有以下两个字段，不得包含任何其他多余的包裹文本或 markdown 标签：
- "history": 包含恰好 3 个字符串的数组，代表同一个客户连续说出的 3 句话（轮次1 -> 轮次2 -> 轮次3）。
- "label": 整数，严格对应指定的意向演变轨迹。

返回格式示例：
{
  "samples": [
    {
      "history": ["第一句", "第二句", "第三句"],
      "label": 1
    }
  ]
}

语言风格：
高度拟真。口语化，包含网银用户常见的碎碎念、抱怨、催促、特定术语（如：安心存、天天盈、面签、纯债基金、年金险、录音录像、跌破净值）。
"""


def generate_batch_by_llm(label, count_to_generate):
    track = TRACK_DEFINITIONS[label]
    user_prompt = f"请严格生成 {count_to_generate} 条属于【{track['name']}】轨迹的多轮对话样本。\n演变要求：{track['guide']}\n返回格式示例：\n{{\"samples\": [{{\"history\": [\"第一句\", \"第二句\", \"第三句\"], \"label\": {label}}}]}}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,  # 动态读取配置
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
            timeout=60.0  # 添加60秒超时保护
        )

        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)

        # 优先读取规范的 samples 键值
        if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
            return data["samples"]

        # 健壮性兜底兼容旧格式
        if isinstance(data, dict):
            for key in ["samples", "data", "dataset", "list"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            if len(data.keys()) == 1 and isinstance(list(data.values())[0], list):
                return list(data.values())[0]
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  ❌ 调用 LLM 批次生成失败: {e}", flush=True)
        return []


# ==================== 🪐 循环调用与动态清洗 ====================

final_llm_dataset = []
target_plan = {1: TARGET_COUNT, 2: TARGET_COUNT, 3: TARGET_COUNT, 0: TARGET_COUNT}

for lbl, target_count in target_plan.items():
    print(f"\n🎬 开始调度 LLM 攻坚生成标签 {lbl} ({TRACK_DEFINITIONS[lbl]['name']})...", flush=True)
    current_pool = []
    retry_count = 0

    while len(current_pool) < target_count and retry_count < 5:
        needed = target_count - len(current_pool)
        print(f"  --> 正在向 LLM 请求 {needed} 条高质量样本...", flush=True)

        batch = generate_batch_by_llm(lbl, min(needed, 10))  # 降低单次批量生成数到 10，防止超时且提高生成质量

        valid_count = 0
        for sample in batch:
            if "history" in sample and len(sample["history"]) == 3 and sample.get("label") == lbl:
                current_pool.append(sample)
                valid_count += 1

        print(f"  --> 本批次大模型吐出 {len(batch)} 条，通过时序长度校验: {valid_count} 条", flush=True)
        if valid_count == 0:
            retry_count += 1
        time.sleep(1)

    final_llm_dataset.extend(current_pool[:target_count])

random.shuffle(final_llm_dataset)

# 获取当前脚本所在目录作为绝对路径基准
current_dir = os.path.dirname(os.path.abspath(__file__))

# 最佳实践：保存为 .json（适合 Git 追踪）和 .npy（兼容原有逻辑），但 .npy 会被 .gitignore 忽略
json_output_file = os.path.join(current_dir, "temporal_signal_llm_augmented.json")
npy_output_file = os.path.join(current_dir, "temporal_signal_llm_augmented.npy")

with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(final_llm_dataset, f, ensure_ascii=False, indent=2)
np.save(npy_output_file, final_llm_dataset)

print(f"\n🏆 外部配置驱动的 LLM 深度冷启动数据增强完成！共收集 {len(final_llm_dataset)} 条高拟真轨迹语料。", flush=True)
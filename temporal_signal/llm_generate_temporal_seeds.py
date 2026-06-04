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

# 获取当前脚本所在目录作为绝对路径基准
current_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== 旧版4分类轨迹定义（保持不变）====================

# 2. 方案定义的 4 类核心时序轨迹模板 (保持不变)
TRACK_DEFINITIONS = {
    1: {
        "name": "意向升温检测 (T1信号)",
        "guide": "客户起初表现出普遍的随口咨询或对风险的隐性担忧，在第2轮表现出对某类产品细节的关注或资产空置信号，在第3轮明确触发购买倾向、要求做全面资产配置规划或要计划书。"
    },
    2: {
        "name": "真拒绝/兴趣衰减检测 (T7信号)",
        "guide": "客户起初对频繁提醒表达不满或抱怨产品亏损，第2轮展现出高度的不耐烦、愤怒或操作嫌麻烦，第3轮明确表达'销户'、'全额转走'、'以后绝不再买'等彻底流失的决绝态度。"
    },
    3: {
        "name": "异议化解/情绪恢复速率 (T8信号)",
        "guide": "客户起初因为市场跌破净值或听信网络爆雷传言而极度焦虑、不信任并向经理发难，第2轮持续要求解释或对比同业，第3轮在被专业分析和体贴服务安抚后，情绪明显好转，表达'那我就长期持有看看'、'听你安排'等信任恢复信号。"
    },
    0: {
        "name": "常规业务咨询 (平稳常态)",
        "guide": "客户全程情绪平静，连续3轮都在进行常规的、无明显情绪和意向波动的银行业务咨询（如问外币利率、改限额流程、开存款证明等）。"
    }
}

SYSTEM_PROMPT = """你是一名资深的银行营销数据专家。你需要为'客户经理数字分身话术Agent'生成供机器学习模型训练用的时序冷启动种子语料。

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


def _get_client():
    """延迟初始化大模型客户端（避免导入时立即报错）。"""
    if not API_KEY:
        raise ValueError("❌ 启动失败：未能在 .env 配置文件或环境变量中检测到 'LLM_API_KEY'，请检查你的配置！")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_batch_by_llm(label, count_to_generate, client=None, system_prompt=None, track_defs=None):
    """通用批次生成函数，支持传入自定义 system_prompt 和 track_defs。"""
    if client is None:
        client = _get_client()
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if track_defs is None:
        track_defs = TRACK_DEFINITIONS

    track = track_defs[label]
    user_prompt = (
        f"请严格生成 {count_to_generate} 条属于【{track['name']}】轨迹的多轮对话样本。\n"
        f"演变要求：{track['guide']}\n"
        f"返回格式示例：\n"
        f"{{\"samples\": [{{\"history\": [\"第一句\", \"第二句\", \"第三句\"], \"label\": {label}}}]}}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
            timeout=60.0
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


def _generate_task_dataset(task_name, track_defs, system_prompt, label_list, target_count, client):
    """为单个任务生成完整数据集，返回样本列表（每条附带 task 字段）。"""
    task_dataset = []
    for lbl in label_list:
        print(f"\n  🎬 [{task_name}] 开始生成标签 {lbl} ({track_defs[lbl]['name']})...", flush=True)
        current_pool = []
        retry_count = 0

        while len(current_pool) < target_count and retry_count < 5:
            needed = target_count - len(current_pool)
            print(f"    --> 正在向 LLM 请求 {needed} 条样本...", flush=True)

            batch = generate_batch_by_llm(
                lbl, min(needed, 10),
                client=client,
                system_prompt=system_prompt,
                track_defs=track_defs
            )

            valid_count = 0
            for sample in batch:
                if "history" in sample and len(sample["history"]) == 3 and sample.get("label") == lbl:
                    sample["task"] = task_name
                    current_pool.append(sample)
                    valid_count += 1

            print(f"    --> 本批次吐出 {len(batch)} 条，通过校验: {valid_count} 条", flush=True)
            if valid_count == 0:
                retry_count += 1
            time.sleep(1)

        task_dataset.extend(current_pool[:target_count])
        print(f"  ✅ [{task_name}] 标签 {lbl} 共收集 {len(current_pool[:target_count])} 条", flush=True)

    return task_dataset


# ==================== 新版多头时序数据生成 ====================

# T1：意向走势（3类）
T1_TRACK_DEFINITIONS = {
    0: {
        "name": "平稳（T1-平稳）",
        "guide": "客户全程情绪平稳，连续3轮普通咨询，无明显意向变化，如常规问利率、问产品说明、问操作流程等，第3轮结束时仍无购买倾向。label=0 代表平稳。"
    },
    1: {
        "name": "升温（T1-升温）",
        "guide": "客户从随口询问→第2轮开始关注产品细节或提及资产空置→第3轮明确触发购买意向或要求配置方案/计划书/面签。label=1 代表升温。"
    },
    2: {
        "name": "降温（T1-降温）",
        "guide": "客户从第1轮初期感兴趣或关注产品→第2轮逐渐冷淡、犹豫、说等一等→第3轮失去兴趣、转移话题或明确说暂时不考虑了。label=2 代表降温。"
    },
}

T1_SYSTEM_PROMPT = """你是一名资深的银行营销数据专家。你需要为'客户意向走势追踪模型(T1)'生成训练用的时序对话语料。

T1任务标签含义（必须严格遵守）：
- label=0：平稳 —— 客户全程情绪平稳，连续3轮普通咨询无明显意向变化
- label=1：升温 —— 客户从随口询问→关注产品细节→明确触发购买意向或要求配置方案
- label=2：降温 —— 客户从初期感兴趣→逐渐冷淡→失去兴趣或转移话题

格式要求：
严格输出一个 JSON 对象，包含 "samples" 列表。每条样本必须包含：
- "history": 恰好 3 个字符串的数组（客户连续说出的3句话）
- "label": 整数（0/1/2，严格对应上述T1标签含义，不得混淆）

返回格式示例：
{
  "samples": [
    {"history": ["第一句", "第二句", "第三句"], "label": 1}
  ]
}

语言风格：高度拟真、口语化，包含银行特定术语（安心存、天天盈、面签、纯债基金、年金险、大额存单、跌破净值等）。
"""

# T7：拒绝类型（4类）
T7_TRACK_DEFINITIONS = {
    0: {
        "name": "习惯推脱（T7-习惯推脱）",
        "guide": "客户口头拒绝但无真实异议，3轮对话中反复用时间或忙碌作为借口（如'等我忙完/过两天再说/现在不方便/回头再聊'），实质没有任何拒绝理由，只是惯性推托。label=0 代表习惯推脱。"
    },
    1: {
        "name": "需要理由（T7-需要理由）",
        "guide": "客户真正需要更多信息才能决定，3轮对话中提出具体问题（如'给我分析一下/我要再想想/有没有数据/收益怎么算/风险在哪里'），态度是理性观望而非拒绝。label=1 代表需要理由。"
    },
    2: {
        "name": "真无需求（T7-真无需求）",
        "guide": "客户明确表示不需要该产品，3轮对话中清晰表达'我不需要理财/我已经在别处配好了/我不感兴趣这类产品/这不适合我'等明确拒绝，态度坚定。label=2 代表真无需求。"
    },
    3: {
        "name": "谈判策略（T7-谈判策略）",
        "guide": "客户用拒绝作为筹码来谈判优惠，3轮对话中提出条件（如'利率再高点我就考虑/有没有专属活动/其他银行给我优惠你们能不能匹配/有啥额外回馈'），实质是在博弈让步。label=3 代表谈判策略。"
    },
}

T7_SYSTEM_PROMPT = """你是一名资深的银行营销数据专家。你需要为'客户拒绝类型识别模型(T7)'生成训练用的时序对话语料。

T7任务标签含义（必须严格遵守，与其他任务完全不同）：
- label=0：习惯推脱 —— 口头拒绝但无真实异议（"等我忙完/过两天再说/现在不方便"）
- label=1：需要理由 —— 真正需要更多信息才能决定（"给我分析一下/我要再想想/有没有数据"）
- label=2：真无需求 —— 明确表示不需要该产品（"我不需要理财/我已经在别处配好了"）
- label=3：谈判策略 —— 用拒绝作为筹码来谈判优惠（"利率再高点我就考虑/有没有专属活动"）

格式要求：
严格输出一个 JSON 对象，包含 "samples" 列表。每条样本必须包含：
- "history": 恰好 3 个字符串的数组（客户连续说出的3句话）
- "label": 整数（0/1/2/3，严格对应上述T7标签含义，绝不能按意向演变轨迹理解）

返回格式示例：
{
  "samples": [
    {"history": ["第一句", "第二句", "第三句"], "label": 0}
  ]
}

语言风格：高度拟真、口语化，包含银行特定术语（安心存、天天盈、面签、纯债基金、年金险、大额存单、跌破净值等）。
"""

# T8：化解程度（3类）
T8_TRACK_DEFINITIONS = {
    0: {
        "name": "深度化解（T8-深度化解）",
        "guide": "客户从第1轮强烈异议（质疑产品安全/收益/银行信誉）→第2轮继续追问或对比同业→第3轮被专业分析彻底说服，完全认可，愿意长期持有或立即采取行动（如'好的听你的/那我马上办/我相信你的判断'）。label=0 代表深度化解。"
    },
    1: {
        "name": "表面化解（T8-表面化解）",
        "guide": "客户从第1轮异议→第2轮情绪略有平复→第3轮暂时接受但仍有顾虑，语气变软但并非真心信服（如'那我先观察观察/下次再说吧/暂时先这样'）。label=1 代表表面化解。"
    },
    2: {
        "name": "未化解（T8-未化解）",
        "guide": "客户从第1轮异议→第2轮重复不满或升级情绪→第3轮异议没有被解决，态度持续负面甚至更加强硬（如'还是不行/你说的我不信/反正我不会买/彻底失望了'）。label=2 代表未化解。"
    },
}

T8_SYSTEM_PROMPT = """你是一名资深的银行营销数据专家。你需要为'客户异议化解程度评估模型(T8)'生成训练用的时序对话语料。

T8任务标签含义（必须严格遵守）：
- label=0：深度化解 —— 从强烈异议→被专业分析彻底说服→完全认可（愿意长期持有/立即行动）
- label=1：表面化解 —— 情绪平复但仍有顾虑（暂时接受/下次再说）
- label=2：未化解 —— 持续不满，异议没有被解决，态度负面

格式要求：
严格输出一个 JSON 对象，包含 "samples" 列表。每条样本必须包含：
- "history": 恰好 3 个字符串的数组（客户连续说出的3句话）
- "label": 整数（0/1/2，严格对应上述T8标签含义）

返回格式示例：
{
  "samples": [
    {"history": ["第一句", "第二句", "第三句"], "label": 2}
  ]
}

语言风格：高度拟真、口语化，包含银行特定术语（安心存、天天盈、面签、纯债基金、年金险、大额存单、跌破净值等）。
"""


def main_multihead():
    """生成 T1/T7/T8 三个多头时序任务的训练数据，分别存为独立 JSON 文件。"""
    if not API_KEY:
        raise ValueError("❌ 启动失败：未能在 .env 配置文件或环境变量中检测到 'LLM_API_KEY'，请检查你的配置！")

    print("===================================================================")
    print("🚀 多头时序数据生成引擎启动 [T1/T7/T8]")
    print(f"  ├── 模型基座 URL: {BASE_URL}")
    print(f"  ├── 选定模型实例: {MODEL_NAME}")
    print(f"  └── 每类时序目标: {TARGET_COUNT} 条")
    print("===================================================================\n")

    client = _get_client()

    # ---- T1：意向走势（3类 × 60 = 180条）----
    print("\n======= 开始生成 T1（意向走势）数据 =======", flush=True)
    t1_data = _generate_task_dataset(
        task_name="T1",
        track_defs=T1_TRACK_DEFINITIONS,
        system_prompt=T1_SYSTEM_PROMPT,
        label_list=[0, 1, 2],
        target_count=TARGET_COUNT,
        client=client
    )
    random.shuffle(t1_data)
    t1_path = os.path.join(current_dir, "temporal_t1.json")
    with open(t1_path, "w", encoding="utf-8") as f:
        json.dump(t1_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ T1 数据已保存至: {t1_path}（共 {len(t1_data)} 条）", flush=True)

    # ---- T7：拒绝类型（4类 × 60 = 240条）----
    print("\n======= 开始生成 T7（拒绝类型）数据 =======", flush=True)
    t7_data = _generate_task_dataset(
        task_name="T7",
        track_defs=T7_TRACK_DEFINITIONS,
        system_prompt=T7_SYSTEM_PROMPT,
        label_list=[0, 1, 2, 3],
        target_count=TARGET_COUNT,
        client=client
    )
    random.shuffle(t7_data)
    t7_path = os.path.join(current_dir, "temporal_t7.json")
    with open(t7_path, "w", encoding="utf-8") as f:
        json.dump(t7_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ T7 数据已保存至: {t7_path}（共 {len(t7_data)} 条）", flush=True)

    # ---- T8：化解程度（3类 × 60 = 180条）----
    print("\n======= 开始生成 T8（化解程度）数据 =======", flush=True)
    t8_data = _generate_task_dataset(
        task_name="T8",
        track_defs=T8_TRACK_DEFINITIONS,
        system_prompt=T8_SYSTEM_PROMPT,
        label_list=[0, 1, 2],
        target_count=TARGET_COUNT,
        client=client
    )
    random.shuffle(t8_data)
    t8_path = os.path.join(current_dir, "temporal_t8.json")
    with open(t8_path, "w", encoding="utf-8") as f:
        json.dump(t8_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ T8 数据已保存至: {t8_path}（共 {len(t8_data)} 条）", flush=True)

    print("\n🏆 T1/T7/T8 多头时序数据生成完成！", flush=True)
    print(f"  T1: {len(t1_data)} 条  →  {t1_path}", flush=True)
    print(f"  T7: {len(t7_data)} 条  →  {t7_path}", flush=True)
    print(f"  T8: {len(t8_data)} 条  →  {t8_path}", flush=True)


# ==================== 旧版4分类生成逻辑（保持兼容）====================

def main_legacy():
    """旧版4分类时序数据生成（保持兼容，生成 temporal_signal_llm_augmented.json）。"""
    if not API_KEY:
        raise ValueError("❌ 启动失败：未能在 .env 配置文件或环境变量中检测到 'LLM_API_KEY'，请检查你的配置！")

    print("===================================================================")
    print("🚀 成功载入外部配置文件 [.env]（旧版4分类生成模式）")
    print(f"  ├── 模型基座 URL: {BASE_URL}")
    print(f"  ├── 选定模型实例: {MODEL_NAME}")
    print(f"  └── 每类时序目标: {TARGET_COUNT} 条")
    print("===================================================================\n")

    client = _get_client()

    final_llm_dataset = []
    target_plan = {1: TARGET_COUNT, 2: TARGET_COUNT, 3: TARGET_COUNT, 0: TARGET_COUNT}

    for lbl, target_count in target_plan.items():
        print(f"\n🎬 开始调度 LLM 攻坚生成标签 {lbl} ({TRACK_DEFINITIONS[lbl]['name']})...", flush=True)
        current_pool = []
        retry_count = 0

        while len(current_pool) < target_count and retry_count < 5:
            needed = target_count - len(current_pool)
            print(f"  --> 正在向 LLM 请求 {needed} 条高质量样本...", flush=True)

            batch = generate_batch_by_llm(lbl, min(needed, 10), client=client)

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

    json_output_file = os.path.join(current_dir, "temporal_signal_llm_augmented.json")
    npy_output_file = os.path.join(current_dir, "temporal_signal_llm_augmented.npy")

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(final_llm_dataset, f, ensure_ascii=False, indent=2)
    np.save(npy_output_file, final_llm_dataset)

    print(f"\n🏆 外部配置驱动的 LLM 深度冷启动数据增强完成！共收集 {len(final_llm_dataset)} 条高拟真轨迹语料。", flush=True)


if __name__ == "__main__":
    main_multihead()

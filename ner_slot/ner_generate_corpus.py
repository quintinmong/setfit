import os
import json
import time
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 0. 载入外部 .env 配置（自动向上级回溯寻址根目录的 .env）
load_dotenv(find_dotenv())
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

if not API_KEY:
    raise ValueError("❌ 未在环境变量或 .env 中找到 'LLM_API_KEY'")

print("===================================================================")
print("🚀 启动基于大模型（LLM）的自定义 NER 语料冷启动流水线...")
print(f"  ├── 基座 URL: {BASE_URL} | 实例: {MODEL_NAME}")
print("===================================================================\n")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 方案定义的四大金融专属实体类型
ENTITY_DEFINITIONS = {
    "TIME": "时间实体。例如：下周二、明天下午三点、这期到期后、忙完这段时间、节日当天。",
    "WAIT_COND": "触发动作的等待条件或前置事件。例如：工程款下来、资金到账、股市震荡完、降准落地、放额度时。",
    "RELATION_PERSON": "对话提及的实际关系人或决策人。例如：我父母、我家孩子、老太婆、小微企业主管。",
    "MATERIAL": "特定的业务材料、方案或资产证明。例如：年金保险计划书、存款证明、纯债基金、大额存单。"
}

SYSTEM_PROMPT = """你是一名银行合规数据标注专家。你需要为客户经理数字分身 Agent 的命名实体识别（NER）子系统生成高质量的中文冷启动种子语料。

格式要求：
你必须严格输出一个 JSON 数组，数组内包含指定数量的 JSON 对象。每个对象必须严格包含以下两个字段，不得包含任何 Markdown 包裹标签或解释性文字：
- "text": 客户说的一句完整的金融/理财对话文本。
- "entities": 一个数组，包含该文本中抽取的实体对象。每个实体对象包含：
    * "word": 抽取的实体明文字符串。
    * "type": 必须是 ["TIME", "WAIT_COND", "RELATION_PERSON", "MATERIAL"] 之一。

生成风格：
高度贴合银行网银/手机银行真实口语。文本必须包含 1 到 3 个上述实体类型，实体字词必须在 text 中完全一致。

示例：
[
  {
    "text": "手头刚下来一笔工程款，你帮我做个全面的资产配置规划？",
    "entities": [
      {"word": "工程款", "type": "WAIT_COND"}
    ]
  },
  {
    "text": "你上次说的那款年金保险，计划书再发我研究一下，下周我带我父母去面签。",
    "entities": [
      {"word": "年金保险", "type": "MATERIAL"},
      {"word": "计划书", "type": "MATERIAL"},
      {"word": "下周", "type": "TIME"},
      {"word": "我父母", "type": "RELATION_PERSON"}
    ]
  }
]
"""


def generate_ner_batch(count_to_generate):
    user_prompt = f"请严格生成 {count_to_generate} 条高质量银行客户对话的 NER 标注样本，确保均衡覆盖以下实体类型：\n{json.dumps(ENTITY_DEFINITIONS, ensure_ascii=False, indent=2)}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)

        # 兼容处理一层 key 包裹
        if isinstance(data, dict):
            for key in ["samples", "data", "dataset", "list", "corpus"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            if len(data.keys()) == 1 and isinstance(list(data.values())[0], list):
                return list(data.values())[0]
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  ❌ 调用大模型生成 NER 批次失败: {e}")
        return []


# ==================== 🪐 循环收集与本地 BIO 转换 ====================
target_total = 100  # 体验版冷启动先生成 100 条高质量带槽位文本
ner_raw_dataset = []

while len(ner_raw_dataset) < target_total:
    needed = target_total - len(ner_raw_dataset)
    print(
        f"--> 正在向大模型请求 {min(needed, 15)} 条 NER 槽位样本 (当前进度: {len(ner_raw_dataset)}/{target_total})...")
    batch = generate_ner_batch(min(needed, 15))

    valid_count = 0
    for sample in batch:
        if "text" in sample and "entities" in sample and isinstance(sample["entities"], list):
            # 校验实体是否真的在文本里
            is_valid = True
            for ent in sample["entities"]:
                if ent.get("word") not in sample["text"]:
                    is_valid = False
                    break
            if is_valid:
                ner_raw_dataset.append(sample)
                valid_count += 1

    print(f"  └── 本批次大模型吐出 {len(batch)} 条，合规通过校验: {valid_count} 条")
    time.sleep(1)

# 保存大模型生成的原生 JSON 格式，方便调试与槽位匹配测试
output_json_path = "./ner_slot/ner_raw_corpus.json"
os.makedirs("./ner_slot", exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(ner_raw_dataset, f, ensure_ascii=False, indent=2)

print(f"\n🏆 NER 槽位冷启动原始语料成功固化到: '{output_json_path}'")

# 展示 2 条看效果
for case in ner_raw_dataset[:2]:
    print(f"\n文本: {case['text']}")
    for e in case['entities']:
        print(f"  └── 抽取实体 -> 词: {e['word']} | 类型: {e['type']}")
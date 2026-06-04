import os
import json
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel

# ==================== NER 标签定义 ====================
NER_LABELS = ["O", "B-TIME", "I-TIME", "B-WAIT_COND", "I-WAIT_COND",
              "B-RELATION_PERSON", "I-RELATION_PERSON", "B-MATERIAL", "I-MATERIAL"]
NER_ID2LABEL = {i: l for i, l in enumerate(NER_LABELS)}


# ==================== 🤖 NER 推理器 ====================
class NERInferencer:
    """加载训练好的 BGE+Linear NER 头，对单句文本做命名实体识别推理。"""

    def __init__(self, bert_path, head_path, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path).to(device)
        self.bert.eval()
        self.head = nn.Linear(self.bert.config.hidden_size, len(NER_LABELS)).to(device)
        self.head.load_state_dict(torch.load(head_path, map_location=device))
        self.head.eval()

    def predict(self, text: str) -> list:
        """返回 [{"word": "...", "type": "TIME"}, ...] 格式的实体列表。"""
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(outputs.last_hidden_state)
            pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

        entities = []
        current_entity = None  # {"type": ..., "_char_start": ..., "_char_end": ...}

        for pred_id, (start, end) in zip(pred_ids, offset_mapping):
            if start == end:  # 特殊/填充 token
                if current_entity:
                    entities.append({"word": text[current_entity["_char_start"]:current_entity["_char_end"]], "type": current_entity["type"]})
                    current_entity = None
                continue

            label = NER_ID2LABEL[pred_id]

            if label.startswith("B-"):
                if current_entity:
                    entities.append({"word": text[current_entity["_char_start"]:current_entity["_char_end"]], "type": current_entity["type"]})
                current_entity = {"type": label[2:], "_char_start": start, "_char_end": end}
            elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
                current_entity["_char_end"] = end
            else:
                if current_entity:
                    entities.append({"word": text[current_entity["_char_start"]:current_entity["_char_end"]], "type": current_entity["type"]})
                current_entity = None

        if current_entity:
            entities.append({"word": text[current_entity["_char_start"]:current_entity["_char_end"]], "type": current_entity["type"]})

        return entities



# ==================== 🧊 1. 字符级 BIO 标签对齐层 ====================
def convert_to_bio(text, entities):
    """
    将文本和实体列表严格对齐转化为传统的 BIO 标签序列 (用于未来微调 MacBERT)
    """
    labels = ["O"] * len(text)
    for ent in entities:
        word = ent["word"]
        ent_type = ent["type"]

        # 在文本中查找实体的起始偏移量
        start_idx = text.find(word)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(word)

        # 填充 BIO 标签
        labels[start_idx] = f"B-{ent_type}"
        for i in range(start_idx + 1, end_idx):
            labels[i] = f"I-{ent_type}"

    return labels


# ==================== 🕒 2. 时间层分层解析器 (Time Parser) ====================
def parse_time_logic(time_text):
    """
    根据方案要求，对 TIME 实体进行多层逻辑解析，将其转化为标准化时间戳或业务信号
    """
    now = datetime.now()
    time_text_clean = time_text.strip()

    if "明天" in time_text_clean:
        target_date = now + timedelta(days=1)
        return {"type": "EXACT_DATE", "value": target_date.strftime("%Y-%m-%d"), "desc": "明天"}
    elif "下周二" in time_text_clean:
        # 计算到下周二还有几天
        days_ahead = 1 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = now + timedelta(days=days_ahead + 7)
        return {"type": "EXACT_DATE", "value": target_date.strftime("%Y-%m-%d"), "desc": "下周二"}
    elif "到期后" in time_text_clean or "忙完" in time_text_clean or "等我" in time_text_clean:
        return {"type": "EVENT_DEPENDENT", "value": "PENDING_EVENT", "desc": "依赖前置事件触发"}

    return {"type": "FUZZY_TIME", "value": None, "desc": f"模糊时间段: {time_text_clean}"}


# ==================== 🎰 3. 槽位状态机管理层 (Slot Filling Machine) ====================
class SlotFillingStateMachine:
    """
    依照方案设计的槽位三态（FILLED / SUGGESTED / EMPTY）与决策逻辑进行槽位维护
    """

    def __init__(self):
        # 初始化四大业务核心槽位
        self.slots = {
            "RESERVED_TIME": {"status": "EMPTY", "value": None, "raw": None},  # 预约时间
            "WAIT_CONDITION": {"status": "EMPTY", "value": None, "raw": None},  # 前置条件
            "DECISION_MAKER": {"status": "EMPTY", "value": None, "raw": None},  # 关键关系人
            "TARGET_MATERIAL": {"status": "EMPTY", "value": None, "raw": None}  # 目标材料/产品
        }

    def inject_entities(self, entities):
        """
        根据实体识别结果，动态更新状态机内部各个槽位的状态三态
        """
        for ent in entities:
            word = ent["word"]
            ent_type = ent["type"]

            if ent_type == "TIME":
                parsed_time = parse_time_logic(word)
                self.slots["RESERVED_TIME"] = {
                    "status": "FILLED" if parsed_time["value"] else "SUGGESTED",
                    "value": parsed_time["value"] if parsed_time["value"] else parsed_time["desc"],
                    "raw": word
                }
            elif ent_type == "WAIT_COND":
                self.slots["WAIT_CONDITION"] = {"status": "FILLED", "value": word, "raw": word}
            elif ent_type == "RELATION_PERSON":
                # 指代词或关系人注入
                status = "SUGGESTED" if word in ["他", "她", "他们"] else "FILLED"
                self.slots["DECISION_MAKER"] = {"status": status, "value": word, "raw": word}
            elif ent_type == "MATERIAL":
                self.slots["TARGET_MATERIAL"] = {"status": "FILLED", "value": word, "raw": word}

    def emit_downstream_decision(self):
        """
        根据当前槽位的饱满度，生成给话术 Agent 决策层的执行指令 (触发流水 vs 继续追问)
        """
        # 如果核心的时间和物料都齐备了，直接触发业务流程
        if self.slots["TARGET_MATERIAL"]["status"] == "FILLED" and self.slots["RESERVED_TIME"]["status"] == "FILLED":
            return "🎯 [ACTION_TRIGGER] -> 槽位就绪，触发后台网银流水或自动创建客户经理跟进任务清单。"

        # 如果缺材料，但有时间
        if self.slots["TARGET_MATERIAL"]["status"] == "EMPTY" and self.slots["RESERVED_TIME"]["status"] != "EMPTY":
            return "❓ [ASK_MATERIAL] -> 时间已定，触发追问话术：‘请问您具体想看的是哪一款理财或保险计划书呢？’"

        # 如果缺明确时间
        if self.slots["RESERVED_TIME"]["status"] == "EMPTY":
            return "❓ [ASK_TIME] -> 核心时间槽位缺失，触发追问话术：‘小张随时恭候，您看下周或者什么时间比较方便，我帮您提前锁好额度？’"

        return "🔄 [KEEP_CONVERSATION] -> 保持多维话术沟通，提取更多潜在语义。"

    def print_slots_report(self):
        print("  ┌── 📊 当前槽位状态机看板 (Slot State Report):")
        for slot_name, info in self.slots.items():
            print(
                f"  │   ├── {slot_name:<16}: 状态:[{info['status']:<9}] | 映射值:{str(info['value']):<15} | 原始实体:{str(info['raw'])}")
        print("  └── 🏁")



# ==================== 🪐 4. 真实数据闭环集成压力测试 (路径终极加固版) ====================
def main():
    print("===================================================================")
    print("🚀 启动命名实体识别（NER）对齐与槽位填充（Slot Filling）业务引擎...")
    print("===================================================================\n")

    # 动态获取当前脚本 (ner_infer_and_slot_fill.py) 所在的绝对目录路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 强制锁定同级目录下的数据文件绝对路径
    DATA_PATH = os.path.join(CURRENT_DIR, "ner_raw_corpus.json")

    print(f"📡 动态路径追踪：正在锁定绝对路径读取数据 -> {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 错误：在目标路径 [{DATA_PATH}] 未找到数据文件，请确认该文件是否与当前脚本在同一目录下！")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        test_corpus = json.load(f)

    print(f"成功载入大模型生成的原始槽位语料，共计抽取前 3 个典型对抗用例进行流程推演：\n")

    for idx, case in enumerate(test_corpus[:3]):
        text = case["text"]
        entities = case["entities"]

        print(f"🔥 【测试用例 {idx + 1}】 客户发言: \"{text}\"")

        # A. 打印 BIO 字符对齐数组 (检验底层特征对齐)
        bio_tags = convert_to_bio(text, entities)
        print(
            f"  ├── 🏷 BIO 序列标注对齐片段: {[''.join(t) + '/' + g for t, g in zip(text, bio_tags)][:8]}... (共{len(bio_tags)}字)")

        # B. 送入槽位状态机进行更新
        sf_machine = SlotFillingStateMachine()
        sf_machine.inject_entities(entities)

        # C. 打印当前的看板状态
        sf_machine.print_slots_report()

        # D. 输出下游决策判断
        decision = sf_machine.emit_downstream_decision()
        print(f"  └── 🧭 话术决策路由判定: {decision}\n")


if __name__ == "__main__":
    main()
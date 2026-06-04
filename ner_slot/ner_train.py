import sys
import os

# ================= 🚨 猴子补丁：防止 transformers 报错 🚨 =================
import transformers.training_args
if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

BERT_PATH = os.path.join(root_dir, "my_final_six_intents_model/bge_encoder")
DATA_PATH = os.path.join(current_dir, "ner_raw_corpus.json")
SAVE_PATH = os.path.join(root_dir, "my_final_six_intents_model/ner_head_weights.pth")

LABELS = ["O", "B-TIME", "I-TIME", "B-WAIT_COND", "I-WAIT_COND",
          "B-RELATION_PERSON", "I-RELATION_PERSON", "B-MATERIAL", "I-MATERIAL"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)
MAX_LENGTH = 128
EPOCHS = 30
BATCH_SIZE = 8
LR = 2e-3


def align_labels_to_tokens(tokenizer, text, entities):
    """字符级BIO对齐到token级，返回token标签列表和encoding。"""
    char_labels = ["O"] * len(text)
    for ent in entities:
        word = ent["word"]
        ent_type = ent["type"]
        start_idx = text.find(word)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(word)
        char_labels[start_idx] = f"B-{ent_type}"
        for i in range(start_idx + 1, end_idx):
            char_labels[i] = f"I-{ent_type}"

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    offset_mapping = encoding.pop("offset_mapping")

    token_labels = []
    for (start, end) in offset_mapping:
        if start == end:  # [CLS] / [SEP] / [PAD]
            token_labels.append(-100)
        else:
            label_str = char_labels[start] if start < len(char_labels) else "O"
            token_labels.append(LABEL2ID.get(label_str, 0))

    return token_labels, encoding


class NERDataset(Dataset):
    def __init__(self, encodings_list, labels_list):
        self.input_ids = [e["input_ids"] for e in encodings_list]
        self.attention_mask = [e["attention_mask"] for e in encodings_list]
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


class BertNERModel(nn.Module):
    def __init__(self, bert_path, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        return self.classifier(sequence_output)


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("===================================================================")
    print(f"🚀 NER 分类头训练引擎启动 | 核心设备: {device}")
    print("===================================================================")

    print(f"\n📂 加载 NER 语料: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  └── 共载入 {len(corpus)} 条标注样本")

    print(f"\n🧠 加载底座 Tokenizer: {BERT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    print("\n⚡️ 正在进行 Token 级标签对齐...")
    encodings_list = []
    labels_list = []
    for sample in corpus:
        token_labels, encoding = align_labels_to_tokens(tokenizer, sample["text"], sample["entities"])
        encodings_list.append(encoding)
        labels_list.append(token_labels)
    print(f"  └── 标签对齐完成，共 {len(labels_list)} 条序列")

    dataset = NERDataset(encodings_list, labels_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n🏗 初始化 BertNERModel（冻结底座，只训 Linear Head）...")
    model = BertNERModel(BERT_PATH, NUM_LABELS).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  └── 可训练参数: {trainable:,}（底座已冻结）")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    print(f"\n🪐 开始训练 (Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR})...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ├── Epoch [{epoch:02d}/{EPOCHS}] -> Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.classifier.state_dict(), SAVE_PATH)
    print(f"\n🏆 NER 分类头已成功保存至: '{SAVE_PATH}'")


if __name__ == "__main__":
    main()

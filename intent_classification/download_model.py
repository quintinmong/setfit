import os
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModel, AutoTokenizer

# 0. 载入外部 .env 配置（自动向上级回溯寻址根目录的 .env）
load_dotenv(find_dotenv())

# 1. 从配置中抽取，若不存在则使用 MacBERT-base 官方模型
model_id = os.getenv("EMBEDDING_MODEL_ID", "hfl/chinese-macbert-base")

# 2. 存入配置对应的本地路径，若为相对路径则自动解析为相对于项目根目录的路径
local_dir = os.getenv("EMBEDDING_MODEL_DIR", "./models/chinese-macbert-base")
if not os.path.isabs(local_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    local_dir = os.path.abspath(os.path.join(root_dir, local_dir))

print(f"开始下载 MacBERT 底座: {model_id} ...")

# 3. 通过 Transformers 下载 tokenizer/model 并持久化
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, use_safetensors=False)
hidden_size = getattr(model.config, "hidden_size", None)
if hidden_size != 768:
    raise ValueError(
        f"当前模型 hidden_size={hidden_size}，但本项目 MacBERT 方案要求 768 维。"
        "请将 .env 中的 EMBEDDING_MODEL_ID/EMBEDDING_MODEL_DIR 更新为 "
        "hfl/chinese-macbert-base / ./models/chinese-macbert-base。"
    )
os.makedirs(local_dir, exist_ok=True)
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"\nMacBERT 底座已下载并保存至 {local_dir}")

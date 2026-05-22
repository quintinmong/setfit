import os
from modelscope import snapshot_download
from dotenv import load_dotenv, find_dotenv

# 0. 载入外部 .env 配置（自动向上级回溯寻址根目录的 .env）
load_dotenv(find_dotenv())

# 1. 从配置中抽取，若不存在则回退至默认 1.5 中文底座
model_id = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-small-zh-v1.5")

# 2. 存入配置对应的本地路径，若为相对路径则自动解析为相对于项目根目录的路径
local_dir = os.getenv("EMBEDDING_MODEL_DIR", "./models/bge-small-zh-v1.5")
if not os.path.isabs(local_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    local_dir = os.path.abspath(os.path.join(root_dir, local_dir))

print(f"🚀 开始全速下载 {model_id}...")

# 3. 阿里云全速下载
snapshot_download(
    model_id=model_id,
    local_dir=local_dir,
    cache_dir=None
)

print(f"\n🎉 底座已完美下载并装入 {local_dir}")

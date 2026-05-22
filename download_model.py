import os
from modelscope import snapshot_download

# 1. 严格遵守方案，换回 1.5 中文版底座
model_id = "BAAI/bge-small-zh-v1.5"

# 2. 存入方案对应的本地路径
local_dir = "./models/bge-small-zh-v1.5"

print(f"🚀 严格执行方案！开始全速下载 {model_id}...")

# 3. 阿里云全速下载
snapshot_download(
    model_id=model_id,
    local_dir=local_dir,
    cache_dir=None
)

print(f"\n🎉 1.5版本底座已完美打包装入 {local_dir}")
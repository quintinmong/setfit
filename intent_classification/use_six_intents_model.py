import os
import sys
import joblib
from sentence_transformers import SentenceTransformer

# 路径加固
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# 适配 git worktree 场景
_main_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(root_dir))) if ".claude/worktrees" in root_dir else root_dir
model_dir = os.path.abspath(os.path.join(_main_repo_root, "my_final_six_intents_model"))

# 加载编码器
encoder = SentenceTransformer(os.path.join(model_dir, "bge_encoder"))

# 加载 artifact，校验版本
artifact = joblib.load(os.path.join(model_dir, "classification_heads.pkl"))
assert artifact.get("version") == 2, "旧版模型格式，请重新运行 six_intent_server.py"
heads = artifact["heads"]
label_maps = artifact["label_maps"]

# 测试样本
test_sentences = [
    "收益太低了，想转去招行",
    "手机银行怎么操作提前支取大额存单？",
    "我看网上说金融危机要来了，你们银行靠得住吗？",
    "手头有笔工程款，帮我做个资产配置规划吧",
    "垃圾银行，我要把钱全转走销户！",
]

print("===== 6维意图分类推理测试 =====\n")
for sentence in test_sentences:
    features = encoder.encode([sentence], show_progress_bar=False)
    print(f"  输入: 「{sentence}」")
    for head_name in heads:
        pred = heads[head_name].predict(features)[0]
        label_str = label_maps[head_name][pred]
        print(f"    {head_name}: {label_str}")
    print()

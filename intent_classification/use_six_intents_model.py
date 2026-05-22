import os
import joblib
from sentence_transformers import SentenceTransformer

# 获取相对于当前脚本根目录的绝对路径，确保不论从何处运行，位置均一致
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

model_dir = os.path.abspath(os.path.join(root_dir, "my_final_six_intents_model"))

# 直接从刚才保存的路径加载（仅花 0.2 秒）
encoder = SentenceTransformer(os.path.join(model_dir, "bge_encoder"))
heads = joblib.load(os.path.join(model_dir, "classification_heads.pkl"))

# 在线拿一句话来测试
features = encoder.encode(["收益太低了，想转去招行"], show_progress_bar=False)
pred_scene = heads["scene"].predict(features)[0]
print("秒开加载后，识别出来的场景维度为：", pred_scene)

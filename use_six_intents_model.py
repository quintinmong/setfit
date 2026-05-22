import joblib
from sentence_transformers import SentenceTransformer

# 直接从刚才保存的路径加载（仅花 0.2 秒）
encoder = SentenceTransformer("./my_final_six_intents_model/bge_encoder")
heads = joblib.load("./my_final_six_intents_model/classification_heads.pkl")

# 在线拿一句话来测试
features = encoder.encode(["收益太低了，想转去招行"], show_progress_bar=False)
pred_scene = heads["scene"].predict(features)[0]
print("秒开加载后，识别出来的场景维度为：", pred_scene)
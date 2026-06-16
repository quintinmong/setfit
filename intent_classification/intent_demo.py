import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression

# 1. 模拟冷启动数据
train_data = {
    "text": [
        "请问手机银行在哪里可以买理财？",
        "我想看看民生银行最新的定期存款利率",
        "这个理财产品的申购入口在哪里",
        "找不到理财购买页面了，帮我看一下",
        "现在的理财收益也太低了吧，真差劲",
        "怎么天天跌啊，感觉比以前差多了",
        "你们这个APP太难用了，经常卡顿",
        "气死我了，今天又亏了这么多"
    ],
    "label": [0, 0, 0, 0, 1, 1, 1, 1]
}

# 获取相对于当前脚本根目录的绝对路径，确保不论从何处运行，位置均一致
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from shared_encoder import ENCODER_ARTIFACT_DIR, MacBertEncoder, resolve_encoder_path

# 2. 加载共享冻结 MacBERT 底座
MODEL_PATH = resolve_encoder_path(root_dir)
if MODEL_PATH is None:
    raise FileNotFoundError("找不到 MacBERT 底座！请先运行 python3 intent_classification/download_model.py。")

print(f"正在从本地加载 MacBERT 底座模型: {MODEL_PATH} ...")
encoder = MacBertEncoder(MODEL_PATH)

# 3. 冻结底座抽特征，只训练轻量分类头
print("开始抽取训练特征并拟合二分类头...")
X_train = encoder.encode(train_data["text"], show_progress_bar=False)
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, train_data["label"])
print("训练完成！")

# 5. 现场合围推理测试
print("\n--- 开始测试推理 ---")
test_inputs = [
    "怎么在手机上买基金啊？",
    "收益才3.2%，垃圾银行"
]

X_test = encoder.encode(test_inputs, show_progress_bar=False)
preds = clf.predict(X_test)

for text, pred in zip(test_inputs, preds):
    label_name = "咨询响应" if pred == 0 else "情绪抱怨"
    print(f"输入: '{text}' ---> 识别意图为: [{label_name}]")

# 6. 保存模型
save_path = os.path.abspath(os.path.join(root_dir, "my_first_intent_model"))
os.makedirs(save_path, exist_ok=True)
encoder.save(os.path.join(save_path, ENCODER_ARTIFACT_DIR))
joblib.dump(clf, os.path.join(save_path, "binary_intent_head.pkl"))
print(f"\n业务模型已成功保存至 '{save_path}'")

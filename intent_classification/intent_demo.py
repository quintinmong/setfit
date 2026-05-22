import sys

# ================= 🚨 猴子补丁终极进化版 🚨 =================
# 把它改成一个返回字符串的匿名函数，完美契合 setfit() 的调用
import transformers.training_args

if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import os
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

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

train_dataset = Dataset.from_dict(train_data)

# 获取相对于当前脚本根目录的绝对路径，确保不论从何处运行，位置均一致
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# 2. 严格指向你本地用 ModelScope 下载好的 1.5 路径
MODEL_PATH = os.path.abspath(os.path.join(root_dir, "models/bge-small-zh-v1.5"))

print(f"正在从本地加载 BGE-1.5 底座模型: {MODEL_PATH} ...")
model = SetFitModel.from_pretrained(MODEL_PATH)

# 3. 设置轻量训练参数
training_args = TrainingArguments(
    batch_size=4,
    num_epochs=2,
    evaluation_strategy="no"
)

# 4. 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("开始少样本微调训练...")
trainer.train()
print("训练完成！")

# 5. 现场合围推理测试
print("\n--- 开始测试推理 ---")
test_inputs = [
    "怎么在手机上买基金啊？",
    "收益才3.2%，垃圾银行"
]

preds = model.predict(test_inputs)

for text, pred in zip(test_inputs, preds):
    label_name = "咨询响应" if pred == 0 else "情绪抱怨"
    print(f"输入: '{text}' ---> 识别意图为: [{label_name}]")

# 6. 保存模型
save_path = os.path.abspath(os.path.join(root_dir, "my_first_intent_model"))
model.save_pretrained(save_path)
print(f"\n业务模型已成功保存至 '{save_path}'")

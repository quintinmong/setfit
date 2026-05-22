import sys

# ================= 🚨 猴子补丁：防止 transformers 报错 🚨 =================
import transformers.training_args
if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import os
import torch
import joblib  # 用于保存机器学习分类头
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# 获取相对于当前脚本根目录的绝对路径，确保不论从何处运行，位置均一致
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# 1. 严格使用方案指定的本地 1.5 底座
MODEL_PATH = os.path.abspath(os.path.join(root_dir, "models/bge-small-zh-v1.5"))

print(f"正在初始化共享底座 SentenceTransformer: {MODEL_PATH} ...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = SentenceTransformer(MODEL_PATH, device=device)

# 2. 60条符合方案标准的多维高密度数据集
# 字段说明：sc(场景), th(快慢), em(情绪), as_sig(资产信号), sk(熟练度), tr(信任度)
train_data = [
    # ==================== 【场景：咨询响应 (sc=0)】 ====================
    {"text": "请问手机银行在哪里可以买理财？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "我想看看民生银行最新的定期存款利率", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "找不到理财购买页面了，帮我看一下，急死人了", "sc": 0, "th": 0, "em": 2, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "手机上怎么操作提前支取大额存单啊？", "sc": 0, "th": 0, "em": 0, "as_sig": 1, "sk": 1, "tr": 0},
    {"text": "小张，我想问下这款中低风险的产品保本吗？", "sc": 0, "th": 0, "em": 2, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "你们网点那个安心存产品，每天几点开始放额度？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "合同里写的这个管理费和托管费是怎么扣的？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "怎么看我买的基金今天收益是多少？在哪点进去？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "这几款天天盈理财有什么区别吗？哪个随时能取？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "你们这儿的外币存款，美元一年期利率现在是多少？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "我这岁数大了眼花，APP上找不到风险评估在哪里做。", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "请问如果我要开个存款证明，必须去网点吗？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "转账限额怎么修改？我想把单笔限额调高到十万。", "sc": 0, "th": 0, "em": 0, "as_sig": 1, "sk": 1, "tr": 0},
    {"text": "这个储蓄国债 and 电子式国债有什么不一样吗？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "小张，今天发行的那款专享理财，编码是多少啊？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},

    # ==================== 【场景：情绪抱怨 (sc=1)】 ====================
    {"text": "现在的理财收益也太低了吧，真差劲", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "怎么天天跌啊，感觉比以前差多了！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "你们这个APP太难用了，经常卡顿闪退", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "气死我了，今天怎么又亏了这么多钱，搞什么啊", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "垃圾银行，体验太差了，我要把钱全转走销户！", "sc": 1, "th": 0, "em": 1, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "收益才3.2%，感觉比以前差多了", "sc": 1, "th": 1, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "招行收益好像更高，有点动摇了", "sc": 1, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "我看网上说现在要发生金融危机了，理财还安全吗？", "sc": 1, "th": 1, "em": 2, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "隔壁建设银行天天给我推4.0的利息，你们怎么这么低？", "sc": 1, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "买的时候说是稳健型，结果绿成这样，你们骗人吧！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "坑人啊，说好年化回报挺高的，这都几个月了全是负的！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "微众银行的活期都比你们定期强，真后悔放你们这。", "sc": 1, "th": 1, "em": 1, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "听说最近国债都要爆雷了？现在信托也不行，到底能信谁？", "sc": 1, "th": 1, "em": 2, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "你们客户经理之前推荐的时候拍胸脯，现在亏了就找借口！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "为什么别人家的APP操作那么傻瓜，你们的这么绕？", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 1, "tr": 0},

    # ==================== 【场景：主动服务/关系维护 (sc=2)】 ====================
    {"text": "谢谢小张经理，你的建议很专业", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "早啊，今天有什么新产品推荐吗？", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "节日祝福收到了，也祝你节日快乐", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "行，那我下周有空去网点找你做个面签吧", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "最近确实比较忙，等我忙完这段时间再找你配置", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "你们民生银行的服务态度确实没得说，挺体贴的", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "最近股市震荡成这样，我总觉得心里不踏实，钱放哪好？", "sc": 2, "th": 1, "em": 2, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "手头刚下来一笔工程款，你帮我做个全面的资产配置规划？", "sc": 2, "th": 1, "em": 0, "as_sig": 1, "sk": 0, "tr": 0},
    {"text": "我想给我家孩子存笔教育金，不知道怎么规划长线稳定", "sc": 2, "th": 1, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "小张，听说最近国债利率又降了？对我们持有有影响吗？", "sc": 2, "th": 1, "em": 2, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "收到你的降息提醒了，那这期到期后我换成纯债基金行吗？", "sc": 2, "th": 1, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "我准备给父母留一笔养老钱，追求绝对安全，推荐个方案。", "sc": 2, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 0},
    {"text": "没关系，刚好路过你们网点，顺便进来喝杯水看看你。", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "你上次说的那款年金保险，计划书再发我研究一下。", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "听说有专属的老客回馈活动？我能参加吗？", "sc": 2, "th": 0, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},

    # ==================== 【极高对抗度追加混合集】 ====================
    {"text": "网上说现在有很多银行爆雷，你们民生银行靠得住吗？", "sc": 1, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "我买了一百多万的稳健理财结果还亏，你们要给我个解释！", "sc": 1, "th": 0, "em": 1, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "钱放网商银行安全还是放你们大额存单安全？说真心话。", "sc": 0, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "怎么购买理财还要录音录像这么麻烦？老太婆操作不来！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "换了新手机，登录APP提示密码锁定了，现在怎么办？", "sc": 0, "th": 0, "em": 2, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "把钱放你们这我心里直打鼓，现在利息还这么低，我想转走。", "sc": 1, "th": 1, "em": 2, "as_sig": 1, "sk": 0, "tr": 1},
    {"text": "你们这大额存单要三十万起购？手机上怎么凑单买？", "sc": 0, "th": 0, "em": 0, "as_sig": 1, "sk": 1, "tr": 0},
    {"text": "现在市场全在降息，买长期定期划算还是买年金保险划算？", "sc": 2, "th": 1, "em": 0, "as_sig": 0, "sk": 0, "tr": 0},
    {"text": "别天天发微信推销了，烦不烦，有高收益产品再找我！", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "我卡里还有二十万活期，你看看帮我怎么规划能抗通胀？", "sc": 2, "th": 1, "em": 0, "as_sig": 1, "sk": 0, "tr": 0},
    {"text": "手机验证码一直收不到，你们系统是不是瘫痪了啊？", "sc": 1, "th": 0, "em": 1, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "我看别家收益都挺高，你们产品经理是不是水平不行啊？", "sc": 1, "th": 1, "em": 1, "as_sig": 0, "sk": 0, "tr": 1},
    {"text": "小张，刚看新闻降准了，对我们小微企业贷款有好处吗？", "sc": 2, "th": 1, "em": 0, "as_sig": 1, "sk": 0, "tr": 0},
    {"text": "我年纪大了记不住密码，能不能改成手机刷脸登录？", "sc": 0, "th": 0, "em": 0, "as_sig": 0, "sk": 1, "tr": 0},
    {"text": "你们这保本的玩意儿一个都没了？那本金亏了你们赔吗？", "sc": 1, "th": 0, "em": 2, "as_sig": 0, "sk": 0, "tr": 1}
]

train_texts = [item["text"] for item in train_data]
y_sc = [item["sc"] for item in train_data]
y_th = [item["th"] for item in train_data]
y_em = [item["em"] for item in train_data]
y_as = [item["as_sig"] for item in train_data]
y_sk = [item["sk"] for item in train_data]
y_tr = [item["tr"] for item in train_data]

# 3. 底座统一抽取高维语义特征（只进行一次）
print(f"核心优化：底座对 {len(train_texts)} 条样本进行特征提取...")
X_train = encoder.encode(train_texts, show_progress_bar=False)

# 4. 并行训练六路完全解耦的分类头
print("正在并行训练 6 个独立的轻量分类头...")
head_sc = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_sc)
head_th = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_th)
head_em = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_em)
head_as = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_as)
head_sk = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_sk)
head_tr = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_tr)
print("🏆 六路并行分类头训练完成！")

# ==================== 💾 核心加固：将成果物持久化固化到本地硬盘 ====================
SAVE_DIR = os.path.abspath(os.path.join(root_dir, "my_final_six_intents_model"))
os.makedirs(SAVE_DIR, exist_ok=True)

# 备份或直接固化底座模型
encoder.save(os.path.join(SAVE_DIR, "bge_encoder"))

# 打包 6 个分类头存入一个单一的字典文件
heads_dict = {
    "scene": head_sc,
    "thinking": head_th,
    "emotion": head_em,
    "asset": head_as,
    "skill": head_sk,
    "trust": head_tr
}
joblib.dump(heads_dict, os.path.join(SAVE_DIR, "classification_heads.pkl"))

print(f"\n🔥 恭喜！全量模型成果已成功保存至本地文件夹：'{SAVE_DIR}'")
print(f"1. 底座模型位于: {os.path.join(SAVE_DIR, 'bge_encoder')}")
print(f"2. 6路串联分类头位于: {os.path.join(SAVE_DIR, 'classification_heads.pkl')}")

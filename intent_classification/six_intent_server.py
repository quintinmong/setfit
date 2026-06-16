import sys

# ================= 猴子补丁：防止 transformers 报错 =================
import transformers.training_args
if not hasattr(transformers.training_args, "default_logdir"):
    transformers.training_args.default_logdir = lambda: "runs"
# ===================================================================

import os
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# 路径加固
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from temporal_signal.temporal_constants import INTENT_HEAD_NAMES, INTENT_LABEL_MAPS
from shared_encoder import ENCODER_ARTIFACT_DIR, MacBertEncoder, resolve_encoder_path

# 1. 使用方案指定的本地底座（路径加固：兼容 git worktree 环境）
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

_main_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(root_dir))) if ".claude/worktrees" in root_dir else root_dir
MODEL_PATH = resolve_encoder_path(root_dir)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "找不到 MacBERT 底座！请先运行 python3 intent_classification/download_model.py。"
    )

print(f"正在初始化共享冻结 MacBERT 底座: {MODEL_PATH} ...")
encoder = MacBertEncoder(MODEL_PATH, device=device)

# 2. 60条重标注数据集（新6维标签体系）
# 字段说明：
#   sc  = scene (0=咨询响应, 1=情绪抱怨, 2=主动服务/关系维护)
#   st  = sub_type (0=操作类咨询, 1=产品/利率咨询, 2=宏观/深度分析)
#   el  = emotion_label (0=平静, 1=愤怒, 2=焦虑, 3=不信任)
#   is_ = info_source (0=无, 1=竞品/同业, 2=网络/媒体, 3=自身经验)
#   sk_t = skill_trigger (0=无, 1=SK-COMP, 2=SK-EMPATHY, 3=SK-EDUCATE, 4=SK-OPERATE)
#   il  = intent_label (0=普通咨询, 1=异议-收益, 2=异议-安全, 3=促单-高意向, 4=流失风险)
train_data = [
    # ==================== 【场景：咨询响应 (sc=0)】 ====================
    # 1. "请问手机银行在哪里可以买理财？"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0(操作类)
    # el: em_old=0, tr_old=0 → 0(平静)
    # is_: 无竞品/网络/经验 → 0
    # sk_t: sk_old=1→4(SK-OPERATE); 无竞品,无情绪,th_old=0 → 4
    # il: 无流失/促单/异议词 → 0
    {"text": "请问手机银行在哪里可以买理财？",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 2. "我想看看民生银行最新的定期存款利率"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: sk_old=0,th_old=0 → 1(产品/利率)
    # el: 0 → 0
    # is_: 无 → 0
    # sk_t: sk_old=0,无竞品,无情绪,th_old=0 → 0
    # il: 0
    {"text": "我想看看民生银行最新的定期存款利率",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 3. "找不到理财购买页面了，帮我看一下，急死人了"
    # sc=0,sk_old=1,th_old=0,em_old=2,tr_old=0
    # st: sk_old=1 AND sc=0 → 0(操作类)
    # el: em_old=2 → 2(焦虑)
    # is_: 无 → 0
    # sk_t: sk_old=1→4; 无竞品; em_old=2但不是em=1(愤怒); th_old=0 → 4(SK-OPERATE)
    # il: 0
    {"text": "找不到理财购买页面了，帮我看一下，急死人了",
     "sc": 0, "st": 0, "el": 2, "is_": 0, "sk_t": 4, "il": 0},

    # 4. "手机上怎么操作提前支取大额存单啊？"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: sk_old=1→4 → 4
    # il: 0
    {"text": "手机上怎么操作提前支取大额存单啊？",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 5. "小张，我想问下这款中低风险的产品保本吗？"
    # sc=0,sk_old=0,th_old=0,em_old=2,tr_old=1
    # st: sk_old=0,th_old=0 → 1
    # el: em_old=2,tr_old=1 but em≠0 → 2(焦虑保留)
    # is_: 无 → 0
    # sk_t: sk_old=0,无竞品,em_old=2(not 1),th_old=0 → 0
    # il: "保本"不属于关键词 → 0
    {"text": "小张，我想问下这款中低风险的产品保本吗？",
     "sc": 0, "st": 1, "el": 2, "is_": 0, "sk_t": 0, "il": 0},

    # 6. "你们网点那个安心存产品，每天几点开始放额度？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "你们网点那个安心存产品，每天几点开始放额度？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 7. "合同里写的这个管理费和托管费是怎么扣的？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "合同里写的这个管理费和托管费是怎么扣的？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 8. "怎么看我买的基金今天收益是多少？在哪点进去？"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: 4
    # il: 0
    {"text": "怎么看我买的基金今天收益是多少？在哪点进去？",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 9. "这几款天天盈理财有什么区别吗？哪个随时能取？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "这几款天天盈理财有什么区别吗？哪个随时能取？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 10. "你们这儿的外币存款，美元一年期利率现在是多少？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "你们这儿的外币存款，美元一年期利率现在是多少？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 11. "我这岁数大了眼花，APP上找不到风险评估在哪里做。"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: 4
    # il: 0
    {"text": "我这岁数大了眼花，APP上找不到风险评估在哪里做。",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 12. "请问如果我要开个存款证明，必须去网点吗？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "请问如果我要开个存款证明，必须去网点吗？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 13. "转账限额怎么修改？我想把单笔限额调高到十万。"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: 4
    # il: 0
    {"text": "转账限额怎么修改？我想把单笔限额调高到十万。",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 14. "这个储蓄国债 and 电子式国债有什么不一样吗？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "这个储蓄国债 and 电子式国债有什么不一样吗？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 15. "小张，今天发行的那款专享理财，编码是多少啊？"
    # sc=0,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "小张，今天发行的那款专享理财，编码是多少啊？",
     "sc": 0, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # ==================== 【场景：情绪抱怨 (sc=1)】 ====================
    # 16. "现在的理财收益也太低了吧，真差劲"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: sk_old=0,th_old=0 → 1
    # el: em_old=1 → 1(愤怒,不满足tr覆盖条件因em≠0)
    # is_: 无竞品/网络/经验 → 0 (含"以前"?→"比以前"含有经验感, 但规则是"之前/上次/我买过/用过/以前")
    #   文本:"现在的理财收益也太低了吧，真差劲" → 无规则词 → 0
    # sk_t: sk_old=0,无竞品,em_old=1(愤怒)且无竞品→2(SK-EMPATHY)
    # il: "差劲"→含"差"→tr_old=1→1(异议-收益)
    {"text": "现在的理财收益也太低了吧，真差劲",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 17. "怎么天天跌啊，感觉比以前差多了！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: em_old=1 → 1
    # is_: "以前" → 3(自身经验)
    # sk_t: sk_old=0,无竞品,em_old=1(愤怒)→2(SK-EMPATHY)
    # il: "差" + tr_old=1 → 1(异议-收益)
    {"text": "怎么天天跌啊，感觉比以前差多了！",
     "sc": 1, "st": 1, "el": 1, "is_": 3, "sk_t": 2, "il": 1},

    # 18. "你们这个APP太难用了，经常卡顿闪退"
    # sc=1,sk_old=1,th_old=0,em_old=1,tr_old=0
    # st: sk_old=1 but sc=1≠0 → th_old=0 → 1(产品/利率)
    # el: em_old=1 → 1
    # is_: 无 → 0
    # sk_t: sk_old=1→4基; 无竞品,em_old=1(愤怒)且无竞品→2(SK-EMPATHY)覆盖
    # il: 无流失/促单/异议词 → 0
    {"text": "你们这个APP太难用了，经常卡顿闪退",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 0},

    # 19. "气死我了，今天怎么又亏了这么多钱，搞什么啊"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em_old=1(愤怒),无竞品 → 2(SK-EMPATHY)
    # il: "亏"但无竞品对比,tr_old=1 → 1(异议-收益)? "亏"含在"低/差/不给力/亏/竞品收益更高"中→是→1
    {"text": "气死我了，今天怎么又亏了这么多钱，搞什么啊",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 20. "垃圾银行，体验太差了，我要把钱全转走销户！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em_old=1,无竞品 → 2
    # il: "销户"/"转走" → 4(流失风险)优先
    {"text": "垃圾银行，体验太差了，我要把钱全转走销户！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 4},

    # 21. "收益才3.2%，感觉比以前差多了"
    # sc=1,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2(宏观/深度)
    # el: em_old=0,tr_old=0 → 0
    # is_: "以前" → 3
    # sk_t: sk_old=0,无竞品,em_old=0(不是愤怒),th_old=1→3(SK-EDUCATE)
    # il: "差" but tr_old=0 → 规则是"低/差/不给力/亏/竞品收益更高 and tr_old=1" → tr_old=0所以不触发1 → 0
    {"text": "收益才3.2%，感觉比以前差多了",
     "sc": 1, "st": 2, "el": 0, "is_": 3, "sk_t": 3, "il": 0},

    # 22. "招行收益好像更高，有点动摇了"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2(焦虑保留)
    # is_: "招行" → 1(竞品/同业)
    # sk_t: 含竞品("招行") AND tr_old=1 → 1(SK-COMP)优先
    # il: "低"+"竞品对比"+"tr_old=1" → 1(异议-收益)
    {"text": "招行收益好像更高，有点动摇了",
     "sc": 1, "st": 2, "el": 2, "is_": 1, "sk_t": 1, "il": 1},

    # 23. "我看网上说现在要发生金融危机了，理财还安全吗？"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: "网上" → 2(网络/媒体)
    # sk_t: 无竞品; em_old=2(not 1,不是愤怒); th_old=1→3(SK-EDUCATE)?
    #   规则：先检竞品+tr→1; 再检em=1无竞品→2; 再检th=1无竞无情绪怒→3
    #   em=2不是1,无竞品,th=1→3
    # il: "金融危机"→"异议-安全" → 2
    {"text": "我看网上说现在要发生金融危机了，理财还安全吗？",
     "sc": 1, "st": 2, "el": 2, "is_": 2, "sk_t": 3, "il": 2},

    # 24. "隔壁建设银行天天给我推4.0的利息，你们怎么这么低？"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: "建行"(建设银行)→ 1
    # sk_t: 含竞品("建设银行") AND tr_old=1 → 1(SK-COMP)
    # il: "低"+"tr_old=1" → 1(异议-收益)
    {"text": "隔壁建设银行天天给我推4.0的利息，你们怎么这么低？",
     "sc": 1, "st": 2, "el": 2, "is_": 1, "sk_t": 1, "il": 1},

    # 25. "买的时候说是稳健型，结果绿成这样，你们骗人吧！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em_old=1,无竞品 → 2(SK-EMPATHY)
    # il: "骗人" → 2(异议-安全)
    {"text": "买的时候说是稳健型，结果绿成这样，你们骗人吧！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 2},

    # 26. "坑人啊，说好年化回报挺高的，这都几个月了全是负的！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em_old=1,无竞品 → 2
    # il: "亏"(全是负的包含亏损意) + tr_old=1 → 1(异议-收益)
    {"text": "坑人啊，说好年化回报挺高的，这都几个月了全是负的！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 27. "微众银行的活期都比你们定期强，真后悔放你们这。"
    # sc=1,sk_old=0,th_old=1,em_old=1,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=1 → 1
    # is_: "微众" → 1(竞品)
    # sk_t: 含竞品("微众银行") AND tr_old=1 → 1(SK-COMP)
    # il: "低"(含义,活期>定期反向)+"tr_old=1" → 1(异议-收益)
    {"text": "微众银行的活期都比你们定期强，真后悔放你们这。",
     "sc": 1, "st": 2, "el": 1, "is_": 1, "sk_t": 1, "il": 1},

    # 28. "听说最近国债都要爆雷了？现在信托也不行，到底能信谁？"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: "听说"→类似"网络/媒体" → 2
    # sk_t: 无竞品,em=2(not 1),th=1→3(SK-EDUCATE)
    # il: "爆雷" → 2(异议-安全)
    {"text": "听说最近国债都要爆雷了？现在信托也不行，到底能信谁？",
     "sc": 1, "st": 2, "el": 2, "is_": 2, "sk_t": 3, "il": 2},

    # 29. "你们客户经理之前推荐的时候拍胸脯，现在亏了就找借口！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: "之前" → 3(自身经验)
    # sk_t: em_old=1,无竞品 → 2(SK-EMPATHY)
    # il: "亏" + tr_old=1 → 1(异议-收益)
    {"text": "你们客户经理之前推荐的时候拍胸脯，现在亏了就找借口！",
     "sc": 1, "st": 1, "el": 1, "is_": 3, "sk_t": 2, "il": 1},

    # 30. "为什么别人家的APP操作那么傻瓜，你们的这么绕？"
    # sc=1,sk_old=1,th_old=0,em_old=1,tr_old=0
    # st: sc=1≠0所以sk_old=1 AND sc=0不满足; th_old=0 → 1
    # el: em_old=1 → 1
    # is_: 无 → 0
    # sk_t: em_old=1,无竞品 → 2(SK-EMPATHY)
    # il: 无 → 0
    {"text": "为什么别人家的APP操作那么傻瓜，你们的这么绕？",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 0},

    # ==================== 【场景：主动服务/关系维护 (sc=2)】 ====================
    # 31. "谢谢小张经理，你的建议很专业"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "谢谢小张经理，你的建议很专业",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 32. "早啊，今天有什么新产品推荐吗？"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "早啊，今天有什么新产品推荐吗？",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 33. "节日祝福收到了，也祝你节日快乐"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "节日祝福收到了，也祝你节日快乐",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 34. "行，那我下周有空去网点找你做个面签吧"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: "面签" → 3(促单-高意向)
    {"text": "行，那我下周有空去网点找你做个面签吧",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 3},

    # 35. "最近确实比较忙，等我忙完这段时间再找你配置"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: "配置" → 3(促单-高意向)
    {"text": "最近确实比较忙，等我忙完这段时间再找你配置",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 3},

    # 36. "你们民生银行的服务态度确实没得说，挺体贴的"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "你们民生银行的服务态度确实没得说，挺体贴的",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 37. "最近股市震荡成这样，我总觉得心里不踏实，钱放哪好？"
    # sc=2,sk_old=0,th_old=1,em_old=2,tr_old=0
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=0 → 2
    # is_: 无 → 0
    # sk_t: 无竞品,em=2(not 1),th=1→3(SK-EDUCATE)
    # il: 0
    {"text": "最近股市震荡成这样，我总觉得心里不踏实，钱放哪好？",
     "sc": 2, "st": 2, "el": 2, "is_": 0, "sk_t": 3, "il": 0},

    # 38. "手头刚下来一笔工程款，你帮我做个全面的资产配置规划？"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: 无 → 0
    # sk_t: 无竞品,em=0(not 1),th=1→3(SK-EDUCATE)
    # il: "配置" → 3(促单-高意向)
    {"text": "手头刚下来一笔工程款，你帮我做个全面的资产配置规划？",
     "sc": 2, "st": 2, "el": 0, "is_": 0, "sk_t": 3, "il": 3},

    # 39. "我想给我家孩子存笔教育金，不知道怎么规划长线稳定"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: 无 → 0
    # sk_t: th=1,无竞品,em=0 → 3(SK-EDUCATE)
    # il: "规划"→含"规划"类似"配置" → 3(促单-高意向)
    {"text": "我想给我家孩子存笔教育金，不知道怎么规划长线稳定",
     "sc": 2, "st": 2, "el": 0, "is_": 0, "sk_t": 3, "il": 3},

    # 40. "小张，听说最近国债利率又降了？对我们持有有影响吗？"
    # sc=2,sk_old=0,th_old=1,em_old=2,tr_old=0
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=0 → 2
    # is_: "听说"→2(网络/媒体)
    # sk_t: 无竞品,em=2(not 1),th=1→3(SK-EDUCATE)
    # il: 0
    {"text": "小张，听说最近国债利率又降了？对我们持有有影响吗？",
     "sc": 2, "st": 2, "el": 2, "is_": 2, "sk_t": 3, "il": 0},

    # 41. "收到你的降息提醒了，那这期到期后我换成纯债基金行吗？"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: 无 → 0
    # sk_t: th=1,无竞品,em=0 → 3
    # il: 0
    {"text": "收到你的降息提醒了，那这期到期后我换成纯债基金行吗？",
     "sc": 2, "st": 2, "el": 0, "is_": 0, "sk_t": 3, "il": 0},

    # 42. "我准备给父母留一笔养老钱，追求绝对安全，推荐个方案。"
    # sc=2,sk_old=0,th_old=1,em_old=2,tr_old=0
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=0 → 2
    # is_: 无 → 0
    # sk_t: th=1,无竞品,em=2(not 1) → 3(SK-EDUCATE)
    # il: "方案"类似"配置" → 3(促单-高意向)
    {"text": "我准备给父母留一笔养老钱，追求绝对安全，推荐个方案。",
     "sc": 2, "st": 2, "el": 2, "is_": 0, "sk_t": 3, "il": 3},

    # 43. "没关系，刚好路过你们网点，顺便进来喝杯水看看你。"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: 无 → 0
    # sk_t: 0
    # il: 0
    {"text": "没关系，刚好路过你们网点，顺便进来喝杯水看看你。",
     "sc": 2, "st": 1, "el": 0, "is_": 0, "sk_t": 0, "il": 0},

    # 44. "你上次说的那款年金保险，计划书再发我研究一下。"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: "上次" → 3(自身经验)
    # sk_t: 0
    # il: "计划书" → 3(促单-高意向)
    {"text": "你上次说的那款年金保险，计划书再发我研究一下。",
     "sc": 2, "st": 1, "el": 0, "is_": 3, "sk_t": 0, "il": 3},

    # 45. "听说有专属的老客回馈活动？我能参加吗？"
    # sc=2,sk_old=0,th_old=0,em_old=0,tr_old=0
    # st: 1
    # el: 0
    # is_: "听说"→2
    # sk_t: 0
    # il: 0
    {"text": "听说有专属的老客回馈活动？我能参加吗？",
     "sc": 2, "st": 1, "el": 0, "is_": 2, "sk_t": 0, "il": 0},

    # ==================== 【极高对抗度追加混合集】 ====================
    # 46. "网上说现在有很多银行爆雷，你们民生银行靠得住吗？"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: "网上" → 2
    # sk_t: 无竞品,em=2(not 1),th=1→3(SK-EDUCATE)
    # il: "爆雷" → 2(异议-安全)
    {"text": "网上说现在有很多银行爆雷，你们民生银行靠得住吗？",
     "sc": 1, "st": 2, "el": 2, "is_": 2, "sk_t": 3, "il": 2},

    # 47. "我买了一百多万的稳健理财结果还亏，你们要给我个解释！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em=1,无竞品 → 2(SK-EMPATHY)
    # il: "亏"+tr_old=1 → 1(异议-收益)
    {"text": "我买了一百多万的稳健理财结果还亏，你们要给我个解释！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 48. "钱放网商银行安全还是放你们大额存单安全？说真心话。"
    # sc=0,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: "网商" → 1(竞品)
    # sk_t: 含竞品("网商银行") AND tr_old=1 → 1(SK-COMP)
    # il: "安全"但无"爆雷/骗人/金融危机/不安全/靠得住"关键词 → 0
    {"text": "钱放网商银行安全还是放你们大额存单安全？说真心话。",
     "sc": 0, "st": 2, "el": 2, "is_": 1, "sk_t": 1, "il": 0},

    # 49. "怎么购买理财还要录音录像这么麻烦？老太婆操作不来！"
    # sc=1,sk_old=1,th_old=0,em_old=1,tr_old=0
    # st: sc=1≠0;th_old=0 → 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em=1,无竞品 → 2(SK-EMPATHY)
    # il: 0
    {"text": "怎么购买理财还要录音录像这么麻烦？老太婆操作不来！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 0},

    # 50. "换了新手机，登录APP提示密码锁定了，现在怎么办？"
    # sc=0,sk_old=1,th_old=0,em_old=2,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: em_old=2,tr_old=0 → 2
    # is_: 无 → 0
    # sk_t: sk_old=1→4; 无竞品,em=2(not 1),th=0 → 4(SK-OPERATE)
    # il: 0
    {"text": "换了新手机，登录APP提示密码锁定了，现在怎么办？",
     "sc": 0, "st": 0, "el": 2, "is_": 0, "sk_t": 4, "il": 0},

    # 51. "把钱放你们这我心里直打鼓，现在利息还这么低，我想转走。"
    # sc=1,sk_old=0,th_old=1,em_old=2,tr_old=1
    # st: th_old=1 → 2
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: 无 → 0
    # sk_t: 无竞品,em=2(not 1),th=1→3(SK-EDUCATE)
    # il: "转走" → 4(流失风险)
    {"text": "把钱放你们这我心里直打鼓，现在利息还这么低，我想转走。",
     "sc": 1, "st": 2, "el": 2, "is_": 0, "sk_t": 3, "il": 4},

    # 52. "你们这大额存单要三十万起购？手机上怎么凑单买？"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: 4
    # il: 0
    {"text": "你们这大额存单要三十万起购？手机上怎么凑单买？",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 53. "现在市场全在降息，买长期定期划算还是买年金保险划算？"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: 无 → 0
    # sk_t: th=1,无竞品,em=0 → 3(SK-EDUCATE)
    # il: 0
    {"text": "现在市场全在降息，买长期定期划算还是买年金保险划算？",
     "sc": 2, "st": 2, "el": 0, "is_": 0, "sk_t": 3, "il": 0},

    # 54. "别天天发微信推销了，烦不烦，有高收益产品再找我！"
    # sc=1,sk_old=0,th_old=0,em_old=1,tr_old=1
    # st: 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em=1,无竞品 → 2
    # il: "低"(隐含拒绝低收益)+tr_old=1 → 1(异议-收益)
    {"text": "别天天发微信推销了，烦不烦，有高收益产品再找我！",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 55. "我卡里还有二十万活期，你看看帮我怎么规划能抗通胀？"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: 无 → 0
    # sk_t: th=1,无竞品,em=0 → 3
    # il: "规划" → 3(促单-高意向)
    {"text": "我卡里还有二十万活期，你看看帮我怎么规划能抗通胀？",
     "sc": 2, "st": 2, "el": 0, "is_": 0, "sk_t": 3, "il": 3},

    # 56. "手机验证码一直收不到，你们系统是不是瘫痪了啊？"
    # sc=1,sk_old=1,th_old=0,em_old=1,tr_old=0
    # st: sc=1≠0;th_old=0 → 1
    # el: 1
    # is_: 无 → 0
    # sk_t: em=1,无竞品 → 2
    # il: 0
    {"text": "手机验证码一直收不到，你们系统是不是瘫痪了啊？",
     "sc": 1, "st": 1, "el": 1, "is_": 0, "sk_t": 2, "il": 0},

    # 57. "我看别家收益都挺高，你们产品经理是不是水平不行啊？"
    # sc=1,sk_old=0,th_old=1,em_old=1,tr_old=1
    # st: th_old=1 → 2
    # el: 1
    # is_: "别家"→隐含竞品但无"招行/建行/微众/网商/招商"具体名称 → 0
    # sk_t: 无具体竞品名,em=1,无竞品→2(SK-EMPATHY)
    # il: "低/差"+"tr_old=1" → "别家收益都挺高"隐含"你们低"+tr=1 → 1
    {"text": "我看别家收益都挺高，你们产品经理是不是水平不行啊？",
     "sc": 1, "st": 2, "el": 1, "is_": 0, "sk_t": 2, "il": 1},

    # 58. "小张，刚看新闻降准了，对我们小微企业贷款有好处吗？"
    # sc=2,sk_old=0,th_old=1,em_old=0,tr_old=0
    # st: th_old=1 → 2
    # el: 0
    # is_: "新闻" → 2(网络/媒体)
    # sk_t: th=1,无竞品,em=0 → 3(SK-EDUCATE)
    # il: 0
    {"text": "小张，刚看新闻降准了，对我们小微企业贷款有好处吗？",
     "sc": 2, "st": 2, "el": 0, "is_": 2, "sk_t": 3, "il": 0},

    # 59. "我年纪大了记不住密码，能不能改成手机刷脸登录？"
    # sc=0,sk_old=1,th_old=0,em_old=0,tr_old=0
    # st: sk_old=1 AND sc=0 → 0
    # el: 0
    # is_: 无 → 0
    # sk_t: 4
    # il: 0
    {"text": "我年纪大了记不住密码，能不能改成手机刷脸登录？",
     "sc": 0, "st": 0, "el": 0, "is_": 0, "sk_t": 4, "il": 0},

    # 60. "你们这保本的玩意儿一个都没了？那本金亏了你们赔吗？"
    # sc=1,sk_old=0,th_old=0,em_old=2,tr_old=1
    # st: 1
    # el: em_old=2,tr_old=1 but em≠0 → 2
    # is_: 无 → 0
    # sk_t: em=2(not 1),无竞品,th=0 → 0(none triggered)
    # il: "亏"+tr_old=1 → 1(异议-收益)
    {"text": "你们这保本的玩意儿一个都没了？那本金亏了你们赔吗？",
     "sc": 1, "st": 1, "el": 2, "is_": 0, "sk_t": 0, "il": 1},
]

train_texts = [item["text"] for item in train_data]
y_sc = [item["sc"] for item in train_data]
y_st = [item["st"] for item in train_data]
y_el = [item["el"] for item in train_data]
y_is = [item["is_"] for item in train_data]
y_sk = [item["sk_t"] for item in train_data]
y_il = [item["il"] for item in train_data]

# 3. 底座统一抽取高维语义特征（只进行一次，共享编码）
print(f"核心优化：底座对 {len(train_texts)} 条样本进行特征提取...")
X_all = encoder.encode(train_texts, show_progress_bar=False)

# 4. 80/20 划分（所有头共用同一次切分，保持样本一致性）
print("正在进行 80/20 训练/测试集划分...")
from sklearn.model_selection import train_test_split
indices = list(range(len(train_texts)))
idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X_all[idx_train]
X_test = X_all[idx_test]


def get_labels(y_list, idx):
    return [y_list[i] for i in idx]


# 5. 并行训练六路完全解耦的分类头，并逐一评估
print("正在并行训练 6 个独立的轻量分类头...\n")

# 保存目录：在 git worktree 环境下回退到主仓根目录
_save_root = _main_repo_root if ".claude/worktrees" in root_dir else root_dir
SAVE_DIR = os.path.abspath(os.path.join(_save_root, "my_final_six_intents_model"))
os.makedirs(SAVE_DIR, exist_ok=True)

head_params = {"class_weight": "balanced", "max_iter": 1000}


def train_and_evaluate(y_all, head_name):
    y_train = get_labels(y_all, idx_train)
    y_test = get_labels(y_all, idx_test)
    head = LogisticRegression(**head_params).fit(X_train, y_train)
    y_pred = head.predict(X_test)
    n_cls = len(INTENT_LABEL_MAPS[head_name])
    target_names = [INTENT_LABEL_MAPS[head_name][i] for i in range(n_cls)]
    print(f"  [{head_name}] 测试集分布: {dict(sorted(Counter(y_test).items()))}")
    print(classification_report(
        y_test, y_pred,
        labels=list(range(n_cls)),
        target_names=target_names,
        zero_division=0
    ))
    return head


head_sc = train_and_evaluate(y_sc, "scene")
head_st = train_and_evaluate(y_st, "sub_type")
head_el = train_and_evaluate(y_el, "emotion_label")
head_is = train_and_evaluate(y_is, "info_source")
head_sk = train_and_evaluate(y_sk, "skill_trigger")
head_il = train_and_evaluate(y_il, "intent_label")

print("六路并行分类头训练完成！")

# 6. 保存 v2 格式 artifact
encoder.save(os.path.join(SAVE_DIR, ENCODER_ARTIFACT_DIR))

artifact = {
    "heads": {
        "scene": head_sc,
        "sub_type": head_st,
        "emotion_label": head_el,
        "info_source": head_is,
        "skill_trigger": head_sk,
        "intent_label": head_il,
    },
    "label_maps": INTENT_LABEL_MAPS,
    "version": 2
}
joblib.dump(artifact, os.path.join(SAVE_DIR, "classification_heads.pkl"))

print(f"\n全量模型成果已成功保存至本地文件夹：'{SAVE_DIR}'")
print(f"1. 底座模型位于: {os.path.join(SAVE_DIR, ENCODER_ARTIFACT_DIR)}")
print(f"2. 6路分类头(v2格式)位于: {os.path.join(SAVE_DIR, 'classification_heads.pkl')}")

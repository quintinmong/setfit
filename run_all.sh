#!/bin/bash

# 遇错即停，确保 pipeline 的健壮性
set -e

# 定义彩色 Banner 提示
print_banner() {
    echo -e "\n==================================================================="
    echo -e "🪐 $1"
    echo -e "===================================================================\n"
}

# 1. 自动定位 Python 解释器
print_banner "步骤 0: 探测 Python 解释器与虚拟环境"
if [ -d ".venv312" ]; then
    PYTHON_BIN=".venv312/bin/python3"
    echo "✅ 成功匹配预置虚拟环境: .venv312"
elif [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python3"
    echo "✅ 成功匹配通用虚拟环境: .venv"
else
    PYTHON_BIN="python3"
    echo "⚠️ 未找到局部虚拟环境目录，将降级调用全局: python3"
fi

# 确认 python 版本
$PYTHON_BIN --version

# 2. 下载共享基础底座模型
print_banner "步骤 1: 启动共享语义向量底座模型下载 (BGE-small)"
$PYTHON_BIN intent_classification/download_model.py

# 3. 微调底座并训练 6 个机器学习分类头
print_banner "步骤 2: 启动六路独立意图分类器联合微调与拟合"
$PYTHON_BIN intent_classification/six_intent_server.py

# 4. 训练多轮 GRU 时序心智分类器
print_banner "步骤 3: 启动三轮时序窗口 GRU 神经网络模型训练"
$PYTHON_BIN temporal_signal/train_temporal_model.py

# 5. 闭环执行大脑路由进行推理验证
print_banner "步骤 4: 启动终极大脑总路由集成推理测试 (6维+时序+NER)"
$PYTHON_BIN agent_router.py

print_banner "🎉 恭喜！数字分身 Agent 训练流水线一键跑通，模型资产已全部固化就绪！"

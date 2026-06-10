# Flappy Bird Q-Learning

基于表格型 Q-learning 的 Flappy Bird 智能体。

## 环境要求

使用 conda 环境 `alg`：

```bash
conda activate alg
```

主要依赖：Python 3.10, gymnasium, flappy-bird-gymnasium, pygame, numpy, torch

## 训练

```bash
python scripts/train.py <model.pkl> [参数...]
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `model` | 位置参数 | 必填 | 训练完成后模型的保存路径 (.pkl) |
| `--resume` | str | — | 加载已有模型继续训练 (.pkl) |
| `--state-mode` | str | `enhanced` | 状态表示 |
| `--reward-mode` | str | `death_penalty` | 奖励函数 |
| `--alpha` | float | 0.7 | 学习率 |
| `--gamma` | float | 0.95 | 折扣因子 |
| `--epsilon` | float | 0.0 | 探索率 |
| `--iterations` | int | 50000 | 训练局数 |
| `--obs-mul-factor` | int | 30 | 观测值离散化乘数 |
| `--seed` | int | 42 | 随机种子 |
| `--death-penalty` | int | -1000 | 死亡惩罚值 |
| `--max-steps` | int | — | 单局步数上限 |
| `--progress-interval` | int | 1000 | 进度打印间隔（0 = 不打印） |

### `--state-mode` 可选值

| 值 | 特征数 | 特征说明 |
|------|------|------|
| `example` | 3 | 水平距离、到下管距离、垂直速度 |
| `enhanced` | 4 | 水平距离、到缺口中心距离、到下管距离、垂直速度 |
| `enhanced_top` | 5 | enhanced + 到上管距离 |
| `bonus` | 6 | horizontal distance, gap center, top pipe, bottom pipe, velocity, rotation |
| `lookahead` | 5 | enhanced + 下一管道缺口中心距离 |

### `--reward-mode` 可选值

| 值 | 说明 |
|------|------|
| `death_penalty` | 基础奖励：存活 +0.1，过管 +1.0，死亡按 `--death-penalty` 惩罚 |
| `shaped` | 塑形奖励：增加中心接近奖励、速度惩罚、flap 惩罚、触顶惩罚 |
| `shaped_v2` | 塑形奖励 v2：加大中心奖励和触顶惩罚权重 |

### 示例

```bash
# 从头训练
python scripts/train.py models/baseline.pkl \
  --state-mode enhanced --reward-mode death_penalty --iterations 100000

# 加载已有模型继续训练
python scripts/train.py models/improved.pkl \
  --resume models/baseline.pkl --iterations 50000

# bonus1：enhanced_top + shaped_v2
python scripts/train.py models/bonus1.pkl \
  --state-mode enhanced_top --reward-mode shaped_v2 --iterations 100000

# 无进度打印的快速训练
python scripts/train.py test.pkl --iterations 1000 --progress-interval 0
```

## 评估

```bash
python scripts/eval_unbounded.py <model.pkl> [参数...]
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `model` | 位置参数 | 必填 | 待评估的 .pkl 模型文件路径 |
| `--episodes` | int | 50 | 评估局数 |
| `--seed-base` | int | 2026 | 种子基准值 |
| `--obs-mul-factor` | int | 30 | 观测值离散化乘数 |
| `--state-mode` | str | `enhanced` | 状态表示（同上表） |
| `--render` | flag | 关闭 | 开启可视化渲染 |
| `--json-output` | str | — | 评估结果保存为 JSON 文件 |

### 示例

```bash
# 无头评估（50 局）
python scripts/eval_unbounded.py q_best.pkl --episodes 50

# 评估 bonus1 模型（需要指定 state_mode）
python scripts/eval_unbounded.py q_bonus1_best.pkl \
  --state-mode enhanced_top --episodes 50

# 可视化查看（3 局）
python scripts/eval_unbounded.py q_best.pkl --render --episodes 3

# 评估结果输出 JSON
python scripts/eval_unbounded.py q_best.pkl --json-output results/result.json
```

## 可视化

```bash
# 直接观看默认最佳模型
python src/watch_best.py

# 指定模型和局数
python src/watch_best.py --model q_combined_best.pkl --state-mode enhanced_top --episodes 10
```

## 已有模型

| 文件 | state_mode | 说明 |
|------|------|------|
| `q_best.pkl` | enhanced | 基础参数搜索最佳模型 |
| `q_bonus1_best.pkl` | enhanced_top | bonus1 最佳（enhanced_top + death_penalty） |
| `q_bonus_best.pkl` | enhanced | bonus2 最佳（enhanced + shaped_v2） |
| `q_combined_best.pkl` | enhanced_top | 组合最佳（enhanced_top + shaped_v2） |

`results/models/` 中保存了所有实验训练的中间模型。

## 复现实验

基础参数搜索：

```bash
python src/experiment.py --executor process --workers 6
```

bonus 状态和奖励实验：

```bash
python src/experiment.py \
  --grid bonus \
  --executor process \
  --workers 6 \
  --results-dir results_bonus \
  --best-model-path q_bonus_best.pkl
```

生成报告图表：

```bash
python scripts/generate_report_charts.py
```

## 文件结构

```text
.
├── README.md
├── q_best.pkl              # 基础最佳模型
├── q_bonus1_best.pkl       # bonus1 最佳模型
├── q_bonus_best.pkl        # bonus2 最佳模型
├── q_combined_best.pkl     # 组合最佳模型
├── assets/                 # 报告图片
├── results/                # 基础实验数据
├── results_bonus/          # bonus 实验数据
├── scripts/
│   ├── train.py            # 训练脚本
│   ├── eval_unbounded.py   # 评估脚本
│   ├── generate_report_charts.py
│   ├── train_bonus_final.py
│   ├── train_bonus_models.py
│   ├── train_bonus_all.py
│   ├── train_shaped_v2.py
│   └── update_results.py
└── src/
    ├── q_learning.py       # Q-learning 核心实现
    ├── experiment.py       # 批量实验框架
    ├── watch_best.py       # 可视化观看
    ├── train_ai_or_play.py # 原始训练/游玩入口
    └── human_play.py       # 人类试玩
```

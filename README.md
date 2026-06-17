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

## 已有模型

| 文件 | state_mode | 说明 |
|------|------|------|
| `q_base_best.pkl` | enhanced | 基础参数搜索最佳模型 |
| `q_bonus1_best.pkl` | enhanced_top | bonus1 最佳（enhanced_top + death_penalty） |
| `q_bonus_best.pkl` | enhanced | bonus2 最佳（enhanced + shaped_v2） |
| `q_combined_best.pkl` | enhanced_top | 组合最佳（enhanced_top + shaped_v2） |

## 对最佳模型的观察

无画面
```
python scripts/eval_unbounded.py q_best.pkl --state-mode enhanced_top --episodes 5
```
有画面
```
python scripts/eval_unbounded.py q_best.pkl --state-mode enhanced_top --render --episodes 5
```

## 使用 src/ 中的代码

`src/` 提供了另一种更简洁的使用方式：

### 训练

```bash
python src/train_ai_or_play.py --train
```

训练参数（`alpha`、`gamma`、`epsilon`、`iteration`）在 `src/train_ai_or_play.py` 中直接修改。模型自动保存为 `q_MMDD_HHMMSS.pkl`。

### 评估 / 运行 AI

```bash
python src/train_ai_or_play.py --no-train
```

默认加载 `q_best.pkl`，如需更换模型，修改脚本第 32 行的 `path` 变量。

### 人类试玩

```bash
python src/human_play.py
```

### 编程调用

`src/q_learning.py` 中的 `train()` 和 `play()` 函数可直接在 Python 中调用，支持 `state_mode`、`reward_mode` 等完整参数。

```python
from src.q_learning import train, play

ai = train(iteration=50000, alpha=0.7, gamma=0.95, epsilon=0,
           state_mode="enhanced_top", reward_mode="shaped_v2")
ai.save_q("my_model.pkl")

play(ai, episodes=5, state_mode="enhanced_top")
```


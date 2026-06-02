# Flappy Bird Q-learning Final Project

本项目为 Flappy Bird 设计并训练了一个基于表格型 Q-learning 的智能体。代码完成了作业必做部分，并实现了 bonus1 状态表示优化、bonus2 reward shaping，对训练参数进行了多组对比实验，最终保存了最佳模型和实验报告。

![最佳模型游戏画面](assets/294score.png)

## 项目亮点

- 使用 `flappy-bird-gymnasium` 环境训练 Flappy Bird 智能体。
- 使用 Q-learning，而不是 DQN 或其他深度强化学习方法。
- 完成 `GameAI` 的 Q 表查询、更新、epsilon-greedy 动作选择。
- 设计并对比三种状态表示：`example`、`enhanced`、`bonus`。
- 设计并对比两种奖励方式：`death_penalty`、`shaped`。
- 使用多 worker 并行参数搜索，并用 CUDA/PyTorch 做统计与排序。
- 保存最佳模型 `q_best.pkl`，bonus 对比模型 `q_bonus_best.pkl`。
- 提供图文报告 `report.md` 和可视化图表。

## 文件结构

```text
.
├── README.md
├── report.md
├── q_best.pkl
├── q_bonus_best.pkl
├── assets/
│   ├── 294score.png
│   ├── result.png
│   └── charts/
│       ├── base_screen_top12.png
│       ├── base_retrain.png
│       ├── best_scores_line.png
│       ├── bonus_screen.png
│       ├── bonus_retrain.png
│       └── state_reward_key_compare.png
├── docs/
│   ├── flappy_bird_assignment.md
│   └── flappy_bird_course_notes.md
├── results/
│   ├── experiment_results.csv
│   ├── best_summary.json
│   └── models/
├── results_bonus/
│   ├── experiment_results.csv
│   ├── best_summary.json
│   └── models/
├── scripts/
│   └── generate_report_charts.py
└── src/
    ├── q_learning.py
    ├── experiment.py
    ├── watch_best.py
    ├── train_ai_or_play.py
    └── human_play.py
```

## 环境要求

本项目使用名为 `alg` 的 conda 环境运行：

```bash
conda activate alg
```

主要依赖：

- Python 3.10
- `gymnasium`
- `flappy-bird-gymnasium`
- `pygame`
- `numpy`
- `torch`
- `pandas`
- `matplotlib`

当前机器检测到 RTX 4060 Laptop GPU，实验脚本中使用 PyTorch/CUDA 做统计排序；环境交互本身仍主要由 CPU 完成。

## 直接查看最佳模型表现

打开游戏窗口观看 `q_best.pkl` 自动游玩：

```bash
conda run -n alg python src/watch_best.py
```

运行更多局：

```bash
conda run -n alg python src/watch_best.py --episodes 10
```

无图形界面时只看分数：

```bash
conda run -n alg python src/watch_best.py --no-render --episodes 5
```

查看 bonus 阶段最佳模型：

```bash
conda run -n alg python src/watch_best.py --model q_bonus_best.pkl --state-mode enhanced --no-render --episodes 5
```

## 重新训练与复现实验

基础参数搜索：

```bash
conda run -n alg python src/experiment.py --executor process --workers 6
```

bonus 状态和奖励实验：

```bash
conda run -n alg python src/experiment.py \
  --grid bonus \
  --executor process \
  --workers 6 \
  --results-dir results_bonus \
  --best-model-path q_bonus_best.pkl
```

生成报告图表：

```bash
conda run -n alg python scripts/generate_report_charts.py
```

基础编译检查：

```bash
conda run -n alg python -m py_compile src/*.py scripts/*.py
```

## 实验结果概览

基础实验中，最终最佳模型为：

| 模型 | alpha | gamma | epsilon | obs_mul_factor | 50 局均分 | 最高分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_best.pkl` | 0.7 | 0.95 | 0.0 | 30 | 90.52 | 132 |

![基础复训结果](assets/charts/base_retrain.png)

bonus 对比中，初筛阶段状态表示和奖励塑形都有明显效果，但复训后没有超过基础最佳模型，因此最终提交仍以 `q_best.pkl` 为最佳模型，`q_bonus_best.pkl` 作为 bonus 对比产物保存。

![Bonus 初筛结果](assets/charts/bonus_screen.png)

## 状态表示

`src/q_learning.py` 中的 `process_obs` 支持三种状态：

| state_mode | 特征 | 用途 |
| --- | --- | --- |
| `example` | 水平距离、到下管道距离、垂直速度 | 作业示例状态 |
| `enhanced` | 水平距离、到缺口中心距离、到下管道距离、垂直速度 | 最终最佳模型使用 |
| `bonus` | 水平距离、到缺口中心距离、到上管道距离、到下管道距离、垂直速度、旋转角 | bonus1 扩展状态 |

## 奖励设计

基础奖励：

- 存活：保留环境奖励 `+0.1`
- 过管：保留环境奖励 `+1.0`
- 死亡：从 `-1` 放大为 `-1000`

bonus2 奖励塑形：

- 接近管道缺口中心：附加正奖励
- 垂直速度过大：附加惩罚
- flap 动作：附加很小惩罚
- 触顶：附加较大惩罚

## 报告与数据

- 完整报告：`report.md`
- 基础实验数据：`results/experiment_results.csv`
- 基础最佳摘要：`results/best_summary.json`
- bonus 实验数据：`results_bonus/experiment_results.csv`
- bonus 最佳摘要：`results_bonus/best_summary.json`

## 备注

训练脚本保留 `ThreadPoolExecutor` 线程后端，但完整实验使用 `process` 后端，因为 Gym 环境交互和 Q 表更新主要是 Python/CPU 逻辑，线程会受 GIL 影响。CUDA 主要用于批量统计和候选模型排序。

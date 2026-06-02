# AGENT.md

本文件面向后续接手本仓库的 coding agent，概括项目事实、常用命令、产物约定和容易踩坑的地方。

## 项目概览

这是一个 Flappy Bird 强化学习课程项目。核心要求是使用表格型 Q-learning 训练智能体，而不是 DQN 或其他深度强化学习方法。

当前已完成：

- `src/q_learning.py` 中的 `GameAI` Q-learning 逻辑。
- 必做参数搜索与最佳模型保存。
- bonus1：状态表示设计与对比。
- bonus2：reward shaping 与对比。
- PDF 转 Markdown 文档。
- 图文版 `README.md` 和 `report.md`。
- 可视化图表与图表生成脚本。

最终基础最佳模型是 `q_best.pkl`，bonus 对比模型是 `q_bonus_best.pkl`。

## 环境

用户指定 Python 环境为 conda 环境 `alg`。

常用运行方式：

```bash
conda run -n alg python ...
```

已知依赖包括：

- Python 3.10
- `gymnasium`
- `flappy-bird-gymnasium`
- `pygame`
- `torch`
- `pandas`
- `matplotlib`

机器有 RTX 4060 Laptop GPU。实验脚本使用 PyTorch/CUDA 做统计与排序，但 Gym 环境交互和表格 Q 表更新主要是 CPU/Python 逻辑。

## 关键文件

| 路径 | 说明 |
| --- | --- |
| `src/q_learning.py` | Q-learning 智能体、状态处理、奖励塑形、训练和播放 |
| `src/experiment.py` | 参数搜索、bonus 实验、多 worker 训练、结果保存 |
| `src/watch_best.py` | 加载模型并观看/无渲染评估 |
| `scripts/generate_report_charts.py` | 从 CSV 结果生成报告图表 |
| `README.md` | 项目说明和运行手册 |
| `report.md` | 作业报告 |
| `q_best.pkl` | 最终基础最佳模型，默认展示脚本加载它 |
| `q_bonus_best.pkl` | bonus 对比实验最佳模型 |
| `results/` | 基础参数实验结果和复训模型 |
| `results_bonus/` | bonus 状态/奖励实验结果和复训模型 |
| `assets/charts/` | 报告图表 |
| `docs/` | PDF 转换后的 Markdown 文档 |

## Q-learning 实现约定

`GameAI.q` 是字典：

```python
self.q[(tuple(state), action)] = q_value
```

动作空间固定为：

- `0`：不 flap
- `1`：flap

Q-learning 更新公式：

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

`choose_action` 训练时可用 epsilon-greedy；测试/展示时应使用 `use_epsilon=False`。

## 状态与奖励模式

`process_obs(obs, obs_mul_factor=30, state_mode="enhanced")` 支持：

| state_mode | 含义 |
| --- | --- |
| `example` | 作业示例状态：水平距离、到下管道距离、垂直速度 |
| `enhanced` | 最终最佳模型使用：增加到缺口中心的距离 |
| `bonus` | bonus1 扩展：再加入上管道距离和旋转角 |

`shape_reward(..., reward_mode=...)` 支持：

| reward_mode | 含义 |
| --- | --- |
| `death_penalty` | 基础方案：死亡奖励 `-1` 放大为 `-1000` |
| `shaped` | bonus2：在基础奖励外加入缺口中心奖励、速度惩罚、flap 惩罚、触顶惩罚 |

注意：`q_best.pkl` 对应 `state_mode="enhanced"`。如果用错误的状态模式加载模型，表现会不可靠。

## 常用命令

编译检查：

```bash
conda run -n alg python -m py_compile src/*.py scripts/*.py
```

观看基础最佳模型：

```bash
conda run -n alg python src/watch_best.py
```

无渲染验证基础最佳模型：

```bash
conda run -n alg python src/watch_best.py --no-render --episodes 5
```

观看/验证 bonus 模型：

```bash
conda run -n alg python src/watch_best.py \
  --model q_bonus_best.pkl \
  --state-mode enhanced \
  --no-render \
  --episodes 5
```

重新生成图表：

```bash
conda run -n alg python scripts/generate_report_charts.py
```

重新跑基础实验：

```bash
conda run -n alg python src/experiment.py --executor process --workers 6
```

重新跑 bonus 实验：

```bash
conda run -n alg python src/experiment.py \
  --grid bonus \
  --executor process \
  --workers 6 \
  --results-dir results_bonus \
  --best-model-path q_bonus_best.pkl
```

## 实验结果摘要

基础最佳模型：

- 模型：`q_best.pkl`
- 参数：`alpha=0.7`, `gamma=0.95`, `epsilon=0.0`, `obs_mul_factor=30`
- 状态：`enhanced`
- 奖励：`death_penalty`
- 50 局固定评估均分：`90.52`
- 最高分：`132`

bonus 实验结论：

- 初筛中 `enhanced` 和 `bonus` 状态明显优于 `example`。
- `shaped` reward 初筛表现好，但复训未超过基础最佳。
- 因此最终提交模型仍保留 `q_best.pkl`，bonus 结果作为扩展分析。

## 图表与报告

报告图表由 `scripts/generate_report_charts.py` 从以下 CSV 生成：

- `results/experiment_results.csv`
- `results_bonus/experiment_results.csv`

生成位置：

- `assets/charts/base_screen_top12.png`
- `assets/charts/base_retrain.png`
- `assets/charts/best_scores_line.png`
- `assets/charts/bonus_screen.png`
- `assets/charts/bonus_retrain.png`
- `assets/charts/state_reward_key_compare.png`

如果修改 CSV 或实验结果，应重新运行图表脚本，并同步更新 `report.md` 中的文字分析。

## 注意事项

- 不要把项目改成 DQN；作业明确要求使用 Q-learning。
- `q_best.pkl` 是当前最终最佳模型，除非新实验稳定超过它，否则不要覆盖。
- 完整实验耗时较长，高分策略会导致单局更长。`src/experiment.py` 中保留了训练和评估步数上限。
- `ThreadPoolExecutor` 线程后端保留用于满足“多线程”思路，但完整实验推荐 `--executor process`，否则 Python GIL 会限制训练吞吐。
- 图表中的中文字体依赖系统 Noto CJK 字体；脚本已自动配置常见路径。
- `assets/294score.png` 和 `assets/result.png` 是展示截图素材，报告和 README 已引用。
- 工作区可能领先远端提交；提交前先检查 `git status --short --branch`。

## 当前提交习惯

本项目提交信息使用中文，并以 `feat:`、`docs:` 等前缀开头，例如：

```text
feat: 支持状态表示和奖励塑形bonus
docs: 完善README和图文实验报告
docs: 优化图表实验标签可读性
```

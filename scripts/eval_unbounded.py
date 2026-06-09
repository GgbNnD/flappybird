"""
无步数限制评估脚本：加载已有 Q 表模型，运行指定局数的评估，
记录每局真实的 pipe 过关分数（不设步数上限），输出统计指标。
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium
import flappy_bird_gymnasium  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from q_learning import GameAI, process_obs  # noqa: E402


def evaluate_unbounded(ai, obs_mul_factor, state_mode, seeds):
    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    scores = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            while True:
                action = ai.choose_action(
                    process_obs(obs, obs_mul_factor, state_mode),
                    use_epsilon=False,
                )
                obs, _, done, _, info = env.step(action)
                if done:
                    scores.append(int(info["score"]))
                    break
    finally:
        env.close()

    import torch
    values = torch.tensor(scores, dtype=torch.float32, device="cpu")
    return {
        "mean_score": float(values.mean().item()),
        "min_score": int(values.min().item()),
        "max_score": int(values.max().item()),
        "std_score": float(values.std(unbiased=False).item()),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="无步数限制模型评估")
    parser.add_argument("model", help=".pkl 模型文件路径")
    parser.add_argument("--episodes", type=int, default=50, help="评估局数（默认 50）")
    parser.add_argument("--seed-base", type=int, default=2026, help="种子基准（默认 2026）")
    parser.add_argument("--obs-mul-factor", type=int, default=30)
    parser.add_argument("--state-mode", default="enhanced")
    parser.add_argument("--json-output", default=None, help="保存 JSON 路径")
    args = parser.parse_args()

    ai = GameAI()
    ai.load_q(args.model)

    seeds = [args.seed_base + 1000 + i for i in range(args.episodes)]
    print(f"评估 {args.model}，{args.episodes} 局，无步数限制 ...")
    summary = evaluate_unbounded(ai, args.obs_mul_factor, args.state_mode, seeds)

    print(f"  mean={summary['mean_score']:.2f}  min={summary['min_score']}  "
          f"max={summary['max_score']}  std={summary['std_score']:.2f}  "
          f"q_size={len(ai.q)}")
    print(f"  scores: {summary['scores']}")

    if args.json_output:
        output = {**summary, "model": args.model, "episodes": args.episodes}
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {args.json_output}")


if __name__ == "__main__":
    main()

"""训练 Flappy Bird Q-Learning 模型."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q_learning import train  # noqa: E402


STATE_CHOICES = ["example", "enhanced", "enhanced_top", "bonus", "lookahead"]
REWARD_CHOICES = ["death_penalty", "shaped", "shaped_v2"]


def main():
    parser = argparse.ArgumentParser(
        description="训练 Flappy Bird Q-Learning 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python train.py model.pkl --state-mode enhanced --reward-mode death_penalty\n"
               "  python train.py model.pkl --state-mode enhanced_top --reward-mode shaped_v2 --iterations 100000\n"
               "  python train.py model.pkl --resume existing.pkl --iterations 50000",
    )
    parser.add_argument("model", help="训练完成后的模型保存路径 (.pkl)")
    parser.add_argument("--resume", default=None, help="加载已有模型继续训练 (.pkl)")
    parser.add_argument("--state-mode", choices=STATE_CHOICES, default="enhanced")
    parser.add_argument("--reward-mode", choices=REWARD_CHOICES, default="death_penalty")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--obs-mul-factor", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--death-penalty", type=int, default=-1000)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="单局步数上限（默认无限制）")
    parser.add_argument("--progress-interval", type=int, default=1000,
                        help="训练进度打印间隔, 0=不打印")
    args = parser.parse_args()

    output_path = Path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training: state={args.state_mode} reward={args.reward_mode} "
          f"alpha={args.alpha} gamma={args.gamma} epsilon={args.epsilon} "
          f"iterations={args.iterations} obs_mf={args.obs_mul_factor} seed={args.seed}")
    if args.resume:
        print(f"Resuming from: {args.resume}")

    start = time.time()
    ai = train(
        iteration=args.iterations,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        obs_mul_factor=args.obs_mul_factor,
        seed=args.seed,
        death_penalty=args.death_penalty,
        progress_interval=args.progress_interval,
        max_steps_per_episode=args.max_steps,
        state_mode=args.state_mode,
        reward_mode=args.reward_mode,
        load_path=args.resume,
    )
    elapsed = time.time() - start

    ai.save_q(str(output_path))
    print(f"Saved: {output_path}  q_size={len(ai.q)}  time={elapsed:.0f}s")


if __name__ == "__main__":
    main()

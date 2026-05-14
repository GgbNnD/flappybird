import argparse
from pathlib import Path

from q_learning import GameAI, play


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Watch the trained Q-learning model play Flappy Bird.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "q_best.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--obs-mul-factor", type=int, default=30)
    parser.add_argument("--state-mode", choices=["example", "enhanced", "bonus"], default="enhanced")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--audio-on", action="store_true")
    args = parser.parse_args()

    ai = GameAI()
    ai.load_q(args.model)

    render_mode = None if args.no_render else "human"
    play(
        ai,
        audio_on=args.audio_on,
        render_mode=render_mode,
        use_lidar=False,
        episodes=args.episodes,
        obs_mul_factor=args.obs_mul_factor,
        state_mode=args.state_mode,
    )


if __name__ == "__main__":
    main()

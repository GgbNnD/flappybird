from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import time

import gymnasium
import torch
import flappy_bird_gymnasium

from q_learning import GameAI, process_obs, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    alpha: float
    gamma: float
    epsilon: float
    obs_mul_factor: int


def parameter_grid():
    return [
        TrialConfig("baseline", 0.7, 0.95, 0.0, 30),
        TrialConfig("a05_g090_e005_m30", 0.5, 0.90, 0.05, 30),
        TrialConfig("a05_g095_e005_m30", 0.5, 0.95, 0.05, 30),
        TrialConfig("a05_g099_e005_m30", 0.5, 0.99, 0.05, 30),
        TrialConfig("a06_g095_e005_m30", 0.6, 0.95, 0.05, 30),
        TrialConfig("a06_g099_e005_m30", 0.6, 0.99, 0.05, 30),
        TrialConfig("a07_g090_e005_m30", 0.7, 0.90, 0.05, 30),
        TrialConfig("a07_g095_e005_m30", 0.7, 0.95, 0.05, 30),
        TrialConfig("a07_g099_e005_m30", 0.7, 0.99, 0.05, 30),
        TrialConfig("a08_g095_e005_m30", 0.8, 0.95, 0.05, 30),
        TrialConfig("a08_g099_e005_m30", 0.8, 0.99, 0.05, 30),
        TrialConfig("a07_g095_e010_m30", 0.7, 0.95, 0.10, 30),
        TrialConfig("a07_g099_e010_m30", 0.7, 0.99, 0.10, 30),
        TrialConfig("a06_g095_e010_m24", 0.6, 0.95, 0.10, 24),
        TrialConfig("a06_g099_e010_m24", 0.6, 0.99, 0.10, 24),
        TrialConfig("a07_g095_e010_m36", 0.7, 0.95, 0.10, 36),
        TrialConfig("a07_g099_e010_m36", 0.7, 0.99, 0.10, 36),
        TrialConfig("a08_g099_e010_m36", 0.8, 0.99, 0.10, 36),
    ]


def score_summary(scores, device):
    values = torch.tensor(scores, dtype=torch.float32, device=device)
    return {
        "mean_score": float(values.mean().item()),
        "min_score": int(values.min().item()),
        "max_score": int(values.max().item()),
        "std_score": float(values.std(unbiased=False).item()),
    }


def evaluate(ai, obs_mul_factor, seeds, device, max_steps):
    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    scores = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            steps = 0
            while True:
                action = ai.choose_action(
                    process_obs(obs, obs_mul_factor),
                    use_epsilon=False,
                )
                obs, _, done, _, info = env.step(action)
                steps += 1
                if done or steps >= max_steps:
                    scores.append(int(info["score"]))
                    break
    finally:
        env.close()

    summary = score_summary(scores, device)
    summary["scores"] = scores
    return summary


def run_trial(
    config,
    phase,
    iterations,
    eval_seeds,
    model_path,
    device,
    seed_base,
    train_max_steps,
    eval_max_steps,
):
    start = time.time()
    numeric_seed = seed_base + sum(ord(ch) for ch in config.trial_id)
    ai = train(
        iterations,
        config.alpha,
        config.gamma,
        config.epsilon,
        obs_mul_factor=config.obs_mul_factor,
        seed=numeric_seed,
        progress_interval=0,
        max_steps_per_episode=train_max_steps,
    )
    summary = evaluate(ai, config.obs_mul_factor, eval_seeds, device, eval_max_steps)

    if model_path:
        ai.save_q(str(model_path))

    result = {
        "phase": phase,
        **asdict(config),
        "iterations": iterations,
        "eval_episodes": len(eval_seeds),
        "training_seconds": round(time.time() - start, 3),
        "q_size": len(ai.q),
        "model_path": str(model_path) if model_path else "",
        "scores": json.dumps(summary.pop("scores"), ensure_ascii=False),
        **summary,
    }
    return result


def write_csv(path, rows):
    fieldnames = [
        "phase",
        "trial_id",
        "alpha",
        "gamma",
        "epsilon",
        "obs_mul_factor",
        "iterations",
        "eval_episodes",
        "mean_score",
        "min_score",
        "max_score",
        "std_score",
        "q_size",
        "training_seconds",
        "model_path",
        "scores",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_top_configs(results, configs, top_k, device):
    means = torch.tensor([row["mean_score"] for row in results], device=device)
    order = torch.argsort(means, descending=True).tolist()
    config_map = {config.trial_id: config for config in configs}
    return [config_map[results[index]["trial_id"]] for index in order[:top_k]]


def run_phase(
    configs,
    phase,
    iterations,
    eval_seeds,
    models_dir,
    device,
    seed_base,
    workers,
    executor_name,
    train_max_steps,
    eval_max_steps,
):
    rows = []
    executor_cls = ThreadPoolExecutor if executor_name == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        futures = []
        for config in configs:
            model_path = None
            if phase == "retrain":
                model_path = models_dir / f"{config.trial_id}_{iterations}.pkl"
            futures.append(
                executor.submit(
                    run_trial,
                    config,
                    phase,
                    iterations,
                    eval_seeds,
                    model_path,
                    device,
                    seed_base,
                    train_max_steps,
                    eval_max_steps,
                )
            )

        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{phase}] {row['trial_id']} mean={row['mean_score']:.2f} "
                f"max={row['max_score']} time={row['training_seconds']}s",
                flush=True,
            )

    rows.sort(key=lambda row: row["mean_score"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run threaded Q-learning experiments.")
    parser.add_argument("--screen-iterations", type=int, default=30000)
    parser.add_argument("--retrain-iterations", type=int, default=100000)
    parser.add_argument("--screen-episodes", type=int, default=20)
    parser.add_argument("--retrain-episodes", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--executor", choices=["thread", "process"], default="thread")
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--best-model-path", default=str(PROJECT_ROOT / "q_best.pkl"))
    parser.add_argument("--train-max-steps", type=int, default=1000)
    parser.add_argument("--eval-max-steps", type=int, default=5000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = parameter_grid()
    screen_seeds = [args.seed_base + i for i in range(args.screen_episodes)]
    retrain_seeds = [args.seed_base + 1000 + i for i in range(args.retrain_episodes)]

    print(
        f"Using {args.workers} {args.executor} worker(s), stats device={device}",
        flush=True,
    )
    worker_device = torch.device("cpu") if args.executor == "process" else device
    screen_rows = run_phase(
        configs,
        "screen",
        args.screen_iterations,
        screen_seeds,
        models_dir,
        worker_device,
        args.seed_base,
        args.workers,
        args.executor,
        args.train_max_steps,
        args.eval_max_steps,
    )
    top_configs = select_top_configs(screen_rows, configs, args.top_k, device)
    retrain_rows = run_phase(
        top_configs,
        "retrain",
        args.retrain_iterations,
        retrain_seeds,
        models_dir,
        worker_device,
        args.seed_base + 10000,
        args.workers,
        args.executor,
        args.train_max_steps,
        args.eval_max_steps,
    )

    all_rows = screen_rows + retrain_rows
    write_csv(results_dir / "experiment_results.csv", all_rows)

    best = retrain_rows[0]
    best_model_path = Path(args.best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best["model_path"], best_model_path)
    best_summary = {
        "best_trial": best,
        "screen_iterations": args.screen_iterations,
        "retrain_iterations": args.retrain_iterations,
        "screen_episodes": args.screen_episodes,
        "retrain_episodes": args.retrain_episodes,
        "workers": args.workers,
        "executor": args.executor,
        "train_max_steps": args.train_max_steps,
        "eval_max_steps": args.eval_max_steps,
        "worker_stats_device": str(worker_device),
        "selection_device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    with (results_dir / "best_summary.json").open("w", encoding="utf-8") as file:
        json.dump(best_summary, file, ensure_ascii=False, indent=2)

    print(f"Best model copied to {best_model_path}: {best['trial_id']}", flush=True)


if __name__ == "__main__":
    main()

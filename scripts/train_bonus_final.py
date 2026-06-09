"""训练 bonus1 (enhanced_top) + bonus2 复训 + 组合模型。"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q_learning import train

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # (name, state_mode, reward_mode, obs_mf, iterations)
    ("enhanced_top_shaped_v2", "enhanced_top", "shaped_v2", 30, 50000),
    ("enhanced_top_death",      "enhanced_top", "death_penalty", 30, 50000),
    ("enhanced_shaped_v2",      "enhanced",     "shaped_v2", 30, 100000),
]

ALPHA, GAMMA, EPSILON = 0.7, 0.95, 0.0

for name, state_mode, reward_mode, obs_mf, iters in CONFIGS:
    suffix = f"{iters//1000}k"
    path = MODELS_DIR / f"{name}_{iters}.pkl"
    if path.exists():
        print(f"SKIP {name}: already exists")
        continue
    print(f"\n{'='*60}")
    print(f"Training {name} ({suffix}): state={state_mode} reward={reward_mode}")
    print(f"{'='*60}")
    start = time.time()
    ai = train(
        iteration=iters,
        alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
        obs_mul_factor=obs_mf, seed=42, max_steps_per_episode=1000,
        state_mode=state_mode, reward_mode=reward_mode,
    )
    elapsed = time.time() - start
    ai.save_q(str(path))
    print(f"Saved {path.name} | q_size={len(ai.q)} | time={elapsed:.0f}s")

print("\nAll done.")

"""训练 bonus 模型（各 50k 局，含步数上限）。"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q_learning import train

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # (name, state_mode, reward_mode, obs_mul_factor)
    ("lookahead_death_20",    "lookahead", "death_penalty", 20),
    ("enhanced_shaped_v2",    "enhanced",  "shaped_v2",      30),
    ("lookahead_shaped_v2_20","lookahead", "shaped_v2",      20),
]

ITERATIONS = 50000
ALPHA, GAMMA, EPSILON = 0.7, 0.95, 0.0

for name, state_mode, reward_mode, obs_mf in CONFIGS:
    path = MODELS_DIR / f"{name}_50000.pkl"
    if path.exists():
        print(f"SKIP {name}: already exists")
        continue
    print(f"\n{'='*60}")
    print(f"Training {name}: state={state_mode} reward={reward_mode} obs_mf={obs_mf}")
    print(f"{'='*60}")
    start = time.time()
    ai = train(
        iteration=ITERATIONS,
        alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
        obs_mul_factor=obs_mf, seed=42, max_steps_per_episode=1000,
        state_mode=state_mode, reward_mode=reward_mode,
    )
    elapsed = time.time() - start
    ai.save_q(str(path))
    print(f"Saved {path.name} | q_size={len(ai.q)} | time={elapsed:.0f}s")

print("\nAll done.")

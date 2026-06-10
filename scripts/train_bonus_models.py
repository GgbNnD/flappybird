"""快速训练 bonus1/bonus2/组合模型（各 50k 局）。"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q_learning import train

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("lookahead_death",     "lookahead", "death_penalty"),
    ("enhanced_shaped_v2",  "enhanced",  "shaped_v2"),
    ("lookahead_shaped_v2", "lookahead", "shaped_v2"),
]

ITERATIONS = 50000
ALPHA, GAMMA, EPSILON = 0.7, 0.95, 0.0

for name, state_mode, reward_mode in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Training {name}: state={state_mode} reward={reward_mode}")
    print(f"{'='*60}")
    start = time.time()
    ai = train(
        iteration=ITERATIONS,
        alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
        obs_mul_factor=30, seed=42,
        state_mode=state_mode, reward_mode=reward_mode,
    )
    elapsed = time.time() - start
    path = MODELS_DIR / f"{name}_50000.pkl"
    ai.save_q(str(path))
    print(f"Saved {path} | q_size={len(ai.q)} | time={elapsed:.0f}s")

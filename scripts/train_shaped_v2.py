"""训练剩余的 bonus2 模型（enhanced+shaped_v2）—— 单独的脚本."""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q_learning import train

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ITERATIONS = 50000
ALPHA, GAMMA, EPSILON = 0.7, 0.95, 0.0

name = "enhanced_shaped_v2"
print(f"\nTraining {name}: state=enhanced reward=shaped_v2")
start = time.time()
ai = train(
    iteration=ITERATIONS,
    alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
    obs_mul_factor=30, seed=42,
    state_mode="enhanced", reward_mode="shaped_v2",
)
elapsed = time.time() - start
path = MODELS_DIR / f"{name}_50000.pkl"
ai.save_q(str(path))
print(f"Saved {path} | q_size={len(ai.q)} | time={elapsed:.0f}s")

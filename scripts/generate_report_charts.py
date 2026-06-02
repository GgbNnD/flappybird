from pathlib import Path
import ast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets" / "charts"


def configure_fonts():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_bar(df, title, xlabel, ylabel, output, color="#3f7cac", limit=None):
    if limit:
        df = df.head(limit)
    fig_height = max(4.5, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(df[xlabel], df[ylabel], color=color)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("平均分")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for index, value in enumerate(df[ylabel]):
        ax.text(value + 1, index, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    configure_fonts()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(PROJECT_ROOT / "results" / "experiment_results.csv")
    bonus = pd.read_csv(PROJECT_ROOT / "results_bonus" / "experiment_results.csv")

    screen = base[base["phase"] == "screen"].sort_values("mean_score", ascending=False)
    retrain = base[base["phase"] == "retrain"].sort_values("mean_score", ascending=False)
    bonus_screen = bonus[bonus["phase"] == "screen"].sort_values("mean_score", ascending=False)
    bonus_retrain = bonus[bonus["phase"] == "retrain"].sort_values("mean_score", ascending=False)

    save_bar(
        screen,
        "基础参数初筛平均分 Top 12",
        "trial_id",
        "mean_score",
        ASSETS_DIR / "base_screen_top12.png",
        color="#31708e",
        limit=12,
    )
    save_bar(
        retrain,
        "基础参数复训平均分",
        "trial_id",
        "mean_score",
        ASSETS_DIR / "base_retrain.png",
        color="#4f8a5b",
    )
    save_bar(
        bonus_screen,
        "Bonus 状态与奖励初筛平均分",
        "trial_id",
        "mean_score",
        ASSETS_DIR / "bonus_screen.png",
        color="#b65f37",
    )
    save_bar(
        bonus_retrain,
        "Bonus 复训平均分",
        "trial_id",
        "mean_score",
        ASSETS_DIR / "bonus_retrain.png",
        color="#7c5aa6",
    )

    best = retrain.iloc[0]
    scores = ast.literal_eval(best["scores"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", linewidth=1.8, color="#2b6cb0")
    ax.axhline(best["mean_score"], linestyle="--", color="#c94c4c", label=f"平均分 {best['mean_score']:.2f}")
    ax.set_title("最佳模型 50 局评估分数")
    ax.set_xlabel("评估局数")
    ax.set_ylabel("分数")
    ax.set_ylim(0, max(scores) + 15)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "best_scores_line.png", dpi=160)
    plt.close(fig)

    phase_compare = pd.DataFrame(
        [
            {"name": "示例状态 example", "score": 63.70},
            {"name": "增强状态 enhanced", "score": 120.20},
            {"name": "扩展状态 bonus", "score": 115.90},
            {"name": "增强状态 + shaped reward", "score": 119.95},
        ]
    )
    save_bar(
        phase_compare,
        "状态表示与奖励塑形关键对比",
        "name",
        "score",
        ASSETS_DIR / "state_reward_key_compare.png",
        color="#4b7f52",
    )


if __name__ == "__main__":
    main()

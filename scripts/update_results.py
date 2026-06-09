"""
更新 CSV 和 JSON 中的评估分数数据为无步数上限版本。
"""
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path("/home/cells/flappybird")

NEW_RESULTS = {
    "base": {
        "baseline": {"mean": 165.18, "min": 3, "max": 984, "std": 175.66,
            "scores": [19,54,3,126,147,269,278,420,39,254,11,21,76,29,194,234,317,72,320,158,124,96,170,89,98,207,119,125,121,236,26,7,246,32,33,33,984,83,33,680,26,230,53,107,53,330,368,221,242,46]},
        "a05_g090_e005_m30": {"mean": 136.44, "min": 2, "max": 669, "std": 141.80,
            "scores": [102,4,106,95,276,78,4,71,88,137,105,189,69,70,310,9,2,669,564,31,102,3,68,372,103,249,63,120,25,45,36,17,186,31,20,66,293,64,237,278,330,20,190,356,53,125,35,242,17,97]},
        "a06_g099_e005_m30": {"mean": 53.38, "min": 4, "max": 250, "std": 43.82,
            "scores": [11,106,34,20,26,26,10,33,43,43,124,32,250,74,14,41,124,62,109,9,26,22,19,8,14,46,64,59,35,92,86,58,38,88,64,8,88,55,10,45,128,71,105,26,58,4,20,58,40,43]},
    },
    "bonus": {
        "enhanced_reward_base": {"mean": 122.30, "min": 2, "max": 634, "std": 126.11,
            "scores": [163,29,34,491,137,44,191,10,134,67,142,74,44,200,83,10,347,62,10,31,191,167,19,25,206,124,200,191,113,187,257,29,242,283,29,634,28,137,10,40,5,19,2,20,194,28,113,32,250,37]},
        "enhanced_shaped_e000": {"mean": 109.04, "min": 1, "max": 482, "std": 108.71,
            "scores": [17,226,86,398,137,44,86,181,23,125,272,20,23,71,73,29,200,215,61,16,1,59,29,25,124,13,59,173,79,200,98,13,163,43,32,49,10,37,22,166,74,16,190,482,28,373,16,242,79,254]},
        "bonus_reward_base": {"mean": 67.74, "min": 1, "max": 331, "std": 77.52,
            "scores": [19,166,35,79,118,30,10,41,14,67,239,28,1,101,131,10,49,40,11,35,95,10,32,34,44,66,49,107,324,71,4,1,4,183,1,2,29,27,196,180,331,53,52,127,35,26,35,8,14,23]},
    },
}


def update_csv(path, category):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            tid = row["trial_id"]
            if row["phase"] == "retrain" and tid in NEW_RESULTS[category]:
                nr = NEW_RESULTS[category][tid]
                row["mean_score"] = str(nr["mean"])
                row["min_score"] = str(nr["min"])
                row["max_score"] = str(nr["max"])
                row["std_score"] = str(nr["std"])
                row["scores"] = json.dumps(nr["scores"], ensure_ascii=False)
            rows.append(row)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {path}")


def update_json_summary(path, category):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data["best_trial"]
    tid = best["trial_id"]
    if tid in NEW_RESULTS[category]:
        nr = NEW_RESULTS[category][tid]
        best["mean_score"] = nr["mean"]
        best["min_score"] = nr["min"]
        best["max_score"] = nr["max"]
        best["std_score"] = nr["std"]
        best["scores"] = json.dumps(nr["scores"], ensure_ascii=False)
        data["eval_max_steps"] = "unbounded"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {path}")


if __name__ == "__main__":
    update_csv(PROJECT_ROOT / "results" / "experiment_results.csv", "base")
    update_json_summary(PROJECT_ROOT / "results" / "best_summary.json", "base")
    update_csv(PROJECT_ROOT / "results_bonus" / "experiment_results.csv", "bonus")
    update_json_summary(PROJECT_ROOT / "results_bonus" / "best_summary.json", "bonus")
    print("Done.")

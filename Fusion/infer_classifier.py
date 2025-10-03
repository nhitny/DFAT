import json
import numpy as np
import joblib
import csv

# ========= PATHS =========
model_out_path = "/workspace/sanglq/vlsp_2025/huytq/results/joblit/ensemble_optuna_frame_public_bilstm.joblib"
sentiment_json_path = "/workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/after_whisper/features_bilstm_cnn_pb_visec.json"
ser_binary_path = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/ser_public_probabilities.json"
ground_truth_path = "/workspace/sanglq/vlsp_2025/vlsp2025_asrser/datasets/metadata/VLSP2025-ASR-SER_public_test_label.tsv"

wrong_xgb_out = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/wrong_xgb_fixed.json"
ensemble_magic_out = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/ensemble_only_correct.json"

# ========= LOAD PIPELINE =========
pipeline = joblib.load(model_out_path)
scaler = pipeline["scaler"]
xgb = pipeline["xgb"]
rf = pipeline["rf"]
lr = pipeline["lr"]
w_xgb, w_rf, w_lr = pipeline["weights"]
print(f"✅ Loaded ensemble | Weights: XGB={w_xgb}, RF={w_rf}, LR={w_lr}")

# ========= LOAD INPUT DATA =========
with open(sentiment_json_path) as f:
    sentiment = json.load(f)
with open(ser_binary_path) as f:
    ser_binary = json.load(f)

# ========= LOAD GROUND TRUTH =========
ground_truth = {}
with open(ground_truth_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        audio_id = row["path"]
        label = row["emotion"].strip().lower()
        ground_truth[audio_id] = 1 if label == "negative" else 0
print(f"✅ Loaded {len(ground_truth)} ground-truth labels")

# ========= HELPER =========
def extract_feats(entry):
    feats = []
    if isinstance(entry, list):
        for e in entry:
            if isinstance(e, dict) and "feats" in e:
                feats.append(e["feats"])
            elif isinstance(e, list):
                feats.append(e)
    elif isinstance(entry, dict) and "feats" in entry:
        feats.append(entry["feats"])
    feats = np.array(feats, dtype=float)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    if feats.shape[0] == 0 or feats.shape[1] == 0:
        return None
    return np.mean(feats, axis=0)

# ========= MAIN ANALYSIS =========
correct_counts = {"xgb": 0, "rf": 0, "lr": 0, "ensemble": 0}
total = 0
wrong_xgb_fixed_by_others = []
ensemble_only_correct = []

# class-wise tracking
class_wise = {
    "neutral": {"xgb": {"correct": 0, "total": 0},
                "rf": {"correct": 0, "total": 0},
                "lr": {"correct": 0, "total": 0},
                "ensemble": {"correct": 0, "total": 0}},
    "negative": {"xgb": {"correct": 0, "total": 0},
                 "rf": {"correct": 0, "total": 0},
                 "lr": {"correct": 0, "total": 0},
                 "ensemble": {"correct": 0, "total": 0}}
}

for audio_id in sentiment.keys():
    if audio_id not in ground_truth:
        continue
    true_label = ground_truth[audio_id]
    true_class = "negative" if true_label == 1 else "neutral"

    stats_s = extract_feats(sentiment[audio_id])
    stats_r = extract_feats(ser_binary[audio_id])
    if stats_s is None or stats_r is None:
        continue

    feat = [*stats_s, *stats_r]
    X_input = scaler.transform([feat])

    # probabilities
    p_xgb = xgb.predict_proba(X_input)[0, 1]
    p_rf  = rf.predict_proba(X_input)[0, 1]
    p_lr  = lr.predict_proba(X_input)[0, 1]
    p_final = w_xgb*p_xgb + w_rf*p_rf + w_lr*p_lr

    # binary predictions
    pred_xgb = 1 if p_xgb >= 0.5 else 0
    pred_rf  = 1 if p_rf >= 0.5 else 0
    pred_lr  = 1 if p_lr >= 0.5 else 0
    pred_final = 1 if p_final >= 0.5 else 0

    # overall accuracy counts
    total += 1
    if pred_xgb == true_label:
        correct_counts["xgb"] += 1
    if pred_rf == true_label:
        correct_counts["rf"] += 1
    if pred_lr == true_label:
        correct_counts["lr"] += 1
    if pred_final == true_label:
        correct_counts["ensemble"] += 1

    # class-wise stats
    for model_name, pred in zip(
        ["xgb", "rf", "lr", "ensemble"],
        [pred_xgb, pred_rf, pred_lr, pred_final]
    ):
        class_wise[true_class][model_name]["total"] += 1
        if pred == true_label:
            class_wise[true_class][model_name]["correct"] += 1

    # case 1: XGB wrong, but others/ensemble correct
    if pred_xgb != true_label and (
        pred_rf == true_label or pred_lr == true_label or pred_final == true_label
    ):
        wrong_xgb_fixed_by_others.append({
            "audio_id": audio_id,
            "true_label": true_label,
            "p_xgb": round(float(p_xgb), 3),
            "p_rf": round(float(p_rf), 3),
            "p_lr": round(float(p_lr), 3),
            "p_final": round(float(p_final), 3),
            "pred_xgb": pred_xgb,
            "pred_rf": pred_rf,
            "pred_lr": pred_lr,
            "pred_final": pred_final,
        })

    # case 2: Ensemble correct, all three base models wrong
    if (
        pred_final == true_label
        and pred_xgb != true_label
        and pred_rf != true_label
        and pred_lr != true_label
    ):
        ensemble_only_correct.append({
            "audio_id": audio_id,
            "true_label": true_label,
            "p_xgb": round(float(p_xgb), 3),
            "p_rf": round(float(p_rf), 3),
            "p_lr": round(float(p_lr), 3),
            "p_final": round(float(p_final), 3),
            "pred_xgb": pred_xgb,
            "pred_rf": pred_rf,
            "pred_lr": pred_lr,
            "pred_final": pred_final,
        })

# ========= PRINT OVERALL ACCURACY =========
print("\n====== Overall Accuracy ======")
for model, count in correct_counts.items():
    acc = count / total * 100 if total > 0 else 0
    print(f"{model.upper()}: {count}/{total} correct ({acc:.2f}%)")

print(f"\n🔎 Found {len(wrong_xgb_fixed_by_others)} cases where XGB was wrong but others/ensemble corrected it.")
print(f"✨ Found {len(ensemble_only_correct)} cases where only the Ensemble was correct (all base models wrong).")

# ========= PRINT CLASS-WISE RESULTS =========
print("\n====== Class-wise Accuracy ======")
for cls in class_wise:
    print(f"\nClass: {cls.upper()}")
    for model_name, stats in class_wise[cls].items():
        total_cls = stats["total"]
        correct_cls = stats["correct"]
        acc_cls = correct_cls / total_cls * 100 if total_cls > 0 else 0
        print(f"  {model_name.upper()}: {correct_cls}/{total_cls} correct ({acc_cls:.2f}%)")

# ========= SAVE ERROR ANALYSIS =========
with open(wrong_xgb_out, "w", encoding="utf-8") as f:
    json.dump(wrong_xgb_fixed_by_others, f, indent=2, ensure_ascii=False)

with open(ensemble_magic_out, "w", encoding="utf-8") as f:
    json.dump(ensemble_only_correct, f, indent=2, ensure_ascii=False)

print(f"✅ Saved XGB error analysis to {wrong_xgb_out}")
print(f"✅ Saved Ensemble-only cases to {ensemble_magic_out}")

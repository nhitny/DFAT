import json, os, math, joblib
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ========= CONFIG =========
RANDOM_SEED = 42
N_TRIALS_XGB = 10
N_TRIALS_WEIGHTS = 5

# ========= PATHS =========
sentiment_json_path = "/workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/features_lstm_pb.json"
ser_binary_path     = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/ser_public_probabilities.json"
val_labels_path     = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/val_public_labels.json"
model_out_path      = "/workspace/sanglq/vlsp_2025/huytq/results/joblit/ensemble_optuna_frame_public_lstm.joblib"

# ========= UTILITIES =========
def safe_log(x, eps=1e-12):
    return math.log(x if x > eps else eps)

def entropy(probs):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def extract_frame_stats(entry):
    feats = []
    if isinstance(entry, list):
        for e in entry:
            if isinstance(e, dict):
                vec = e.get("feats")
                if vec: feats.append(vec)
            elif isinstance(e, list):  # fallback
                feats.append(e)
    elif isinstance(entry, dict):
        vec = entry.get("feats")
        if vec: feats.append(vec)

    feats = np.array(feats, dtype=float)
    if len(feats.shape) == 1:
        feats = feats.reshape(1, -1)
    if feats.shape[1] == 0:
        print("⚠️ Empty vector encountered in extract_frame_stats")
    return np.mean(feats, axis=0), np.var(feats, axis=0), np.max(feats, axis=0), np.min(feats, axis=0)



# ========= LOAD DATA =========
with open(sentiment_json_path) as f:
    sentiment = json.load(f)
with open(ser_binary_path) as f:
    ser_binary = json.load(f)
with open(val_labels_path) as f:
    val_labels = json.load(f)

# ========= FEATURE EXTRACTION =========
feats = []
X, y, meta_keys = [], [], []
for audio_id, gold in list(val_labels.items()):
    print(f"Processing {audio_id}...")
    # if audio_id not in sentiment or audio_id not in ser_binary:
    #     continue
    mean_s, var_s, max_s, min_s = extract_frame_stats(sentiment[audio_id])
    mean_r, var_r, max_r, min_r = extract_frame_stats(ser_binary[audio_id])
    feat = [
        *mean_s,
        *mean_r    
    ]
    # feats.append(ser_binary[audio_id]["feats"])
    # feats.append(*sentiment[audio_id]["feats"])
    X.append(feat)
    y.append(1 if gold == "negative" else 0)
    meta_keys.append(audio_id)

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)
print("Feature shape:", X.shape)

# ========= SPLIT & SCALE =========
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========= OPTUNA TUNING FOR XGBOOST =========
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
        "scale_pos_weight": float(np.sum(y==0)/np.sum(y==1)),
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        model.fit(scaler.transform(X[train_idx]), y[train_idx])
        preds = model.predict(scaler.transform(X[valid_idx]))
        scores.append(accuracy_score(y[valid_idx], preds))
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB)
best_params = study_xgb.best_params
print("Best XGBoost params:", best_params)

# ========= TRAIN BASE MODELS =========
xgb = XGBClassifier(**best_params)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_SEED)
lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_SEED)

xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# ========= AUTO-TUNE ENSEMBLE WEIGHTS =========
def objective_weights(trial):
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2

    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    proba_rf  = rf.predict_proba(X_test)[:, 1]
    proba_lr  = lr.predict_proba(X_test)[:, 1]
    final_proba = w1*proba_xgb + w2*proba_rf + w3*proba_lr
    preds = (final_proba >= 0.5).astype(int)

    return accuracy_score(y_test, preds)

study_weights = optuna.create_study(direction="maximize")
study_weights.optimize(objective_weights, n_trials=N_TRIALS_WEIGHTS)
best_w = study_weights.best_params
w1, w2 = best_w["w1"], best_w["w2"]
w3 = 1 - w1 - w2
print("Best ensemble weights:", (w1, w2, w3))

# ========= FINAL EVALUATION =========
proba_xgb = xgb.predict_proba(X_test)[:, 1]
proba_rf  = rf.predict_proba(X_test)[:, 1]
proba_lr  = lr.predict_proba(X_test)[:, 1]
final_proba = w1*proba_xgb + w2*proba_rf + w3*proba_lr
final_preds = (final_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, final_preds)
print("Final Accuracy:", acc)
print(classification_report(y_test, final_preds, target_names=["neutral", "negative"]))

# ========= SAVE PIPELINE =========
joblib.dump({
    "scaler": scaler,
    "xgb": xgb,
    "rf": rf,
    "lr": lr,
    "weights": (w1, w2, w3),
}, model_out_path)
print("Saved ensemble pipeline to:", model_out_path)

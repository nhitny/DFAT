#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, pickle, argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def main(args):
    # ========== Load data ==========
    df = pd.read_csv(args.tsv_path, sep="\t", header=None)
    # format: file_name \t label \t transcription
    fnames = df[0].astype(str).tolist()
    labels = df[1].astype(int).tolist()
    texts  = df[2].astype(str).tolist()   # ⚠️ chỗ này mình fix: transcription là cột 2, trước bạn đang lấy nhầm cột 0

    # ========== Load tokenizer ==========
    with open(args.tokenizer_path, "rb") as f:
        tok_info = pickle.load(f)
    tokenizer = tok_info["tokenizer"]
    maxlen = tok_info["maxlen"]

    # ========== Encode text ==========
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen, padding="post")

    # ========== Load model ==========
    base_model = load_model(args.model_path)

    # Lấy feature trước sigmoid
    feat_layer = base_model.layers[-2].output
    feat_model = Model(inputs=base_model.input, outputs=feat_layer)

    # ========== Predict ==========
    print(">> Extracting features (before sigmoid) ...")
    features = feat_model.predict(X, batch_size=256, verbose=1)
    probs = base_model.predict(X, batch_size=256, verbose=1).reshape(-1)

    print(f">> Feature shape: {features.shape}")

    # ========== Save predictions to JSON ==========
    results = {}
    for fname, prob, feat in zip(fnames, probs, features):
        pred_label = int(prob >= 0.5)
        results[os.path.basename(fname)] = {
            "labels": [pred_label],
            "feats": feat.tolist()
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f">> Done. Saved {len(results)} samples to {args.out_json}")

    # ========== Evaluate (WA, UA, WF1) ==========
    print("\n>> Evaluating on ground-truth ...")

    preds_norm = {os.path.basename(k): v for k, v in results.items()}
    y_true, y_pred = [], []

    for fname, true_lbl in zip(fnames, labels):
        fname_base = os.path.basename(fname)
        if fname_base in preds_norm:
            y_true.append(int(true_lbl))
            y_pred.append(int(preds_norm[fname_base]["labels"][0]))

    if len(y_true) == 0:
        raise ValueError("❌ Không tìm thấy file nào trùng giữa ground-truth và predictions!")

    wa  = accuracy_score(y_true, y_pred)                 # Weighted Accuracy
    ua  = balanced_accuracy_score(y_true, y_pred)        # Unweighted Accuracy
    wf1 = f1_score(y_true, y_pred, average="weighted")   # Weighted F1

    print("===== Evaluation =====")
    print(f"Samples: {len(y_true)}")
    print(f"WA (Accuracy): {wa:.6f}")
    print(f"UA (Balanced Acc): {ua:.6f}")
    print(f"WF1 (Weighted F1): {wf1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Emotion Inference & Evaluation")
    parser.add_argument("--tsv_path", type=str, required=True,
                        help="Path to TSV ground-truth/test file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.h5)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer.pkl")
    parser.add_argument("--out_json", type=str, required=True,
                        help="Output JSON file (features + predictions)")

    args = parser.parse_args()
    main(args)

'''
python infer_text2emotion.py \
  --tsv_path datapath \
  --model_path model_path \
  --tokenizer_path tokenizer_patt \
  --out_json features.json
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report


def evaluate(gt_path, pred_path, key="transcription"):
    # --- Load data ---
    gt_df = pd.read_csv(gt_path, sep="\t")
    pred_df = pd.read_csv(pred_path, sep="\t")

    print(">> Ground-truth sample:\n", gt_df.head(), "\n")
    print(">> Prediction sample:\n", pred_df.head(), "\n")

    # --- Merge theo transcription ---
    df = pd.merge(gt_df, pred_df, on=key, suffixes=("_gt", "_pred"))
    print(">> Merged sample:\n", df.head(), "\n")

    y_true = df["emotion_gt"].astype(str).tolist()
    y_pred = df["emotion_pred"].astype(str).tolist()

    # --- Metrics ---
    wa = accuracy_score(y_true, y_pred)               # Weighted Accuracy
    ua = balanced_accuracy_score(y_true, y_pred)      # Unweighted Accuracy
    wf1 = f1_score(y_true, y_pred, average="weighted")

    print("=== Evaluation Metrics ===")
    print(f"WA (Accuracy):     {wa:.4f}")
    print(f"UA (Balanced Acc): {ua:.4f}")
    print(f"WF1 (Weighted F1): {wf1:.4f}\n")

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions vs ground-truth")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to ground-truth TSV (with transcription, emotion_gt)")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to prediction TSV (with transcription, emotion_pred)")
    parser.add_argument("--key", type=str, default="transcription",
                        help="Column name to merge on (default: transcription)")

    args = parser.parse_args()
    evaluate(args.gt_path, args.pred_path, args.key)

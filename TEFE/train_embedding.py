#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, unicodedata
from typing import List, Iterator
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def simple_clean(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)  # chuẩn hoá unicode
    s = " ".join(s.strip().split())
    return s

class SentenceStream:
    def __init__(self, tsv_paths: List[str], text_col: str):
        self.tsv_paths = tsv_paths
        self.text_col = text_col
    def __iter__(self) -> Iterator[list]:
        for p in self.tsv_paths:
            df = pd.read_csv(p, sep="\t")
            if self.text_col not in df.columns:
                raise ValueError(f"Column '{self.text_col}' not found in {p}. "
                                 f"Available: {df.columns.tolist()}")
            for t in df[self.text_col].astype(str).fillna(""):
                yield simple_clean(t).split()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv_paths", type=str, nargs="+", required=True, help="List of TSV files")
    ap.add_argument("--text_col", type=str, default="transcription", help="Text column name")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for embeddings")
    ap.add_argument("--model", type=str, default="word2vec", choices=["word2vec","fasttext"])
    ap.add_argument("--dim", type=int, default=1024, help="Embedding dimension")
    ap.add_argument("--window", type=int, default=5, help="Context window size")
    ap.add_argument("--min_count", type=int, default=2, help="Min word frequency")
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    sentences = SentenceStream(args.tsv_paths, args.text_col)

    if args.model == "word2vec":
        from gensim.models import Word2Vec
        print(">> Training Word2Vec...")
        m = Word2Vec(
            sentences=sentences,
            vector_size=args.dim,
            window=args.window,
            min_count=args.min_count,
            sg=args.sg,
            workers=os.cpu_count() or 4,
            epochs=args.epochs,
            seed=42,
        )
        kv = m.wv
    else:
        from gensim.models import FastText
        print(">> Training FastText...")
        m = FastText(
            sentences=sentences,
            vector_size=args.dim,
            window=args.window,
            min_count=args.min_count,
            sg=args.sg,
            workers=os.cpu_count() or 4,
            epochs=args.epochs,
            seed=42,
        )
        kv = m.wv

    kv_path = os.path.join(args.out_dir, "embeddings_visec.kv")
    print(f">> Saving KeyedVectors to: {kv_path}")
    kv.save(kv_path)

    vec_path = os.path.join(args.out_dir, "embeddings_visec.vec")
    print(f">> Exporting word2vec text format to: {vec_path}")
    kv.save_word2vec_format(vec_path, binary=False)  # có header "vocab dim"

    with open(os.path.join(args.out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(">> Done.")

if __name__ == "__main__":
    main()

'''  
CUDA_VISIBLE_DEVICES=4 taskset -c 24-49 \
python /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/base/train_embedding.py \
  --tsv_paths /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/data_after_infer/data_visec_whisper-ft.tsv \
  --text_col transcription \
  --out_dir /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/saved \
  --model word2vec --dim 1024 --sg 1 --window 3 --min_count 1 --epochs 10
  
CUDA_VISIBLE_DEVICES=4 taskset -c 24-49 \
python /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/base/train_embedding.py \
  --tsv_paths /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/data_after_infer/data_two_cols.tsv \
  --text_col transcription \
  --out_dir /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/saved \
  --model word2vec --dim 1024 --sg 1 --window 3 --min_count 1 --epochs 10

'''  

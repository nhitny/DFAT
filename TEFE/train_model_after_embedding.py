#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_with_emb.py
-----------------------
This is your original train_model.py extended to load a pretrained word embedding
(trained by train_word_embedding.py) and inject it into the Embedding layer.

It supports:
- --emb_init random|gensim_kv|txt_vec
- --emb_path path to embeddings.kv (for gensim_kv) or embeddings.vec (for txt_vec)
- --freeze_embed flag to freeze the embedding layer

Everything else remains compatible with your previous CLI.
"""
import os, sys, argparse, random, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dropout, Dense, SpatialDropout1D,
    Conv1D, GlobalMaxPooling1D, Concatenate, BatchNormalization, ReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed); os.environ["PYTHONHASHSEED"]="0"

def ensure_dir(p):
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def simple_clean(s): return str(s).strip()

def log_visible_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs visible to TF:", gpus)
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

# --------- Embedding helpers ----------
def load_gensim_kv(path):
    try:
        from gensim.models import KeyedVectors
    except Exception as e:
        raise RuntimeError("Please install gensim: pip install gensim") from e
    print(f">> Loading KeyedVectors: {path}")
    return KeyedVectors.load(path, mmap='r')

def load_txt_vec(path):
    print(f">> Loading text vectors: {path}")
    vecs = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3: 
                continue
            w = parts[0]
            try:
                v = np.asarray(parts[1:], dtype="float32")
            except ValueError:
                continue
            vecs[w] = v
    return vecs

def build_embedding_matrix(tokenizer, vocab_size, embed_dim, emb_init, emb_path, freeze_embed):
    """
    Returns (embedding_matrix, trainable_embed, mask_zero)
    mask_zero=True for (Bi)LSTM paths
    """
    word_index = tokenizer.word_index
    mat = np.random.uniform(-0.05, 0.05, size=(vocab_size, embed_dim)).astype("float32")
    mat[0] = 0.0
    if emb_init == "random" or not emb_path:
        print(">> Embedding init: RANDOM")
        return mat, (not freeze_embed), True

    if emb_init == "gensim_kv":
        kv = load_gensim_kv(emb_path)
        hit = 0
        for w, idx in word_index.items():
            if idx >= vocab_size: continue
            if w in kv:
                vec = kv[w]
            elif w.lower() in kv:
                vec = kv[w.lower()]
            else:
                vec = None
            if vec is not None and len(vec) == embed_dim:
                mat[idx] = vec
                hit += 1
        print(f">> Coverage from KV: {hit}/{vocab_size-1} = {100.0*hit/max(1,(vocab_size-1)):.2f}%")
        return mat, (not freeze_embed), True

    if emb_init == "txt_vec":
        tv = load_txt_vec(emb_path)
        hit = 0
        for w, idx in word_index.items():
            if idx >= vocab_size: continue
            vec = tv.get(w) or tv.get(w.lower())
            if vec is not None and len(vec) == embed_dim:
                mat[idx] = vec
                hit += 1
        print(f">> Coverage from TXT: {hit}/{vocab_size-1} = {100.0*hit/max(1,(vocab_size-1)):.2f}%")
        return mat, (not freeze_embed), True

    print(">> Unknown emb_init; using RANDOM.")
    return mat, (not freeze_embed), True

# -------------- Models -----------------
# -------------- Models -----------------
def build_lstm(vocab_size, maxlen=200, embed_dim=1024, rnn_units=1024, lr=3e-4,
               embedding_matrix=None, trainable_embed=True, mask_zero=True):
    inp = Input(shape=(maxlen,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen,
                      mask_zero=mask_zero, weights=[embedding_matrix],
                      trainable=trainable_embed)(inp)
    else:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen, mask_zero=mask_zero)(inp)
    # LSTM trực tiếp output 1024 chiều
    x = LSTM(rnn_units, dropout=0.2, recurrent_dropout=0.0, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid", dtype="float32")(x)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m


def build_bilstm(vocab_size, maxlen=200, embed_dim=1024, rnn_units=512, lr=3e-4,
                 embedding_matrix=None, trainable_embed=True, mask_zero=True):
    inp = Input(shape=(maxlen,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen,
                      mask_zero=mask_zero, weights=[embedding_matrix],
                      trainable=trainable_embed)(inp)
    else:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen, mask_zero=mask_zero)(inp)
    # BiLSTM: 512 fw + 512 bw = 1024 chiều
    x = Bidirectional(LSTM(rnn_units, dropout=0.2, recurrent_dropout=0.0,
                           return_sequences=False))(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid", dtype="float32")(x)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m


def build_textcnn(vocab_size, maxlen=200, embed_dim=1024,
                  filters=256, kernel_sizes=(3,4,5,6), lr=3e-4,
                  embedding_matrix=None, trainable_embed=True):
    inp = Input(shape=(maxlen,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen,
                      mask_zero=False, weights=[embedding_matrix],
                      trainable=trainable_embed)(inp)
    else:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen, mask_zero=False)(inp)
    x = SpatialDropout1D(0.2)(x)

    pools = []
    for k in kernel_sizes:
        c = Conv1D(filters, kernel_size=k, padding="valid", strides=1, use_bias=False)(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        p = GlobalMaxPooling1D()(c)
        pools.append(p)

    # Tổng filters * số kernel = 1024
    h = Concatenate()(pools) if len(pools) > 1 else pools[0]
    h = Dropout(0.5)(h)
    out = Dense(1, activation="sigmoid", dtype="float32")(h)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m


def build_bilstm_cnn(vocab_size, maxlen=200, embed_dim=1024,
                     rnn_units=256, filters=256, kernel_sizes=(3,4,5,6), lr=3e-4,
                     embedding_matrix=None, trainable_embed=True, mask_zero=True):
    inp = Input(shape=(maxlen,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen,
                      mask_zero=mask_zero, weights=[embedding_matrix],
                      trainable=trainable_embed)(inp)
    else:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen, mask_zero=mask_zero)(inp)

    x = Bidirectional(LSTM(rnn_units, dropout=0.2, recurrent_dropout=0.0,
                           return_sequences=True))(x)

    pools = []
    for k in kernel_sizes:
        c = Conv1D(filters, kernel_size=k, padding="valid", strides=1, use_bias=False)(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        p = GlobalMaxPooling1D()(c)
        pools.append(p)

    h = Concatenate()(pools) if len(pools) > 1 else pools[0]
    h = Dropout(0.5)(h)
    out = Dense(1, activation="sigmoid", dtype="float32")(h)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m


def build_lstm_cnn(vocab_size, maxlen=200, embed_dim=1024,
                   rnn_units=256, filters=256, kernel_sizes=(3,4,5,6), lr=3e-4,
                   embedding_matrix=None, trainable_embed=True, mask_zero=True):
    inp = Input(shape=(maxlen,), dtype="int32")
    if embedding_matrix is not None:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen,
                      mask_zero=mask_zero, weights=[embedding_matrix],
                      trainable=trainable_embed)(inp)
    else:
        x = Embedding(vocab_size, embed_dim, input_length=maxlen, mask_zero=mask_zero)(inp)

    # LSTM giữ sequence cho CNN
    x = LSTM(rnn_units, dropout=0.2, recurrent_dropout=0.0, return_sequences=True)(x)

    pools = []
    for k in kernel_sizes:
        c = Conv1D(filters, kernel_size=k, padding="valid", strides=1, use_bias=False)(x)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        p = GlobalMaxPooling1D()(c)
        pools.append(p)

    h = Concatenate()(pools) if len(pools) > 1 else pools[0]
    h = Dropout(0.5)(h)
    out = Dense(1, activation="sigmoid", dtype="float32")(h)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
    return m

# --------------- Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_type", type=str, default="cnn",
                choices=["cnn", "lstm", "bilstm", "bilstm_cnn", "lstm_cnn"])
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--vocab_size", type=int, default=40000)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--rnn_units", type=int, default=1024 , help="Dùng cho (bi)LSTM")
    ap.add_argument("--filters", type=int, default=128, help="Số filters/branch của CNN")
    ap.add_argument("--kernel_sizes", type=str, default="3,4,5", help="Kernel sizes cho CNN, ví dụ '3,4,5'")
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_mixed_precision", action="store_true")
    ap.add_argument("--use_class_weight", action="store_true")

    # New: embedding options
    ap.add_argument("--emb_init", type=str, default="random",
                    choices=["random","gensim_kv","txt_vec"],
                    help="How to initialize embedding layer")
    ap.add_argument("--emb_path", type=str, default=None,
                    help="Path to embeddings.kv (gensim_kv) or embeddings.vec (txt_vec)")
    ap.add_argument("--freeze_embed", action="store_true", help="Freeze embedding layer if set")
    args = ap.parse_args()

    set_seed(args.seed)
    log_visible_gpus()
    if args.use_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print(">> Mixed precision: ON")

    # Paths
    out_dir = args.out_dir
    model_dir = os.path.join(out_dir, "models_visec")
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(out_dir); ensure_dir(model_dir); ensure_dir(log_dir)

    best_model_path = os.path.join(model_dir, f"best_{args.model_type}.h5")
    final_model_path = os.path.join(model_dir, f"final_{args.model_type}.h5")
    tokenizer_path = os.path.join(out_dir, "tokenizer.pkl")
    hist_csv_path = os.path.join(log_dir, f"train_history_{args.model_type}.csv")
    long_sent_log = os.path.join(log_dir, f"long_sentences_{args.model_type}.tsv")

    # Data
    df = pd.read_csv(args.tsv_path, sep="\t")
    # Backward compat with your file which expects 'transcription' and 'emotion_binary'
    if not {"transcription","emotion_binary"}.issubset(df.columns):
        print("Lỗi: TSV cần cột 'transcription' & 'emotion_binary'", file=sys.stderr); sys.exit(1)
    df["transcription"] = df["transcription"].astype(str).map(simple_clean)
    df = df.dropna(subset=["transcription","emotion_binary"]).reset_index(drop=True)

    X = df["transcription"].values
    y = df["emotion_binary"].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    tok = Tokenizer(num_words=args.vocab_size, oov_token="<unk>")
    tok.fit_on_texts(X_train)
    tr_seq = tok.texts_to_sequences(X_train)
    va_seq = tok.texts_to_sequences(X_val)

    with open(long_sent_log, "w", encoding="utf-8") as f:
        f.write("split\tlen\ttext\n")
        for name, seqs, texts in [("train", tr_seq, X_train), ("val", va_seq, X_val)]:
            for s, t in zip(seqs, texts):
                if len(s) > args.maxlen: f.write(f"{name}\t{len(s)}\t{t}\n")

    X_train_pad = pad_sequences(tr_seq, maxlen=args.maxlen, padding="post", truncating="post")
    X_val_pad   = pad_sequences(va_seq, maxlen=args.maxlen, padding="post", truncating="post")
    vocab_eff = min(args.vocab_size, len(tok.word_index)+1)
    print(f">> Vocab size (effective): {vocab_eff}")

    # Build embedding matrix (random / kv / txt)
    embedding_matrix, trainable_embed, mask_zero = build_embedding_matrix(
        tok, vocab_eff, args.embed_dim, args.emb_init, args.emb_path, args.freeze_embed
    )

    # Build model with embedding weights
    if args.model_type == "cnn":
        ks = tuple(int(k.strip()) for k in args.kernel_sizes.split(",") if k.strip())
        model = build_textcnn(vocab_eff, args.maxlen, args.embed_dim, args.filters, ks, args.lr,
                              embedding_matrix=embedding_matrix, trainable_embed=trainable_embed)
    elif args.model_type == "lstm":
        model = build_lstm(vocab_eff, args.maxlen, args.embed_dim, args.rnn_units, args.lr,
                           embedding_matrix=embedding_matrix, trainable_embed=trainable_embed, mask_zero=mask_zero)
    elif args.model_type == "bilstm_cnn":
        ks = tuple(int(k.strip()) for k in args.kernel_sizes.split(",") if k.strip())
        model = build_bilstm_cnn(vocab_eff, args.maxlen, args.embed_dim,
                                 args.rnn_units, args.filters, ks, args.lr,
                                 embedding_matrix=embedding_matrix, trainable_embed=trainable_embed, mask_zero=mask_zero)
    elif args.model_type == "lstm_cnn":
        ks = tuple(int(k.strip()) for k in args.kernel_sizes.split(",") if k.strip())
        model = build_lstm_cnn(vocab_eff, args.maxlen, args.embed_dim,
                            args.rnn_units, args.filters, ks, args.lr,
                            embedding_matrix=embedding_matrix,
                            trainable_embed=trainable_embed,
                            mask_zero=mask_zero)
    else:
        model = build_bilstm(vocab_eff, args.maxlen, args.embed_dim, args.rnn_units, args.lr,
                             embedding_matrix=embedding_matrix, trainable_embed=trainable_embed, mask_zero=mask_zero)
    model.summary()

    # Callbacks
    cbs = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(hist_csv_path, append=False),
    ]

    class_weight = None
    if args.use_class_weight:
        classes = np.array([0,1])
        w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {0: float(w[0]), 1: float(w[1])}
        print(">> class_weight:", class_weight)

    # Train
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=cbs, class_weight=class_weight, verbose=1
    )

    # Save final + tokenizer
    model.save(final_model_path)
    import pickle
    with open(tokenizer_path, "wb") as f:
        pickle.dump({"tokenizer": tok, "maxlen": args.maxlen, "vocab_size": vocab_eff}, f)

    best_idx = int(np.argmin(history.history["val_loss"]))
    print(f">> Done. Best epoch: {best_idx+1} | "
          f"val_loss={history.history['val_loss'][best_idx]:.4f} | "
          f"val_acc={history.history['val_accuracy'][best_idx]:.4f} | "
          f"val_auc={history.history['val_auc'][best_idx]:.4f}")
    print(">> Best:", best_model_path)
    print(">> Final:", final_model_path)
    print(">> Tokenizer:", tokenizer_path)
    print(">> CSV log:", hist_csv_path)

if __name__ == "__main__":
    main()
    
'''
CUDA_VISIBLE_DEVICES=4 taskset -c 24-49 \
python /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/base/train_model_after_eb.py \
   --tsv_path /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/data_after_infer/data_two_cols.tsv \
   --out_dir  /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/exp_kv_lstm-cnn \
   --model_type cnn \
   --maxlen 200 --vocab_size 6000 \
   --embed_dim 1024 \
   --emb_init gensim_kv \
   --emb_path /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/saved/embeddings_visec.kv \
   --filters 256 --kernel_sizes 3,4,5,6 \
   --batch_size 64 --epochs 30 \
   --freeze_embed

CUDA_VISIBLE_DEVICES=4 taskset -c 49-70 \
python /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/base/train_model_after_eb.py \
  --tsv_path /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/data_after_infer/data_two_cols.tsv\
   --out_dir /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/exp_kv_lstm-cnn \
   --model_type bilstm \
   --maxlen 200 \
   --vocab_size 6000 \
   --embed_dim 1024 \
   --emb_init gensim_kv \
   --emb_path /workspace/sanglq/vlsp_2025/sanglq/asr/text-to-emotion/eb/saved/embeddings_visec.kv \
   --rnn_units 1024 \
   --batch_size 64 \
   --epochs 30 \
   --freeze_embed
'''

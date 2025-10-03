CUDA_VISIBLE_DEVICES=4 taskset -c 24-49 \
python DFAT/TEFE/train_embedding.py \
  --tsv_paths data.tsv \
  --text_col transcription \
  --out_dir  DFAT/TEFE/eb/saved \
  --model word2vec --dim 1024 --sg 1 --window 3 --min_count 1 --epochs 10
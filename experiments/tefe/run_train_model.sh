CUDA_VISIBLE_DEVICES=4 taskset -c 49-70 \
python DFAT/TEFE/train_model_after_embedding.py \
  --tsv_path datapath.tsv\
   --out_dir DFAT/TEFE/eb/ \
   --model_type bilstm \
   --maxlen 200 \
   --vocab_size 6000 \
   --embed_dim 1024 \
   --emb_init gensim_kv \
   --emb_path DFAT/TEFE/eb/saved/embeddings_visec.kv \
   --rnn_units 1024 \
   --batch_size 64 \
   --epochs 30 \
   --freeze_embed
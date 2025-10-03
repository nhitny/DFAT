import os
import json
from funasr import AutoModel
from tqdm import tqdm

# Load SER model
model = AutoModel(model="iic/emotion2vec_plus_large")

# Input directory
wav_dir = "/workspace/sanglq/vlsp_2025/vlsp2025_asrser/datasets/private_test"
save_json = "/workspace/sanglq/vlsp_2025/huytq/results/json_out/ser_private_probabilities.json"

ser_results = {}

for fname in tqdm(os.listdir(wav_dir)):
    if fname.endswith(".wav"):
        wav_path = os.path.join(wav_dir, fname)
        try:
            res = model.generate(wav_path, granularity="utterance")
            result = res[0]
            print(result)
            labels = [lbl.split("/")[-1] for lbl in result["labels"]]  # emotion names
            feats = result["feats"].tolist()  # probability values

            ser_results[fname] = {"labels": labels, "feats": feats}

        except Exception as e:
            print(f"Error with {fname}: {e}")

# Save all results for fusion later
with open(save_json, "w") as f:
    json.dump(ser_results, f, indent=2)

print(f"SER probabilities saved to {save_json}")

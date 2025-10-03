import os
import wandb

import torch
import numpy as np
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, concatenate_datasets, Audio
from transformers.trainer_utils import get_last_checkpoint

# Set your WandB API key here
os.environ["WANDB_API_KEY"] = "519a1fadc8c73649e3630c33e948560412005149"
wandb.login()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad and truncate labels to max length = 448
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels

        return batch


def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # Extract input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]

    # Tokenize target text (sentence)
    labels = processor.tokenizer(batch["sentence"]).input_ids

    # Truncate label to max length 448
    if len(labels) > 448:
        labels = labels[:448]

    batch["labels"] = labels
    return batch


def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in the labels as we can't decode them
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id

    # Decode the predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer_metric = evaluate.load("wer")
    wer_result = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_result}

from datasets import load_dataset, concatenate_datasets, Audio
import glob

def load_and_prepare_dataset(processor):
    # Group files by prefix
    data_path = "/dataset/vlsp2025-asr-ser/28k_vietnamese_voice_augmented_of_VigBigData/data"
    data_files = {
        "train_1": sorted(glob.glob(f"{data_path}/train_1-*.parquet")),
        "train_2": sorted(glob.glob(f"{data_path}/train_2-*.parquet")),
        "train_3": sorted(glob.glob(f"{data_path}/train_3-*.parquet")),
        "train_4": sorted(glob.glob(f"{data_path}/train_4-*.parquet")),
        "train_5": sorted(glob.glob(f"{data_path}/train_5-*.parquet")),
        "test": sorted(glob.glob(f"{data_path}/test-*.parquet")),
    }

    # Load all parts
    dataset = load_dataset("parquet", data_files=data_files)
    print(f"\t ---> Loaded splits: {dataset.keys()}")

    # Merge train_* shards
    train_dataset = concatenate_datasets([
        dataset["train_1"],
        dataset["train_2"],
        dataset["train_3"],
        dataset["train_4"],
        dataset["train_5"],
    ])

    # Optional: use part of train for validation
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # Cast audio column
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Map processing
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    val_dataset = val_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=val_dataset.column_names,
        num_proc=4,
    )

    return train_dataset, val_dataset


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    wandb.init(
        project="whisper-28k-vietnamese-VinBigData",
        name=args.output_dir.split("/")[-1],
        config={
            "model": args.model_dir,
            "learning_rate": 3e-5,
            "batch_size": 5,
            "epochs": 10,
            "fp16": True,
            "gradient_accumulation": 2
        }
    )

    processor = WhisperProcessor.from_pretrained(args.model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)

    train_dataset, val_dataset = load_and_prepare_dataset(processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_steps=500,
        # max_steps=4000,
        num_train_epochs=10,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",  # Set to "steps" or "no" for validation
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        # generation_max_length=225,
        save_steps=1000,
        save_total_limit=2,
        eval_steps=1000,
        logging_steps=25,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        tokenizer=processor.feature_extractor,
    )

    # After trainer.train()
    trainer.train()
    
    # Determine where to save
    if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
        save_path = trainer.state.best_model_checkpoint
        print(f"Loading and saving best model from: {save_path}")
        model = WhisperForConditionalGeneration.from_pretrained(save_path)
    else:
        save_path = trainer.state.output_dir
        print(f"No best checkpoint found, saving current model from: {save_path}")
    
    # Save both model and processor to the same path
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)


    wandb.finish()

if __name__ == "__main__":
    main()

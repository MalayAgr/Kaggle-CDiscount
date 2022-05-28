from typing import Callable

import numpy as np
import torch
from datasets import load_metric
from torch import nn
from torch.utils import data
from transformers import (Trainer, TrainingArguments, ViTFeatureExtractor,
                          ViTForImageClassification, load_metric)

from .config import Config
from .dataset import SampleType


def get_metrics():
    metric = load_metric("accuracy")

    def compute_metrics(p: torch.Tensor) -> dict:
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    return compute_metrics


def get_model(cat2idx: dict[str, int], idx2cat: dict[str, str]) -> nn.Module:
    cat2idx = {category: str(idx) for category, idx in cat2idx.items()}
    return ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=Config.PRETRAINED_MODEL,
        num_labels=len(idx2cat),
        id2label=idx2cat,
        label2id=cat2idx,
    )


def training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=Config.BATCH_SIZE,
        evaluation_strategy="steps",
        num_train_epochs=Config.EPOCHS,
        fp16=Config.FP16,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=Config.LR,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True,
    )


def get_trainer(
    model: nn.Module,
    output_dir: str,
    collate_fn: Callable[[tuple[SampleType, ...]], SampleType],
    compute_metrics: Callable[[torch.Tensor], dict],
    train_ds: data.Dataset,
    val_ds: data.Dataset,
    feature_extractor: ViTFeatureExtractor,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args(output_dir),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
    )

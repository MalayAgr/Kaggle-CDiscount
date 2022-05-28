import io

import albumentations as A
import bson
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ViTFeatureExtractor

from src.config import Config

SampleType = dict[str, torch.Tensor | int]


class CDiscountDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds_filepath: str,
        metadata_df: pd.DataFrame,
        has_labels: bool = True,
        transforms: A.Compose = None,
    ) -> None:
        self.f = open(ds_filepath, "rb")
        self.md_df = metadata_df
        self.transforms = transforms
        self.has_labels = has_labels

    def read_image(self, record_metadata: pd.Series) -> Image:
        start: int = record_metadata["start"]
        length: int = record_metadata["length"]
        idx: int = record_metadata["img_idx"]

        self.f.seek(start)
        record = bson.decode(self.f.read(length))

        img = record["imgs"][idx]["picture"]
        return Image.open(io.BytesIO(img))

    def __len__(self) -> int:
        return len(self.md_df)

    def __getitem__(self, idx: int) -> SampleType:
        metadata = self.md_df.iloc[idx]
        img = self.read_image(metadata)

        img = np.array(img)

        if self.transforms is not None:
            img: np.ndarray = self.transforms(image=img)["image"]

        img = np.swapaxes(img, -1, 0)

        example = {"pixel_values": torch.tensor(img)}

        if self.has_labels is True:
            example["label"] = metadata["label"]

        return example


def collate_fn(examples: tuple[SampleType, ...]) -> SampleType:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def get_feature_extractor() -> ViTFeatureExtractor:
    return ViTFeatureExtractor.from_pretrained(Config.PRETRAINED_MODEL)


def resize_transform(feature_extractor: ViTFeatureExtractor) -> A.Resize:
    size = feature_extractor.size
    return A.Resize(size, size, interpolation=cv2.INTER_AREA)


def normalize_transform(
    feature_extractor: ViTFeatureExtractor,
) -> A.Normalize:
    return A.Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std,
        always_apply=True,
    )


def training_transforms(feature_extractor: ViTFeatureExtractor) -> A.Compose:
    size = feature_extractor.size
    return A.Compose(
        [
            resize_transform(feature_extractor),
            A.RandomResizedCrop(size, size, p=0.5),
            A.HorizontalFlip(),
            normalize_transform(feature_extractor),
        ]
    )


def validation_transforms(feature_extractor: ViTFeatureExtractor) -> A.Compose:
    size = feature_extractor.size
    return A.Compose(
        [
            resize_transform(feature_extractor),
            A.CenterCrop(size, size, p=0.5),
            normalize_transform(feature_extractor),
        ]
    )


def test_transforms(feature_extractor: ViTFeatureExtractor) -> A.Compose:
    return A.Compose(
        [resize_transform(feature_extractor), normalize_transform(feature_extractor)]
    )

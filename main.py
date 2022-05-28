from sklearn import model_selection

from src import utils
from src.config import Config
from src.dataset import (
    CDiscountDataset,
    collate_fn,
    get_feature_extractor,
    training_transforms,
    validation_transforms,
)
from src.model import get_metrics, get_model, get_trainer


def main() -> None:
    # Make metadata CSVs
    utils.make_metadata_csv("train.bson")
    utils.make_metadata_csv("test.bson")

    # Go from per product to per image representation
    md = utils.metadata_product2img("train_metadata.csv")

    # Encode labels
    cat2idx, idx2cat = utils.get_category_tables("category_names.csv")

    md = utils.encode_labels(md, cat2idx)

    # Split into training and validation sets
    train_md, val_md = model_selection.train_test_split(
        md, test_size=Config.VAL_SPLIT, shuffle=True, random_state=42
    )

    del md

    # Create datasets
    feature_extractor = get_feature_extractor()

    train_transforms = training_transforms(feature_extractor=feature_extractor)
    val_transforms = validation_transforms(feature_extractor=feature_extractor)

    filepath = Config.get_data_path("train.bson")

    train_ds = CDiscountDataset(
        ds_filepath=filepath, metadata_df=train_md, transforms=train_transforms
    )
    val_ds = CDiscountDataset(
        ds_filepath=filepath, metadata_df=val_md, transforms=val_transforms
    )

    # Init model and trainer
    model = get_model(cat2idx=cat2idx, idx2cat=idx2cat)
    
    trainer = get_trainer(
        model=model,
        output_dir="./vit-base-cdiscount",
        collate_fn=collate_fn,
        compute_metrics=get_metrics(),
        train_ds=train_ds,
        val_ds=val_ds,
        feature_extractor=feature_extractor,
    )


if __name__ == "__main__":
    main()

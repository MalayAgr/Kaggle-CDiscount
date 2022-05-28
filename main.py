from sklearn import model_selection

from src import utils
from src.config import Config
from src.dataset import CDiscountDataset


def main() -> None:
    utils.make_metadata_csv("train.bson")
    utils.make_metadata_csv("test.bson")

    md = utils.metadata_product2img("train_metadata.csv")

    cat2idx, idx2cat = utils.get_category_tables("category_names.csv")
    md = utils.encode_labels(md, cat2idx)

    train_md, val_md = model_selection.train_test_split(
        md, test_size=Config.VAL_SPLIT, shuffle=True, random_state=42
    )

    del md


if __name__ == "__main__":
    main()

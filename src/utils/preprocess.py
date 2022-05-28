import pandas as pd

from ..config import Config


def get_category_tables(filename: str) -> tuple[dict[str, int], dict[str, str]]:
    cat_file = Config.get_data_path(filename)
    df = pd.read_csv(cat_file)
    categories = df["category_id"].astype(str)
    cat2idx = {category: idx for idx, category in enumerate(categories)}
    idx2cat = {str(idx): category for category, idx in cat2idx.items()}
    return cat2idx, idx2cat


def metadata_product2img(filename: str) -> pd.DataFrame:
    filepath = Config.get_data_path(filename, metadata=True)
    md = pd.read_csv(filepath)
    md = md.loc[md.index.repeat(md["n_imgs"])]
    md["img_idx"] = md.groupby("pid").cumcount()
    return md


def encode_labels(metadata: pd.DataFrame, cat2idx: dict[str, int]) -> pd.DataFrame:
    metadata["label"] = metadata["category_id"].map(cat2idx)
    return metadata

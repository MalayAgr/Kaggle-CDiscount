import os


class Config:
    DATA_DIR: str = "./data"

    METADATA_DIR: str = "./data/metadata"

    NPRODS_TRAIN: int = 7_069_896

    NPRODS_TEST: int = 1_768_182

    GC_LIMIT: int = 1000

    GC_DECAY: int = 100_000

    GC_DECAY_AMOUNT: float = 0.99

    PRETRAINED_MODEL: str = "google/vit-base-patch16-224-in21k"

    BATCH_SIZE: int = 16

    EPOCHS: int = 5

    IMG_SIZE: int = 224

    VAL_SPLIT: float = 0.1

    FP16: bool = False
    
    LR: float = 2e-4

    @classmethod
    def get_data_path(cls, filename: str, *, metadata: bool = False) -> str:
        data_dir = cls.DATA_DIR if metadata is False else cls.METADATA_DIR
        return os.path.join(data_dir, filename)

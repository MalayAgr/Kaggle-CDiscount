import gc
import itertools
import os

import bson
import pandas as pd
import tqdm

from ..config import Config

RecordType = tuple[str, int, int, int, str]


def get_metadata(
    filename: str, *, nprods: int = None, has_labels: bool = True
) -> list[RecordType]:
    filepath = Config.get_data_path(filename)

    gc_limit = Config.GC_LIMIT
    decay = Config.GC_DECAY
    decay_amt = Config.GC_DECAY_AMOUNT

    gc_limit_minus_1, decay_minus_1 = gc_limit - 1, decay - 1

    with open(filepath, "rb") as f:
        bdata = bson.decode_file_iter(f)

        if nprods is not None:
            # Limit the number of records
            bdata = itertools.islice(bdata, None, nprods)
        else:
            nprods = Config.NPRODS_TRAIN if "train" in filename else Config.NPRODS_TEST

        pbdata = tqdm(bdata, total=nprods, desc=filename)

        metadata = []

        # Record the current position
        curr = f.tell()

        for idx, d in enumerate(pbdata):
            # Get the starting position
            # And change the current position since once bson decodes a record
            # File pointer has already moved to the next record
            start, curr = curr, f.tell()

            # Get length of record
            length = curr - start

            record = (str(d["_id"]), start, length, len(d["imgs"]))

            if has_labels is True:
                record += (str(d["category_id"]),)

            metadata.append(record)

            # To manage RAM usage
            del d

            # Force garbage collection
            if idx % gc_limit == gc_limit_minus_1:
                gc.collect()

            # Increase the frequency of garbage collection as time progresses
            if idx % decay == decay_minus_1:
                gc_limit *= decay_amt
                gc_limit = int(gc_limit)

        return metadata


def make_metadata_csv(
    filename: str, *, nprods: int = None, has_labels: bool = True
) -> None:
    name, _ = os.path.splitext(filename)
    dest = Config.get_data_path(f"{name}_metadata.csv", metadata=True)

    if os.path.exists(dest):
        overwrite = input(
            "This file already exists. Do you wish to overwrite it? [yY|nN] "
        )

        if overwrite not in ("y", "Y"):
            return

    metadata = get_metadata(filename, nprods=nprods, has_labels=has_labels)

    gc.collect()

    cols = ["pid", "start", "length", "n_imgs"]

    if has_labels is True:
        cols.append("category_id")

    df = pd.DataFrame(metadata, columns=cols)

    df.to_csv(dest, index=False)

    del metadata
    del df
    gc.collect()

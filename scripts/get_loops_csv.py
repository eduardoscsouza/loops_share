import os
import sys
import random
from glob import glob
from math import floor

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.environ["LOOPS_PATH"], "src"))
from utils import *



SETS_SORT_MAP = {"Train":0, "Val":1, "Test":2}
COLUMNS_SORT_ORDER = ["Cross-Val Fold", "Full Class", "Image", "Mask Labeler"]



def get_csv_line(sample_path, processed_dir, original_size_dir):
    sample_name = get_sample_name(sample_path)
    sample_dir = get_sample_class(sample_path, processed_dir)
    sample_dir = os.path.normpath(sample_dir).split(os.sep)
    relpath = os.path.relpath(original_size_dir, processed_dir)

    base_class = sample_dir[0]
    if (base_class == NORMAL_CLASS):
        full_class = os.path.join(*sample_dir)

        img = get_clean_sample_path(sample_path, processed_dir)

        mask = ""
        mask_labeler = ""

        original_img = get_img_path(os.path.join(processed_dir, relpath, full_class, "{}.*".format(sample_name)), processed_dir)
        original_mask = ""
    else:
        full_class = os.path.join(*sample_dir[:-2])

        img = get_img_path(os.path.join(processed_dir, full_class, "imgs", "{}.*".format(sample_name)), processed_dir)

        mask = get_clean_sample_path(sample_path, processed_dir)
        mask_labeler = sample_dir[-1]

        original_img = get_img_path(os.path.join(processed_dir, relpath, full_class, "imgs", "{}.*".format(sample_name)), processed_dir)
        original_mask = get_img_path(os.path.join(processed_dir, relpath, full_class, "masks", mask_labeler, "{}.*".format(sample_name)), processed_dir)

    assert os.path.isfile(os.path.join(processed_dir, img))
    assert os.path.isdir(os.path.join(processed_dir, base_class))
    assert os.path.isdir(os.path.join(processed_dir, full_class))
    assert os.path.isfile(os.path.join(processed_dir, original_img))
    if (base_class != NORMAL_CLASS):
        assert os.path.isfile(os.path.join(processed_dir, mask))
        assert os.path.isdir(os.path.join(processed_dir, full_class, "masks", mask_labeler))
        assert os.path.isfile(os.path.join(processed_dir, original_mask))

    return img, base_class, full_class, mask, mask_labeler, original_img, original_mask



def get_train_splits(class_df, train_split, k_folds):
    imgs_dfs = [img_group for _, img_group in class_df.groupby("Image")]
    random.shuffle(imgs_dfs)

    train = floor(len(imgs_dfs) * train_split)
    train, val = imgs_dfs[:train], imgs_dfs[train:]

    [img_df.insert(0, "Set", "Train") for img_df in train]
    [img_df.insert(0, "Set", "Val") for img_df in val]

    train_val = train + val
    folds_indxs = np.array_split(np.arange(len(train_val)), k_folds)
    train_val = [pd.concat([train_val[indx] for indx in fold_indxs]) for fold_indxs in folds_indxs]

    [fold_df.insert(1, "Cross-Val Fold", fold_i) for fold_df, fold_i in zip(train_val, np.arange(k_folds))]

    return pd.concat(train_val)



if __name__ == '__main__':
    loops_processed_dir = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(os.environ["LOOPS_PATH"], "data", "processed_512")
    loops_original_size_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(os.environ["LOOPS_PATH"], "data", "processed")
    train_split = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.9
    k_folds = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
    test_csv = sys.argv[5] if len(sys.argv) >= 6 else os.path.join(os.environ["LOOPS_PATH"], "data", "test_set.csv")
    seed = int(sys.argv[6]) if len(sys.argv) >= 7 else 17

    random.seed(seed)

    df = []
    queries = [os.path.join(loops_processed_dir, NORMAL_CLASS, "**", "*"),
               os.path.join(loops_processed_dir, POTENTIALLY_MALIGNANT_CLASS, "**", "masks", "*", "*"),
               os.path.join(loops_processed_dir, MALIGNANT_CLASS, "**", "masks", "*", "*")]
    for query in queries:
        all_imgs = [img for img in glob(query, recursive=True) if os.path.isfile(img)]
        assert all([is_valid_img(img) for img in all_imgs])
        df += all_imgs

    df = sorted(df)
    df = [get_csv_line(sample_path, loops_processed_dir, loops_original_size_dir) for sample_path in df]
    df = pd.DataFrame(df, columns=["Image", "Base Class", "Full Class", "Mask", "Mask Labeler", "Original Image", "Original Mask"])

    test_set = pd.read_csv(test_csv)
    test_set = [os.path.splitext(img)[0] for img in test_set["Image"]]
    test_set = [os.path.join(os.path.dirname(img), "imgs", os.path.basename(img)) for img in test_set]
    test_set = [os.path.splitext(img)[0] in test_set for img in df["Image"]]
    test_set = np.asarray(test_set)
    test_df = df[test_set]
    df = df[~test_set]

    test_df.insert(0, "Set", "Test")
    test_df.insert(1, "Cross-Val Fold", "")
    df = [get_train_splits(class_df, train_split, k_folds) for _, class_df in df.groupby("Base Class")]
    df = pd.concat(df)
    df = pd.concat([df, test_df], ignore_index=True)

    df = df.sort_values(COLUMNS_SORT_ORDER)
    df = df.sort_values("Set", kind='stable', key=lambda set_series : [SETS_SORT_MAP[set_split] for set_split in set_series])

    df.to_csv(os.path.join(loops_processed_dir, "loops.csv"), index=False)

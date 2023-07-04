import os
import sys
import tempfile
import importlib
from glob import glob

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight



ACCEPTED_IMAGE_FORMATS = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]

NORMAL_CLASS = "NORMAL"
POTENTIALLY_MALIGNANT_CLASS = "POTENTIALLY MALIGNANT"
MALIGNANT_CLASS = "SCC (MALIGNANT)"



def print_error(filename, error):
    print("From \"{}\":    {}".format(filename, error), file=sys.stderr)



def is_valid_img(img):
    return os.path.isfile(img) and (os.path.splitext(img)[1].lower() in ACCEPTED_IMAGE_FORMATS)

def get_img_path(search_query, processed_dir):
    img = glob(search_query, recursive=True)
    assert len(img) == 1
    assert is_valid_img(img[0])
    return get_clean_sample_path(img[0], processed_dir)



def get_sample_name(sample_path):
    return os.path.splitext(os.path.basename(sample_path))[0]

def get_sample_class(sample_path, processed_dir):
    assert sample_path.startswith(processed_dir)
    sample_class = os.path.dirname(sample_path)[len(processed_dir):]
    sample_class = sample_class[1:] if (sample_class[0] == os.sep) else sample_class
    return sample_class

def get_clean_sample_path(sample_path, processed_dir):
    return os.path.join(get_sample_class(sample_path, processed_dir), os.path.basename(sample_path))

def get_xml_matching_imgs(xml):
    img_path = get_sample_name(xml)
    img_path = glob(os.path.join(os.path.dirname(xml), "{}.*".format(img_path)))
    img_path.remove(xml)

    return img_path



def get_array_of_indices(shape):
    indices = np.indices(shape)
    indices = indices.reshape(indices.shape[0], np.prod(indices.shape[1:])).transpose()
    return indices



def get_clean_df(csv):
    df = pd.read_csv(csv, dtype=str).fillna("")
    df = df[df["Base Class"] != NORMAL_CLASS]
    df = df[df["Set"] != "Test"]

    anna = df["Mask Labeler"] == "anna"
    unnamed = df["Mask Labeler"] == "unnamed"
    assert all((anna & unnamed) == False)
    df = df[anna | unnamed]

    return df



def get_df_class_weights(df, class_col="Base Class", class_mode='categorical'):
    with tempfile.NamedTemporaryFile(suffix=".png") as black_img:
        imageio.imwrite(black_img.name, np.zeros(shape=(16, 16, 3), dtype=np.uint8))

        aux_df = pd.DataFrame.copy(df)
        aux_df["black_img"] = black_img.name
        aux_datagen = ImageDataGenerator().flow_from_dataframe(aux_df, batch_size=64, directory=None, x_col="black_img", y_col=class_col, class_mode=class_mode)

        y_true = np.concatenate([aux_datagen[i][1] for i in range(len(aux_datagen))])
        y_true = np.argmax(y_true, axis=1) if (class_mode=='categorical') else y_true

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_true), y=y_true)
    class_weights = {class_name:class_weight for class_name, class_weight in zip(aux_datagen.class_indices, class_weights)}

    return class_weights

def get_segmentation_class_weights(datagen, class_mode='categorical'):
    counts = {}
    for i in range(len(datagen)):
        y_true = datagen[i][1]
        y_true = np.argmax(y_true, axis=-1) if (class_mode=='categorical') else y_true
        y_true = y_true.flatten()

        batch_ids, batch_counts = np.unique(y_true, return_counts=True)
        for class_id, class_count in zip(batch_ids, batch_counts):
            if class_id in counts.keys():
                counts[class_id] += class_count
            else:
                counts[class_id] = class_count

    n_samples = np.sum(list(counts.values()))
    n_classes = len(counts.keys())
    class_weights = {class_id:(n_samples/(n_classes*class_count)) for class_id, class_count in counts.items()}

    return class_weights



def reload_loops_modules():
    [importlib.reload(mod) for mod in list(sys.modules.values()) if (hasattr(mod, '__file__') and str(mod.__file__).startswith(os.environ["LOOPS_PATH"]))]

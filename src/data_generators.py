import os
import copy
import random

import numpy as np
from tensorflow.keras.utils import Sequence, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class to_categorical(object):
    def __init__(self, n_classes, new_dim=True, dtype=np.float32):
        self.identity = np.identity(n_classes, dtype=dtype)
        self.dims = ... if new_dim else (..., 0)

    def __call__(self, x):
        return self.identity[x[self.dims].astype(np.int32)]



def discretize_mask(mask, thresh=0.5, dtype=np.float32):
    return (mask>thresh).astype(dtype)

def discretize_batch(imgs, masks, thresh=0.5, dtype=np.float32):
    return imgs, discretize_mask(masks, thresh=thresh, dtype=dtype)



def load_img_tf_style(img_path, color_mode='rgb', target_size=None, interpolation='nearest', data_format=None, dtype=None):
    img = load_img(img_path, grayscale=False, color_mode=color_mode, target_size=target_size, interpolation=interpolation, keep_aspect_ratio=False)
    img = img_to_array(img, data_format=data_format, dtype=dtype)

    return img



class BaseSemanticSegmentationGenerator(Sequence):
    def __init__(self, img_col="Image", mask_col="Mask", input_data_dir=os.path.join(os.environ["LOOPS_PATH"], "data", "processed_512"),
                batch_size=4, img_size=(224, 224), mask_size=(224, 224),
                img_gen_kwargs=dict(), mask_gen_kwargs=dict(), img_flow_kwargs=dict(), mask_flow_kwargs=dict(),
                postprocess_func=discretize_batch, seed=None):

        self.base_gen_kwargs = dict(
                            featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            zca_epsilon=1e-06,
                            rotation_range=0,
                            width_shift_range=0.0,
                            height_shift_range=0.0,
                            brightness_range=None,
                            shear_range=0.0,
                            zoom_range=0.0,
                            channel_shift_range=0.0,
                            fill_mode='constant',
                            cval=0.0,
                            horizontal_flip=False,
                            vertical_flip=False,
                            rescale=1.0/255.0,
                            preprocessing_function=None,
                            data_format='channels_last',
                            validation_split=0.0,
                            interpolation_order=1,
                            dtype=None)

        self.img_gen_kwargs = copy.deepcopy(self.base_gen_kwargs)
        self.img_gen_kwargs.update(img_gen_kwargs)

        self.mask_gen_kwargs = copy.deepcopy(self.base_gen_kwargs)
        self.mask_gen_kwargs.update(mask_gen_kwargs)

        self.seed = seed if seed is not None else random.randint(0, 2147483647)
        self.base_flow_kwargs = dict(
                    y=None,
                    batch_size=batch_size,
                    shuffle=True,
                    sample_weight=None,
                    seed=self.seed,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='png',
                    ignore_class_split=False,
                    subset=None)
        self.base_flow_from_dataframe_kwargs = dict(
                    directory=input_data_dir,
                    x_col=img_col,
                    y_col=None,
                    weight_col=None,
                    target_size=img_size,
                    color_mode='rgb',
                    classes=None,
                    class_mode=None,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=self.seed,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='png',
                    subset=None,
                    interpolation='bicubic',
                    validate_filenames=True)
        base_flow_kwargs_complement = {key: val for key, val in self.base_flow_from_dataframe_kwargs.items() if key not in self.base_flow_kwargs.keys()}

        self.img_flow_kwargs = copy.deepcopy(self.base_flow_kwargs)
        self.img_flow_kwargs.update(base_flow_kwargs_complement)
        self.img_flow_kwargs.update(img_flow_kwargs)

        self.mask_flow_kwargs = copy.deepcopy(self.base_flow_kwargs)
        self.mask_flow_kwargs.update(base_flow_kwargs_complement)
        self.mask_flow_kwargs['x_col'] = mask_col
        self.mask_flow_kwargs['target_size'] = mask_size
        self.mask_flow_kwargs['color_mode'] = 'grayscale'
        self.mask_flow_kwargs.update(mask_flow_kwargs)

        assert self.img_flow_kwargs['seed'] == self.mask_flow_kwargs['seed']
        assert self.img_flow_kwargs['shuffle'] == self.mask_flow_kwargs['shuffle']
        assert self.img_flow_kwargs['batch_size'] == self.mask_flow_kwargs['batch_size']

        self.img_gen = ImageDataGenerator(**self.img_gen_kwargs)
        self.mask_gen = ImageDataGenerator(**self.mask_gen_kwargs)

        self.postprocess_func = postprocess_func



    def __get_img_path__(self, img):
        return os.path.join(self.img_flow_kwargs["directory"], img)

    def __get_mask_path__(self, mask):
        return os.path.join(self.mask_flow_kwargs["directory"], mask)

    def __get_all_imgs_paths__(self, df):
        return [self.__get_img_path__(img) for img in df[self.img_flow_kwargs["x_col"]]]

    def __get_all_masks_paths__(self, df):
        return [self.__get_mask_path__(mask) for mask in df[self.mask_flow_kwargs["x_col"]]]



    def __load_img__(self, img):
        return load_img_tf_style(img, color_mode=self.img_flow_kwargs["color_mode"], target_size=self.img_flow_kwargs["target_size"], interpolation=self.img_flow_kwargs["interpolation"], data_format=self.img_gen_kwargs["data_format"], dtype=self.img_gen_kwargs["dtype"])

    def __load_mask__(self, mask):
        return load_img_tf_style(mask, color_mode=self.mask_flow_kwargs["color_mode"], target_size=self.mask_flow_kwargs["target_size"], interpolation=self.mask_flow_kwargs["interpolation"], data_format=self.mask_gen_kwargs["data_format"], dtype=self.mask_gen_kwargs["dtype"])



    def __get_augmented_standardized__(self, img, img_gen):
        params = img_gen.get_random_transform(img.shape)
        img = img_gen.apply_transform(img.astype(img_gen.dtype), params)
        img = img_gen.standardize(img)

        return img

    def __get_augmented_standardized_img__(self, img):
        return self.__get_augmented_standardized__(img, self.img_gen)

    def __get_augmented_standardized_mask__(self, mask):
        return self.__get_augmented_standardized__(mask, self.mask_gen)



class BatchSemanticSegmentationGenerator(BaseSemanticSegmentationGenerator):
    def __init__(self, df, flow_from_disk=False, **kwargs):
        super().__init__(**kwargs)

        if not flow_from_disk:
            imgs = self.__get_all_imgs_paths__(df)
            masks = self.__get_all_masks_paths__(df)

            imgs = np.stack([self.__load_img__(img) for img in imgs])
            masks = np.stack([self.__load_mask__(mask) for mask in masks])

            keys = self.base_flow_kwargs.keys()
            self.img_flow = self.img_gen.flow(imgs, **{key: self.img_flow_kwargs[key] for key in keys})
            self.mask_flow = self.mask_gen.flow(masks, **{key: self.mask_flow_kwargs[key] for key in keys})
        else:
            keys = self.base_flow_from_dataframe_kwargs.keys()
            self.img_flow = self.img_gen.flow_from_dataframe(df, **{key: self.img_flow_kwargs[key] for key in keys})
            self.mask_flow = self.mask_gen.flow_from_dataframe(df, **{key: self.mask_flow_kwargs[key] for key in keys})

        assert self.img_flow.__len__() == self.mask_flow.__len__()

    def __len__(self):
        return self.img_flow.__len__()

    def __getitem__(self, index):
        img, mask = self.img_flow.__getitem__(index), self.mask_flow.__getitem__(index)
        img, mask = (img, mask) if self.postprocess_func is None else self.postprocess_func(img, mask)

        return img, mask



class ImagewiseSemanticSegmentationGenerator(BaseSemanticSegmentationGenerator):
    def __get_imagewise_img__(self, img):
        self.imgs_previous_batches += 1
        if self.img_flow_kwargs['seed'] is not None:
            np.random.seed(self.img_flow_kwargs['seed'] + self.imgs_previous_batches)

        return self.__get_augmented_standardized_img__(self.__load_img__(img))

    def __get_imagewise_mask__(self, mask):
        self.masks_previous_batches += 1
        if self.mask_flow_kwargs['seed'] is not None:
            np.random.seed(self.mask_flow_kwargs['seed'] + self.masks_previous_batches)

        return self.__get_augmented_standardized_mask__(self.__load_mask__(mask))

    def __init__(self, df, flow_imgs_from_disk=False, flow_masks_from_disk=True, **kwargs):
        super().__init__(**kwargs)

        self.imgs = self.__get_all_imgs_paths__(df)
        self.masks = self.__get_all_masks_paths__(df)
        self.imgs_names = [str(img) for img in df[self.img_flow_kwargs["x_col"]]]

        self.flow_imgs_from_disk = flow_imgs_from_disk
        self.flow_masks_from_disk = flow_masks_from_disk

        self.imgs_previous_batches = -1
        self.masks_previous_batches = -1

        self.imgs = self.imgs if self.flow_imgs_from_disk else [self.__get_imagewise_img__(img) for img in self.imgs]
        self.masks = self.masks if self.flow_masks_from_disk else [self.__get_imagewise_mask__(mask) for mask in self.masks]

        assert len(self.imgs) == len(self.masks)
        assert len(self.imgs) == len(self.imgs_names)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, mask, img_name = self.imgs[index], self.masks[index], self.imgs_names[index]

        img = self.__get_imagewise_img__(img) if self.flow_imgs_from_disk else img
        mask = self.__get_imagewise_mask__(mask) if self.flow_masks_from_disk else mask

        img, mask = (img, mask) if self.postprocess_func is None else self.postprocess_func(img, mask)

        return img, mask, img_name

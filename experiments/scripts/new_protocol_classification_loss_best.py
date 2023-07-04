import os
import sys
import gc

import pandas as pd
from tensorflow.keras import applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.join(os.environ["LOOPS_PATH"], "src"))
import data_generators
import model_builders
import training
import utils



exp_name = "new_protocol_classification_loss_best"

input_data_dir = os.path.join(os.environ["LOOPS_PATH"], "data", "processed_512")
df = utils.get_clean_df(os.path.join(input_data_dir, "loops.csv"))
folds = ["0"]

class_weights = utils.get_df_class_weights(df)
df["Weights"] = [class_weights[class_name] for class_name in df["Base Class"]]

input_shape = (224, 224, 3)
augmentation_kwargs = dict(
    #zca_whitening=False,           # May use later
    #zca_epsilon=1e-06,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #brightness_range=(0.8, 1.2),   # (R * f, G * f, B * f)
    shear_range=20,
    zoom_range=0,
    #channel_shift_range=50,        # (R + i, G + i, B + i)
    fill_mode='nearest',            # 'nearest' worked best
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
)
flow_kwargs = dict(
    directory=input_data_dir,
    x_col='Image',
    y_col='Base Class',
    weight_col='Weights',
    target_size=input_shape[:2],
    color_mode='rgb',
    classes=[utils.POTENTIALLY_MALIGNANT_CLASS, utils.MALIGNANT_CLASS],
    class_mode='categorical',
    shuffle=True,
    interpolation='bicubic',
    validate_filenames=True,
)

results_dir = os.path.join(os.environ["LOOPS_PATH"], "experiments", "results")
tensorboard_logdir = os.path.join(os.environ["LOOPS_PATH"], "experiments", "tensorboard_logs")
isic_transfer_dir = os.path.join(os.environ["LOOPS_PATH"], "isic", "experiments", "results", "new_protocol_class_transfer_2019")



weights = None
def get_objs(build_backbone, batch_size, preprocess_input):
    def _get_train_datagen_func(df):
        return ImageDataGenerator(preprocessing_function=preprocess_input, **augmentation_kwargs).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

    def _get_val_datagen_func(df):
        return ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

    def _get_eval_datagen_func(df):
        return _get_val_datagen_func(df)

    def _get_model_func(_):
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
        loss = 'CategoricalCrossentropy'
        metrics = model_builders.get_categorical_classification_metrics()
        weighted_metrics = model_builders.get_categorical_classification_metrics()

        backbone, backbone_extraction_layers = build_backbone(input_shape=input_shape, weights=weights, trainable=True)
        model = model_builders.build_backboned_classifier(backbone, backbone_extraction_layers, output_classes=2, output_activation='softmax')
        model.compile(optimizer, loss, metrics=metrics, weighted_metrics=weighted_metrics)

        return model

    return _get_train_datagen_func, _get_val_datagen_func, _get_eval_datagen_func, _get_model_func

subexp_name = os.path.join(exp_name, "original_unet", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_original_unet_backbone, 16, applications.mobilenet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)

subexp_name = os.path.join(exp_name, "convnext", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_convnextsmall_backbone, 16, applications.mobilenet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)
'''
subexp_name = os.path.join(exp_name, "efficientnet", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_efficientnetv2s_backbone, 16, applications.mobilenet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)

subexp_name = os.path.join(exp_name, "inception", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_inceptionv3_backbone, 16, applications.inception_v3.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)
'''
subexp_name = os.path.join(exp_name, "mobilenet", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_mobilenetv3large_backbone, 16, applications.mobilenet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)
'''
subexp_name = os.path.join(exp_name, "resnet", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnet50v2_backbone, 16, applications.resnet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)

subexp_name = os.path.join(exp_name, "resnetrs", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnetrs50_backbone, 16, applications.resnet_v2.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)

subexp_name = os.path.join(exp_name, "vgg", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_vgg16_backbone, 16, applications.vgg16.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)

subexp_name = os.path.join(exp_name, "xception", f"weights_{weights}")
train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_xception_backbone, 16, applications.xception.preprocess_input)
training.run_cross_validation(subexp_name, model_func,
                            df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                            get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                            get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                            results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                            skip_complete=True, delete_before_run=True)
'''


for weights, subdir in [('imagenet', ''), ('isic2019', 'weights_None'), ('imagenet+isic2019', 'weights_imagenet_frozen_finetuned')]:
    def get_objs(build_backbone, batch_size, preprocess_input, model_family):
        def _get_train_datagen_func(df):
            return ImageDataGenerator(preprocessing_function=preprocess_input, **augmentation_kwargs).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

        def _get_val_datagen_func(df):
            return ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

        def _get_eval_datagen_func(df):
            return _get_val_datagen_func(df)

        def _get_model_func(_):
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
            loss = 'CategoricalCrossentropy'
            metrics = model_builders.get_categorical_classification_metrics()
            weighted_metrics = model_builders.get_categorical_classification_metrics()

            backbone, backbone_extraction_layers = build_backbone(input_shape=input_shape, weights='imagenet', trainable=False)
            if weights != 'imagenet':
                model = model_builders.build_backboned_classifier(backbone, backbone_extraction_layers, output_classes=8, output_activation='softmax')
                model.load_weights(os.path.join(isic_transfer_dir, model_family, subdir, "best_model_weights.h5"))

            model = model_builders.build_backboned_classifier(backbone, backbone_extraction_layers, output_classes=2, output_activation='softmax')
            model.compile(optimizer, loss, metrics=metrics, weighted_metrics=weighted_metrics)

            return model

        return _get_train_datagen_func, _get_val_datagen_func, _get_eval_datagen_func, _get_model_func

    if weights == 'isic2019':
        subexp_name = os.path.join(exp_name, "original_unet", f"weights_{weights}_frozen")
        train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_original_unet_backbone, 16, applications.mobilenet_v2.preprocess_input, "original_unet")
        training.run_cross_validation(subexp_name, model_func,
                                    df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                    get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                    get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                    results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                    skip_complete=True, delete_before_run=True)

    subexp_name = os.path.join(exp_name, "convnext", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_convnextsmall_backbone, 16, applications.mobilenet_v2.preprocess_input, "convnext")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    subexp_name = os.path.join(exp_name, "efficientnet", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_efficientnetv2s_backbone, 16, applications.mobilenet_v2.preprocess_input, "efficientnet")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    subexp_name = os.path.join(exp_name, "inception", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_inceptionv3_backbone, 16, applications.inception_v3.preprocess_input, "inception")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    subexp_name = os.path.join(exp_name, "mobilenet", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_mobilenetv3large_backbone, 16, applications.mobilenet_v2.preprocess_input, "mobilenet")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    subexp_name = os.path.join(exp_name, "resnet", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnet50v2_backbone, 16, applications.resnet_v2.preprocess_input, "resnet")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    subexp_name = os.path.join(exp_name, "resnetrs", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnetrs50_backbone, 16, applications.resnet_v2.preprocess_input, "resnetrs")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    subexp_name = os.path.join(exp_name, "vgg", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_vgg16_backbone, 16, applications.vgg16.preprocess_input, "vgg")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    subexp_name = os.path.join(exp_name, "xception", f"weights_{weights}_frozen")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_xception_backbone, 16, applications.xception.preprocess_input, "xception")
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''


    def get_objs(build_backbone, batch_size, preprocess_input, frozen_subexp_name):
        def _get_train_datagen_func(df):
            return ImageDataGenerator(preprocessing_function=preprocess_input, **augmentation_kwargs).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

        def _get_val_datagen_func(df):
            return ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df, batch_size=batch_size, **flow_kwargs)

        def _get_eval_datagen_func(df):
            return _get_val_datagen_func(df)

        def _get_model_func(fold):
            optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
            loss = 'CategoricalCrossentropy'
            metrics = model_builders.get_categorical_classification_metrics()
            weighted_metrics = model_builders.get_categorical_classification_metrics()

            backbone, backbone_extraction_layers = build_backbone(input_shape=input_shape, weights=None, trainable=True)
            model = model_builders.build_backboned_classifier(backbone, backbone_extraction_layers, output_classes=2, output_activation='softmax')
            model.load_weights(os.path.join(results_dir, frozen_subexp_name, f"fold_{fold}", "best_model_weights.h5"))
            model.compile(optimizer, loss, metrics=metrics, weighted_metrics=weighted_metrics)

            return model

        return _get_train_datagen_func, _get_val_datagen_func, _get_eval_datagen_func, _get_model_func

    if weights == 'isic2019':
        frozen_subexp_name = os.path.join(exp_name, "original_unet", f"weights_{weights}_frozen")
        subexp_name = os.path.join(exp_name, "original_unet", f"weights_{weights}_frozen_finetuned")
        train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_original_unet_backbone, 16, applications.mobilenet_v2.preprocess_input, frozen_subexp_name)
        training.run_cross_validation(subexp_name, model_func,
                                    df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                    get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                    get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                    results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                    skip_complete=True, delete_before_run=True)

    frozen_subexp_name = os.path.join(exp_name, "convnext", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "convnext", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_convnextsmall_backbone, 16, applications.mobilenet_v2.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    frozen_subexp_name = os.path.join(exp_name, "efficientnet", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "efficientnet", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_efficientnetv2s_backbone, 16, applications.mobilenet_v2.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    frozen_subexp_name = os.path.join(exp_name, "inception", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "inception", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_inceptionv3_backbone, 16, applications.inception_v3.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    frozen_subexp_name = os.path.join(exp_name, "mobilenet", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "mobilenet", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_mobilenetv3large_backbone, 16, applications.mobilenet_v2.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''
    frozen_subexp_name = os.path.join(exp_name, "resnet", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "resnet", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnet50v2_backbone, 16, applications.resnet_v2.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    frozen_subexp_name = os.path.join(exp_name, "resnetrs", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "resnetrs", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_resnetrs50_backbone, 16, applications.resnet_v2.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    frozen_subexp_name = os.path.join(exp_name, "vgg", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "vgg", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_vgg16_backbone, 16, applications.vgg16.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)

    frozen_subexp_name = os.path.join(exp_name, "xception", f"weights_{weights}_frozen")
    subexp_name = os.path.join(exp_name, "xception", f"weights_{weights}_frozen_finetuned")
    train_datagen_func, val_datagen_func, eval_datagen_func, model_func = get_objs(model_builders.build_xception_backbone, 16, applications.xception.preprocess_input, frozen_subexp_name)
    training.run_cross_validation(subexp_name, model_func,
                                df=df, folds=folds, training_kwargs=dict(best_model_metric="val_loss", best_model_metric_mode='min'),
                                get_training_train_datagen_func=train_datagen_func, get_training_val_datagen_func=val_datagen_func,
                                get_evaluation_df_func=training.get_categorical_classification_evaluation_df, get_evaluation_train_datagen_func=eval_datagen_func,
                                results_dir=results_dir, tensorboard_logdir=tensorboard_logdir,
                                skip_complete=True, delete_before_run=True)
    '''

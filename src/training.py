import os
import sys
import gc
import copy
import shutil
from time import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import BackupAndRestore, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import clear_session

import utils



def train_model(model, train_datagen, val_datagen,
                epochs=250, epoch_steps=None, val_steps=None,
                base_callbacks=list(),
                tensorboard_logdir="tensorboard_logs",
                best_model_weights_filepath="best_model_weights.h5", best_model_metric="val_imagewise_dice", best_model_metric_mode='max',
                earlystop_metric="loss", earlystop_metric_mode='min', earlystop_min_delta=0.001, earlystop_patience=20,
                backup_dir="backups",
                generator_queue_size=100, generator_workers=os.cpu_count(), use_multiprocessing=True,
                verbose=False):

    if (not use_multiprocessing) and (generator_workers > 1):
        print("WARNING: Using multiple workers without multiprocessing may interfere with random data augmentation if the augmentation utilizes global random number generators", file=sys.stderr)

    callbacks  = copy.copy(base_callbacks)

    callbacks += [TensorBoard(log_dir=tensorboard_logdir,
                            histogram_freq=0,
                            write_graph=False,
                            write_images=False,
                            write_steps_per_second=True,
                            update_freq='epoch',
                            profile_batch=0,
                            embeddings_freq=0,
                            embeddings_metadata=None)]

    callbacks += [ModelCheckpoint(filepath=best_model_weights_filepath,
                                monitor=best_model_metric,
                                mode=best_model_metric_mode,
                                verbose=True,
                                save_best_only=True,
                                save_weights_only=True,
                                save_freq='epoch',
                                options=None,
                                initial_value_threshold=None)]

    callbacks += [EarlyStopping(monitor=earlystop_metric,
                            mode=earlystop_metric_mode,
                            min_delta=earlystop_min_delta,
                            patience=earlystop_patience,
                            verbose=True,
                            baseline=None,
                            restore_best_weights=False)]

    callbacks += [BackupAndRestore(backup_dir,
                                save_freq='epoch',
                                delete_checkpoint=True)]

    model.fit(x=train_datagen,
            y=None,
            batch_size=None,
            class_weight=None,
            sample_weight=None,
            shuffle=True,

            epochs=epochs,
            steps_per_epoch=epoch_steps,
            initial_epoch=0,

            validation_data=val_datagen,
            validation_steps=val_steps,
            validation_batch_size=None,
            validation_freq=1,
            validation_split=0.0,

            max_queue_size=generator_queue_size,
            workers=generator_workers,
            use_multiprocessing=use_multiprocessing,

            callbacks=callbacks,
            verbose=verbose)



def add_metrics_to_conf_matrix_df(df, tn="True Negatives", fp="False Positives", fn="False Negatives", tp="True Positives"):
    tn, fp, fn, tp = df[tn], df[fp], df[fn], df[tp]

    df["Positive IoU"] = tp / (tp + fp + fn)
    df["Negative IoU"] = tn / (tn + fp + fn)
    df["Mean IoU"] = (df["Positive IoU"] + df["Negative IoU"]) / 2.0

    df["Positive Dice"] = (2.0*tp) / ((2.0*tp) + fp + fn)
    df["Negative Dice"] = (2.0*tn) / ((2.0*tn) + fp + fn)
    df["Mean Dice"] = (df["Positive Dice"] + df["Negative Dice"]) / 2.0

    df["True Positive Rate"] = tp / (tp + fn)
    df["True Negative Rate"] = tn / (tn + fp)
    df["Positive Predictive Value"] = tp / (tp + fp)
    df["Negative Predictive Value"] = tn / (tn + fn)

    df["Accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    df["Mean Accuracy"] = (df["True Positive Rate"] + df["True Negative Rate"]) / 2.0

    return df

def get_metrics_df(y_true, y_pred, thresh=0.5):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten()>thresh, y_pred.flatten()>thresh).ravel()

    df = pd.DataFrame([[tn, fp, fn, tp]], columns=["True Negatives", "False Positives", "False Negatives", "True Positives"])
    df = add_metrics_to_conf_matrix_df(df)

    return df

def __get_image_segmentation_evaluation_df__(pred_mask, datagen, i):
    mask, img_name = datagen[i][1:]
    pred_mask = cv2.resize(pred_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC) if (pred_mask.shape[:2] != mask.shape[:2]) else pred_mask

    df = get_metrics_df(mask, pred_mask)
    df.insert(0, "Image", img_name)

    del pred_mask, datagen, i
    del mask, img_name
    gc.collect()

    return df

def get_imagewise_segmentation_evaluation_df(model, datagen, processes=os.cpu_count()):
    datagen_range = range(len(datagen))
    imgs = np.stack([datagen[i][0] for i in datagen_range])
    pred_masks = model.predict(imgs, verbose=False)

    del imgs
    gc.collect()

    with Pool(processes) as pool:
        generator = ((pred_mask, datagen, i) for pred_mask, i in zip(pred_masks, datagen_range))
        df = pool.starmap(__get_image_segmentation_evaluation_df__, generator)
    df = pd.concat(df, ignore_index=True)

    return df

def get_datasetwise_segmentation_evaluation_df(model, datagen):
    datagen = list(datagen)
    imgs = np.concatenate([img for img, _ in datagen])
    masks = np.concatenate([mask for _, mask in datagen])

    del datagen
    gc.collect()

    pred_masks = model.predict(imgs, verbose=False)
    df = get_metrics_df(masks, pred_masks)

    del imgs, masks
    del pred_masks
    gc.collect()

    return df



def __get_classification_evaluation_df__(model, datagen, binary=True, thresh=0.5):
    y_true = np.concatenate([datagen[i][1] for i in range(len(datagen))])
    y_pred = model.predict(datagen, verbose=False)

    if binary:
        average='binary'
        y_true, y_pred = y_true.flatten()>thresh, y_pred.flatten()>thresh

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        df = pd.DataFrame([[tn, fp, fn, tp]], columns=["True Negatives", "False Positives", "False Negatives", "True Positives"])
    else:
        average='macro'
        y_true, y_pred = np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)

        conf_mat = confusion_matrix(y_true, y_pred)
        conf_mat = {f"Confusion Matrix [{i}, {j}]":[conf_mat[i, j]] for i, j in utils.get_array_of_indices(conf_mat.shape)}
        df = pd.DataFrame.from_dict(conf_mat)

    df["Accuracy"] = accuracy_score(y_true, y_pred)
    df["Mean Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    df["Precision"] = precision_score(y_true, y_pred, average=average)
    df["Recall"] = recall_score(y_true, y_pred, average=average)
    df["F1 Score"] = f1_score(y_true, y_pred, average=average)

    return df

def get_binary_classification_evaluation_df(model, datagen, thresh=0.5):
    return __get_classification_evaluation_df__(model, datagen, binary=True, thresh=thresh)

def get_categorical_classification_evaluation_df(model, datagen):
    return __get_classification_evaluation_df__(model, datagen, binary=False)



def run_experiment(exp_name,
                model, training_train_datagen, training_val_datagen, training_kwargs=dict(),
                get_evaluation_df_func=get_imagewise_segmentation_evaluation_df, evaluation_train_datagen=None, evaluation_val_datagen=None,
                results_dir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "results"),
                tensorboard_logdir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "tensorboard_logs"),
                backup_dir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "backups"),
                skip_complete=True, delete_before_run=False):

    t0 = time()

    results_dir = os.path.join(results_dir, exp_name)
    best_model_weights_filepath = os.path.join(results_dir, "best_model_weights.h5")
    metrics_csv_filepath = os.path.join(results_dir, "metrics.csv")
    stdout_filepath = os.path.join(results_dir, "stdout.txt")

    tensorboard_logdir = os.path.join(tensorboard_logdir, exp_name)

    root_backup_dir = backup_dir
    backup_dir = os.path.join(backup_dir, exp_name)

    if skip_complete and os.path.isfile(metrics_csv_filepath):
        return pd.read_csv(metrics_csv_filepath, dtype=str)

    if delete_before_run:
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
        if os.path.isdir(tensorboard_logdir):
            shutil.rmtree(tensorboard_logdir)
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
    else:
        print("WARNING: Not deleting before running. BackupAndRestore doesn't restore the state of other callbacks. Use with caution", file=sys.stderr)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tensorboard_logdir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    base_training_kwargs = dict(best_model_weights_filepath=best_model_weights_filepath,
                                tensorboard_logdir=tensorboard_logdir,
                                backup_dir=backup_dir)
    base_training_kwargs.update(training_kwargs)

    evaluation_train_datagen = evaluation_train_datagen if evaluation_train_datagen is not None else training_train_datagen
    evaluation_val_datagen = evaluation_val_datagen if evaluation_val_datagen is not None else training_val_datagen

    def_stdout = sys.stdout
    with open(stdout_filepath, 'a+') as sys.stdout:
        print(f"Experiment starting at: {t0}")

        train_model(model, training_train_datagen, training_val_datagen, **base_training_kwargs)
        model.load_weights(best_model_weights_filepath)

        train_df = get_evaluation_df_func(model, evaluation_train_datagen)
        val_df = get_evaluation_df_func(model, evaluation_val_datagen)

        train_df.insert(0, "Set", "Train")
        val_df.insert(0, "Set", "Val")
        df = pd.concat([train_df, val_df], ignore_index=True)
        df.to_csv(metrics_csv_filepath, index=False)

        backup_dir = os.path.abspath(backup_dir)
        root_backup_dir = os.path.abspath(root_backup_dir)
        while os.path.isdir(backup_dir) and (not os.listdir(backup_dir)) and (backup_dir != root_backup_dir):
            os.rmdir(backup_dir)
            backup_dir = os.path.abspath(os.path.join(backup_dir, os.pardir))

        tf = time()
        print(f"Experiment finished at: {tf}")
        print(f"Elapsed: {tf - t0}")
    sys.stdout = def_stdout

    return df



def get_fold_datagens(df, fold, fold_col="Cross-Val Fold",
                    get_training_train_datagen_func=None, get_training_val_datagen_func=None,
                    get_evaluation_train_datagen_func=None, get_evaluation_val_datagen_func=None):

    get_training_val_datagen_func = get_training_val_datagen_func if get_training_val_datagen_func is not None else get_training_train_datagen_func

    get_evaluation_train_datagen_func = get_evaluation_train_datagen_func if get_evaluation_train_datagen_func is not None else get_training_train_datagen_func
    get_evaluation_val_datagen_func = get_evaluation_val_datagen_func if get_evaluation_val_datagen_func is not None else get_evaluation_train_datagen_func

    train_indxs = (df[fold_col] != fold)
    train_data, val_data = df[train_indxs], df[~train_indxs]

    training_train_datagen = get_training_train_datagen_func(train_data)
    training_val_datagen = get_training_val_datagen_func(val_data)

    evaluation_train_datagen = get_evaluation_train_datagen_func(train_data)
    evaluation_val_datagen = get_evaluation_val_datagen_func(val_data)

    return training_train_datagen, training_val_datagen, evaluation_train_datagen, evaluation_val_datagen

def run_cross_validation(exp_name, get_model_func,
                        df, folds=None, fold_col="Cross-Val Fold",
                        get_training_train_datagen_func=None, get_training_val_datagen_func=None, training_kwargs=dict(),
                        get_evaluation_df_func=get_imagewise_segmentation_evaluation_df, get_evaluation_train_datagen_func=None, get_evaluation_val_datagen_func=None,
                        results_dir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "results"),
                        tensorboard_logdir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "tensorboard_logs"),
                        backup_dir=os.path.join(os.environ["LOOPS_PATH"], "experiments", "backups"),
                        skip_complete=True, delete_before_run=False):

    metrics_df = []
    folds = folds if folds is not None else sorted(set(df[fold_col]))
    for fold in folds:
        cur_exp_name = os.path.join(exp_name, f"fold_{fold}")

        training_train_datagen, training_val_datagen, \
        evaluation_train_datagen, evaluation_val_datagen = \
        get_fold_datagens(df, fold, fold_col=fold_col,
                        get_training_train_datagen_func=get_training_train_datagen_func, get_training_val_datagen_func=get_training_val_datagen_func,
                        get_evaluation_train_datagen_func=get_evaluation_train_datagen_func, get_evaluation_val_datagen_func=get_evaluation_val_datagen_func)

        model = get_model_func(fold)
        cur_df = run_experiment(cur_exp_name,
                                model, training_train_datagen, training_val_datagen, training_kwargs=training_kwargs,
                                get_evaluation_df_func=get_evaluation_df_func, evaluation_train_datagen=evaluation_train_datagen, evaluation_val_datagen=evaluation_val_datagen,
                                results_dir=results_dir,
                                tensorboard_logdir=tensorboard_logdir,
                                backup_dir=backup_dir,
                                skip_complete=skip_complete, delete_before_run=delete_before_run)

        cur_df.insert(1, "Fold", fold)
        metrics_df += [cur_df]

        del cur_exp_name
        del training_train_datagen, training_val_datagen
        del evaluation_train_datagen, evaluation_val_datagen
        del model, cur_df
        gc.collect()
        clear_session()

    metrics_df = pd.concat(metrics_df, ignore_index=True)
    metrics_df = metrics_df.sort_values("Fold", kind='stable')
    metrics_df = metrics_df.sort_values("Set", kind='stable')
    metrics_df.to_csv(os.path.join(results_dir, exp_name, "metrics.csv"), index=False)

    return metrics_df

import sys
from math import floor, ceil

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout, GlobalAveragePooling2D, Input, Layer, MaxPool2D, Multiply, ZeroPadding2D
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import TrueNegatives, FalsePositives, FalseNegatives, TruePositives, BinaryIoU, CategoricalAccuracy, BinaryAccuracy, Precision, Recall
from tensorflow.keras.applications import ConvNeXtSmall, EfficientNetV2S, InceptionV3, MobileNetV3Large, ResNet50V2, ResNetRS50, VGG16, Xception



class FocalTverskyLoss(Loss):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, epsilon=1e-5, **kwargs):
        super(FocalTverskyLoss, self).__init__(**kwargs)
        self.alpha, self.beta, self.gamma, self.epsilon = alpha, beta, gamma, epsilon

    def call(self, y_true, y_pred):
        tp = tf.math.reduce_sum(y_true * y_pred, axis=[1, 2])           # Get the True Positives for each image for each class
        fn = tf.math.reduce_sum(y_true * (1.0 - y_pred), axis=[1, 2])   # Get the False Negatives for each image for each class
        fp = tf.math.reduce_sum((1.0 - y_true) * y_pred, axis=[1, 2])   # Get the False Positives for each image for each class

        loss = (tp + self.epsilon) / (tp + self.alpha*fn + self.beta*fp + self.epsilon)     # Get the Tversky Index for each image for each class
        loss = tf.math.reduce_mean(loss, axis=-1)                                           # Get the mean Tversky Index between the classes for each image
        loss = 1.0 - loss                                                                   # Invert the Tversky Index to get a loss
        loss = tf.math.pow(loss, self.gamma)                                                # Exponent to get a focal loss

        return loss

class TverskyIndex():
    def __init__(self, alpha=1.0, beta=1.0, thresh=0.5, name=None):
        self.alpha, self.beta, self.thresh = alpha, beta, thresh
        if name is not None:
            self.__name__ = name

    def __call__(self, y_true, y_pred):
        y_true, y_pred = y_true>self.thresh, y_pred>self.thresh

        tp = tf.math.logical_and(y_true, y_pred)
        fn = tf.math.logical_and(y_true, tf.math.logical_not(y_pred))
        fp = tf.math.logical_and(tf.math.logical_not(y_true), y_pred)

        tp = tf.math.reduce_sum(tf.cast(tp, tf.float32), axis=[1, 2])
        fn = tf.math.reduce_sum(tf.cast(fn, tf.float32), axis=[1, 2])
        fp = tf.math.reduce_sum(tf.cast(fp, tf.float32), axis=[1, 2])

        tver = tf.math.divide_no_nan(tp, tp + self.alpha*fn + self.beta*fp)
        tver = tf.math.reduce_mean(tver, axis=-1)

        return tver



class BatchNormedLayer(Layer):
    def __init__(self, layer_class, *args, **kwargs):
        name_kwarg = dict(name=kwargs["name"]) if "name" in kwargs.keys() else dict()
        activation_kwarg = dict(activation=kwargs["activation"]) if "activation" in kwargs.keys() else dict()
        kwargs.pop("name", None)
        kwargs.pop("activation", None)

        super(BatchNormedLayer, self).__init__(**name_kwarg)
        self.layer = layer_class(*args, **kwargs)
        self.batchnorm = BatchNormalization()
        self.actv = Activation(**activation_kwarg)

    def call(self, x):
        x = self.layer(x)
        x = self.batchnorm(x)
        x = self.actv(x)

        return x

class Conv2DBatchNorm(BatchNormedLayer):
    def __init__(self, *args, **kwargs):
        super(Conv2DBatchNorm, self).__init__(Conv2D, *args, **kwargs)

class Conv2DTransposeBatchNorm(BatchNormedLayer):
    def __init__(self, *args, **kwargs):
        super(Conv2DTransposeBatchNorm, self).__init__(Conv2DTranspose, *args, **kwargs)



class AttentionGate(Layer):
    def __init__(self, filters, attention_type='add', activation='relu', **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.theta = Conv2DBatchNorm(filters, 1, strides=1, padding='same', activation=None)
        self.phi = Conv2DBatchNorm(filters, 1, strides=1, padding='same', activation=None)

        self.comb = Add() if (attention_type == 'add') else Multiply()
        self.actv = Activation(activation=activation)
        self.psi = Conv2DBatchNorm(1, 1, strides=1, padding='same', activation='sigmoid')

        self.out = Multiply()

    def call(self, x, gating):
        theta_x = self.theta(x)
        phi_gate = self.phi(gating)

        att = self.comb([theta_x, phi_gate])
        att = self.actv(att)
        att = self.psi(att)

        x = self.out([x, att])

        return x



def build_backboned_unet(backbone, backbone_extraction_layers,
                        bottom_filters=512,
                        upconv_filter_size=3, upconv_batchnorm=True, upconv_dropout=False, upconv_dropout_rate=0.4,
                        convblock_lenght=2, convblock_filter_size=3, convblock_batchnorm=True, convblock_dropout=False, convblock_dropout_rate=0.4,
                        output_filters=1, output_filter_size=3, output_activation='sigmoid',
                        attention_gate=False):

    model_in = backbone.input
    extraction_layers = list(reversed(backbone_extraction_layers))

    upconv_layer_type = Conv2DTransposeBatchNorm if upconv_batchnorm else Conv2DTranspose
    convblock_layer_type = Conv2DBatchNorm if convblock_batchnorm else Conv2D

    expanding = extraction_layers[0]
    skips = extraction_layers[1:]
    depths = reversed(range(len(skips)))
    for skip, depth in zip(skips, depths):
        prefix = f"expanding_depth{depth}"
        filters = int(max(bottom_filters // (2**(len(skips)-depth-1)), 1))

        expanding = upconv_layer_type(filters, upconv_filter_size, strides=2, padding='same', activation='relu', name=f"{prefix}_upconv")(expanding)
        if upconv_dropout:
            expanding = Dropout(upconv_dropout_rate, name=f"{prefix}_upconv_dropout")(expanding)

        if attention_gate:
            skip = AttentionGate(filters, name=f"{prefix}_attention")(skip, expanding)
        expanding = Concatenate(axis=-1, name=f"{prefix}_concat")([expanding, skip])

        for i in range(convblock_lenght):
            expanding = convblock_layer_type(filters, convblock_filter_size, strides=1, padding='same', activation='relu', name=f"{prefix}_conv{i}")(expanding)
            if convblock_dropout:
                expanding = Dropout(convblock_dropout_rate, name=f"{prefix}_conv{i}_dropout")(expanding)

    model_out = Conv2D(output_filters, output_filter_size, strides=1, padding='same', activation=output_activation, name="output")(expanding)
    model = Model(inputs=model_in, outputs=model_out)

    return model

def build_original_unet_backbone(input_shape=(224, 224, 3), max_depth=5,
                                bottom_filters=512,
                                upconv_filter_size=3, upconv_batchnorm=True, upconv_dropout=False, upconv_dropout_rate=0.4,
                                convblock_lenght=2, convblock_filter_size=3, convblock_batchnorm=True, convblock_dropout=False, convblock_dropout_rate=0.4,
                                output_filters=1, output_filter_size=3, output_activation='sigmoid',
                                attention_gate=False,
                                weights='imagenet', trainable=True):

    model_in = Input(shape=input_shape, name="input")
    extraction_layers = []

    convblock_layer_type = Conv2DBatchNorm if convblock_batchnorm else Conv2D

    contracting = model_in
    for depth in range(max_depth):
        prefix = f"contracting_depth{depth}"
        filters = int(max(bottom_filters // (2**(max_depth-1-depth-1)), 1))

        for i in range(convblock_lenght):
            contracting = convblock_layer_type(filters, convblock_filter_size, strides=1, padding='same', activation='relu', name=f"{prefix}_conv{i}")(contracting)
            if convblock_dropout:
                contracting = Dropout(convblock_dropout_rate, name=f"{prefix}_conv{i}_dropout")(contracting)

        extraction_layers += [contracting]
        contracting = MaxPool2D(pool_size=2, strides=2, padding='same', name=f"{prefix}_pool")(contracting)

    model = Model(inputs=model_in, outputs=contracting)

    return model, extraction_layers

def build_original_unet(input_shape=(224, 224, 3), max_depth=5,
                        bottom_filters=512,
                        upconv_filter_size=3, upconv_batchnorm=True, upconv_dropout=False, upconv_dropout_rate=0.4,
                        convblock_lenght=2, convblock_filter_size=3, convblock_batchnorm=True, convblock_dropout=False, convblock_dropout_rate=0.4,
                        output_filters=1, output_filter_size=3, output_activation='sigmoid',
                        attention_gate=False):

    kwargs = dict(bottom_filters=bottom_filters,
                upconv_filter_size=upconv_filter_size, upconv_batchnorm=upconv_batchnorm, upconv_dropout=upconv_dropout, upconv_dropout_rate=upconv_dropout_rate,
                convblock_lenght=convblock_lenght, convblock_filter_size=convblock_filter_size, convblock_batchnorm=convblock_batchnorm, convblock_dropout=convblock_dropout, convblock_dropout_rate=convblock_dropout_rate,
                output_filters=output_filters, output_filter_size=output_filter_size, output_activation=output_activation,
                attention_gate=attention_gate)

    model, extraction_layers = build_original_unet_backbone(input_shape=input_shape, max_depth=max_depth, **kwargs)
    model = build_backboned_unet(model, extraction_layers, **kwargs)

    return model



def build_backboned_classifier(backbone, backbone_extraction_layers,
                            output_classes=1, output_activation='sigmoid'):

    model_in = backbone.input
    model = backbone_extraction_layers[-1]

    if model is not backbone.output:
        print("WARNING: Final extraction layer is not the backbone's final output", file=sys.stderr)

    model = GlobalAveragePooling2D(name="glob_avg_pool")(model)
    model = Dense(output_classes, activation=output_activation, name="output")(model)
    model = Model(inputs=model_in, outputs=model)

    return model



def pad_extraction_layers(input_shape, extraction_layers):
    correct_shapes = 2**np.arange(len(extraction_layers))
    correct_shapes = np.repeat(correct_shapes[..., np.newaxis], 2, axis=1)
    correct_shapes = np.asarray(input_shape[:2])[np.newaxis, ...] / correct_shapes
    correct_shapes = correct_shapes.astype(np.int32)

    padded_extraction_layers = []
    for layer, correct_shape in zip(extraction_layers, correct_shapes):
        layer_shape = layer.shape[1:3]
        height_diff, width_diff = correct_shape[0]-layer_shape[0], correct_shape[1]-layer_shape[1]
        if (height_diff != 0) or (width_diff != 0):
            top_pad, bottom_pad = int(floor(height_diff/2.0)), int(ceil(height_diff/2.0))
            left_pad, right_pad = int(floor(width_diff/2.0)), int(ceil(width_diff/2.0))
            layer = ZeroPadding2D(((top_pad, bottom_pad), (left_pad, right_pad)))(layer)

        padded_extraction_layers += [layer]

    return padded_extraction_layers

def build_convnextsmall_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["tf.__operators__.add_2",
                        "tf.__operators__.add_5",
                        "tf.__operators__.add_32",
                        "layer_normalization"]

    model = ConvNeXtSmall(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    model.trainable = trainable
    extraction_layers = [model.input, AveragePooling2D(pool_size=2, strides=2, padding='same')(model.input)] + [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_efficientnetv2s_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["block1b_add",
                        "block2d_add",
                        "block4a_expand_activation",
                        "block6a_expand_activation",
                        "top_activation"]

    model = EfficientNetV2S(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_inceptionv3_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["activation_2",
                        "activation_4",
                        "activation_28",
                        "activation_74",
                        "mixed10"]

    model = InceptionV3(input_shape=input_shape, include_top=False, weights=weights)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]
    extraction_layers = pad_extraction_layers(input_shape, extraction_layers)

    return model, extraction_layers

def build_mobilenetv3large_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["re_lu_2",
                        "re_lu_6",
                        "multiply_1",
                        "multiply_13",
                        "multiply_19"]

    model = MobileNetV3Large(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_resnet50v2_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["conv1_conv",
                        "conv2_block3_1_relu",
                        "conv3_block4_1_relu",
                        "conv4_block6_1_relu",
                        "post_relu"]

    model = ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_resnetrs50_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["stem_1_stem_act_3",
                        "BlockGroup3__block_0__act_1",
                        "BlockGroup4__block_0__act_1",
                        "BlockGroup5__block_0__act_1",
                        "BlockGroup5__block_2__output_act"]

    model = ResNetRS50(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_vgg16_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["block1_conv2",
                        "block2_conv2",
                        "block3_conv3",
                        "block4_conv3",
                        "block5_conv3"]

    model = VGG16(input_shape=input_shape, include_top=False, weights=weights)
    model.trainable = trainable
    extraction_layers = [model.get_layer(layer).output for layer in extraction_layers]

    return model, extraction_layers

def build_xception_backbone(input_shape=(224, 224, 3), weights='imagenet', trainable=True):
    extraction_layers = ["block2_sepconv2_bn",
                        "block3_sepconv2_bn",
                        "block4_sepconv2_bn",
                        "block13_sepconv2_bn",
                        "block14_sepconv2_act"]

    model = Xception(input_shape=input_shape, include_top=False, weights=weights)
    model.trainable = trainable
    extraction_layers = [model.input] + [model.get_layer(layer).output for layer in extraction_layers]
    extraction_layers = pad_extraction_layers(input_shape, extraction_layers)

    return model, extraction_layers



def get_binary_segmentation_metrics():
    return [TrueNegatives(name="true_negatives"), FalsePositives(name="false_positives"), FalseNegatives(name="false_negatives"), TruePositives(name="true_positives"),
            BinaryIoU(target_class_ids=[1], name="positive_iou"), BinaryIoU(target_class_ids=[0], name="negative_iou"), BinaryIoU(name="mean_iou"),
            TverskyIndex(alpha=1.0, beta=1.0, thresh=0.5, name="imagewise_iou"), TverskyIndex(alpha=0.5, beta=0.5, thresh=0.5, name="imagewise_dice"),
            BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall")]

def build_compile_backboned_unet(backbone, backbone_extraction_layers,
                                optimizer='Adam', loss='BinaryCrossentropy',
                                **kwargs):
    model = build_backboned_unet(backbone, backbone_extraction_layers, **kwargs)
    model.compile(optimizer, loss, metrics=get_binary_segmentation_metrics())

    return model



def get_binary_classification_metrics():
    return [TrueNegatives(name="true_negatives"), FalsePositives(name="false_positives"), FalseNegatives(name="false_negatives"), TruePositives(name="true_positives"),
            BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall")]

def get_categorical_classification_metrics():
    return [CategoricalAccuracy(name="accuracy")]

def build_compile_backboned_classifier(backbone, backbone_extraction_layers,
                                    optimizer='Adam', loss='BinaryCrossentropy',
                                    binary=True,  weighted_metrics=False,
                                    **kwargs):

    loss = 'CategoricalCrossentropy' if ((loss=='BinaryCrossentropy') and (not binary)) else loss
    metrics = get_binary_classification_metrics() if binary else get_categorical_classification_metrics()
    weighted_metrics =  None if (not weighted_metrics) else (get_binary_classification_metrics() if binary else get_categorical_classification_metrics())

    model = build_backboned_classifier(backbone, backbone_extraction_layers, **kwargs)
    model.compile(optimizer, loss, metrics=metrics, weighted_metrics=weighted_metrics)

    return model

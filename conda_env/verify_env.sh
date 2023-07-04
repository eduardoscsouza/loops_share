#!/bin/bash

python -c "
print(\"################################################################################################\")
import tensorflow as tf

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

print(\"\")
print(tf.config.list_logical_devices())
print(tf.__version__)
print(\"################################################################################################\n\n\")
"



python -c "
print(\"################################################################################################\")
import os
import shutil
import urllib.request
import torch

urllib.request.urlretrieve(\"https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py\", \"/tmp/pytorch_mnist.py\")
with open(\"/tmp/pytorch_mnist.py\", \"r+\") as file:
    filedata = file.read()
    filedata = filedata.replace(\"\'--epochs\', type=int, default=14\", \"\'--epochs\', type=int, default=3\")
    filedata = filedata.replace(\"datasets.MNIST(\'../data\'\", \"datasets.MNIST(\'/tmp/mnist_data\'\")
    file.seek(0)
    file.write(filedata)
    file.truncate()

import sys
sys.path.append(\"/tmp/\")
import pytorch_mnist
pytorch_mnist.main()

os.remove(\"/tmp/pytorch_mnist.py\")
shutil.rmtree(\"/tmp/mnist_data/\")

print(\"\")
print(torch.cuda.is_available())
print(torch.__version__)
print(\"################################################################################################\n\n\")
"



python -c "
print(\"################################################################################################\")
import os
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
os.remove(\"yolov5s.pt\")

print(\"\")
model('https://ultralytics.com/images/zidane.jpg').print()
print(\"################################################################################################\n\n\")
"



python -c "
print(\"################################################################################################\")
import tensorrt

assert tensorrt.Builder(tensorrt.Logger())

print(\"\")
print(tensorrt.__version__)
print(\"################################################################################################\n\n\")
"

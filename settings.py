import os
import tensorflow as tf


# Sets the number of threads
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(12)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
Parameters
rgb: are images treated as rgb or grayscale
strategy: strategy to use for the transfered model: can be fine_tuning, feature_extraction or partial_fine_tuning
fixed-layers : only used with partial fine-tuning, wil define how many layers will stay untouched
epochs: how many epochs to go through
batch_size: how many img by batch size
img_size: self explanatory
"""



experiment_name = "TP - Transfer Learning"
params = {
    "rgb": False,
    "include_top": False,
    "input_tensor": None,
    "classifier_activation": "softmax",
    "alpha": 0.35,
    "weights": "imagenet",
    "strategy": "partial_fine_tuning",
    "fixed_layers": 100,
    "epochs": 10,
    "batch_size": 100,
    "img_size": 224,
    "input_shape": (224, 224, 3),
    "pooling": "avg",
    "equilibrate": True,
    "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
    "loss": "categorical_crossentropy",
    "data_augmentation": True,
    "include_preprocessing": False,
    "model": "mobilenetv2"
}
model_name = "Transfer Learning - MobileNetV2"
testing_cycle = 1

added_layers = [
    {
        "type": "dropout",
        "count": 0.3,
        "activation": None
    },
    {
        "type": "dense",
        "count": 128,
        "activation": "relu"
    },
    {
        "type": "dropout",
        "count": 0.2,
        "activation": None
    },
    {
        "type": "dense",
        "count": 2,
        "activation": "softmax"
    },
]

folders = {
    "train": {
        "input": "data/2/chest_xray/train",
        "output": "data/processed/train"
    },
    "test": {
        "input": "data/2/chest_xray/test",
        "output": "data/processed/test"
    },
    "val": {
        "input": "data/2/chest_xray/val",
        "output": "data/processed/val"
    }
}
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
    "strategy": "feature_extraction",
    "fixed-layers": 5,
    "epochs": 10,
    "batch_size": 100,
    "img_size": 224,
    "input_shape": (224, 224, 3),
    "pooling": "avg",
    "equilibrate": True,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "data_augmentation": True,
    "include_preprocessing": False,
    "model": "mobilenetv2"
}
model_name = "Transfer Learning - MobileNetV2"

added_layers = [
    {
        "type": "flatten",
        "count": None,
        "activation": None
    },
    {
        "type": "dense",
        "count": 64,
        "activation": "relu"
    },
    {
        "type": "dropout",
        "count": 0.5,
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
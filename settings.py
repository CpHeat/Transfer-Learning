"""
Parameters
rgb: are images treated as rgb or grayscale
strategy: strategy to use for the transfered model: can be fine_tuning, feature_extraction or partial_fine_tuning
fixed-layers : only used with partial fine-tuning, wil define how many layers will stay untouched
epochs: how many epochs to go through
batch_size: how many img by batch size
img_size: self explanatory
"""
params = {
    "rgb": True,
    "strategy": "feature_extraction",
    "fixed-layers": 5,
    "epochs": 10,
    "batch_size": 200,
    "img_size": 200,
    "input_shape": (200, 200, 3),
    "pooling": "avg",
    "equilibrate": True
}
model_name = "VGG16"

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
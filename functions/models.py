import tensorflow as tf
from tensorflow.keras import layers, models
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2
from keras.applications import MobileNetV2
from keras.applications import DenseNet121
from keras.applications import EfficientNetV2B2
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Recall, Precision

from settings import params, added_layers, model_name


MODEL_CLASSES = {
    "vgg16": VGG16,
    "resnet50v2": ResNet50V2,
    "mobilenetv2": MobileNetV2,
    "densenet121": DenseNet121,
    "efficientnetv2-b2": EfficientNetV2B2,
}

MODEL_SPECIFIC_PARAMS = {
    "mobilenetv2": {
        "alpha": params["alpha"],
    },
    "efficientnetv2-b2": {
        "include_preprocessing": params["include_preprocessing"],
    }
}

if params["input_tensor"]:
    input_tensor = Input(shape=params["input_tensor"])
else:
    input_tensor = None

COMMON_PARAMS = {
    "include_top": params["include_top"],
    "weights": params["weights"],
    "input_tensor": input_tensor,
    "input_shape": params["input_shape"],
    "pooling": params["pooling"],
    "classes": 1000,
    "classifier_activation": params["classifier_activation"],
    "name": params["model"],
}

def initialize_model():
    #Pour le modele VGG16 on suprime la couche supérieur de classification

    """
    le transfer learning evite l'overfitting
    choix du mopdèle préentrainé:
    -VGG-16

    entrée couleur 224*224


    Bloc 1 : Convolution pour repérer les features
    couche relu come fonction d'activation
    pooling entre deux groupes de couches de covnolution (soit 2x2 sans chevauchement soit 3x3 avec chevauxhement)
    Répéter ? selon resultats
    ameliore l'efficacité et evite le suraprentissage

    parametres convolution:
    nb de filtres K
    taile des filtres F
    pas S
    zero-padding P

    Pour la couche de convolution, les filtres sont de petite taille et glissés sur l'image d'un pixel à la fois. La valeur du zero-padding est choisie de sorte que la largeur et la hauteur du volume en entrée ne soient pas modifiées en sortie. En général, on choisit alors F=3,P=1,S=1
    ou F=5,P=2,S=1

    P = (F-1)/2
    S = 1

    parametres pooling:
    taille des cellules F
    pas S

    Pour la couche de pooling, F=2
    et S=2
    est un choix judicieux. Cela permet d'éliminer 75% des pixels en entrée. On peut également trouver F=3
    et S=2
    : dans ce cas, les cellules se chevauchent. Choisir des cellules de plus grande taille provoque une perte trop importante d'informations, et donne de moins bons résultats en pratique

    Flatten
    Bloc final : Dense - fonction logistique pour classification binaire
    """
    model_class = MODEL_CLASSES[params["model"].lower()]
    model_params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS.get(params["model"], {})}   

    base_model = model_class(**model_params)

    # Adjust to chosen strategy
    if params["strategy"] == "fine_tuning":
        for layer in base_model.layers:
            layer.trainable = True
    elif params["strategy"] == "partial_fine_tuning":
        base_model.trainable = True
        for layer in base_model.layers[:params["fixed_layers"]]:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False

    model = models.Sequential()
    model.add(base_model)
    for added_layer in added_layers:
        add_layer(model, added_layer["type"], added_layer["count"], added_layer["activation"])    

    # Compiler le modèle 
    model.compile(optimizer=params["optimizer"],
                loss=params["loss"],
                metrics=[
                    Recall(class_id=0, name='recall_normal'),
                    Recall(class_id=1, name='recall_pneumonia'),
                    Precision(class_id=0, name='precision_normal'),
                    Precision(class_id=1, name='precision_pneumonia'),
                    'accuracy',
                    'auc',
                    'mean_squared_error'
                ])
    
    return model

def initialize_torchxrayvision_model():
    pass

def add_layer(model, type, count, activation):
        layer = None
        if type == "flatten":
            model.add(layers.Flatten())
        elif type == "dense":
            model.add(layers.Dense(count, activation=activation))
        elif type == "dropout":
            model.add(layers.Dropout(count))

        return layer
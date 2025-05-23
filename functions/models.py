import tensorflow as tf
from tensorflow.keras import layers, models
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2
from keras.applications import MobileNetV2
from keras.applications import DenseNet121
from keras.applications import EfficientNetB1

from settings import params


def initialize_vgg16_model():
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

    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=params["input_shape"],
        pooling=params["pooling"],
        classes=1000,
        classifier_activation="softmax",
        name="vgg16",
    )

    # Adjust to chosen strategy
    if params["strategy"] == "fine_tuning":
        for layer in base_model.layers:
            layer.trainable = True
    elif params["strategy"] == "partial_fine_tuning":
        for layer in base_model.layers[:params["fixed-layers"]]:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # Classification binaire
    ])

    # Compiler le modèle 
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['recall', 'accuracy', 'precision', 'auc', 'mean_squared_error'])
    
    return model

def initialize_resnet50_model():
    ResNet50V2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=params["input_shape"],
        pooling=params["pooling"],
        classes=1000,
        classifier_activation="softmax",
        name="resnet50v2",
    )

    

def initialize_mobilnetv2_model():
    MobileNetV2(
        input_shape=params["input_shape"],
        alpha=1.0,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=params["pooling"],
        classes=1000,
        classifier_activation="softmax",
        name="mobilenetv2",
    )

def initialize_densenet121_model():
    DenseNet121(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=params["input_shape"],
        pooling=params["pooling"],
        classes=1000,
        classifier_activation="softmax",
        name="densenet121",
    )

def initialize_efficientnetb1_model():
    EfficientNetB1(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=params["input_shape"],
        pooling=params["pooling"],
        classes=1000,
        classifier_activation="softmax",
        name="efficientnetb1",
    )

def initialize_torchxrayvision_model():
    pass
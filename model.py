from keras import layers
from keras.models import Model #Le modèle est une instance de la classe Model de Keras qui définit l'architecture du réseau neuronal et compile les paramètres d'apprentissage
import string
#réseau neurones pour la reconnaissance de caractères

#symbols = string.ascii_lowercase + "0123456789"
symbols = "bcdefgmnpwxy" + "2345678"
num_symbols = len(symbols)
img_shape = (50, 200, 1) # matrice d'une image

def create_model():
    img = layers.Input(shape=img_shape) # layers.Input couche KERAS  pour definir l'entrée vers reseau neuronne, ici matrice d'une image
    # une couche de convolution en 2D avec 16 filtres(ou canaux), chaque filtre est une matrice 3x3.padding='same' signifie que la sortie aura la même taille que l'entrée
    #La sortie de cette couche est stockée dans conv1. L'activation est fait par relu(non-linéarité contairement à sigmoid)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    # mp1 est la sortie de la couche de maxpooling appliquée sur conv1
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)#technique couramment utilisée dans les réseaux de neurones pour accélérer l'entraînement et améliore la stabilité du modèle
    #transforme mp3(tableaux multi-dimensionnel) en un vecteur unidimensionnel pour la suite du modèle.
    #mp3-> dimension 3 de taille (7, 25, 32) -> une image de 7 pixels de largeur, 25 pixels de hauteur et 32 canaux(ou filtres).
    mp3 = layers.MaxPooling2D(padding='same')(bn)
    #Flatten prend le tenseur mp3 de dimension 3 et le transforme en un tenseur de dimension 1 (ou vecteur) de taille (7 * 25 * 32) = 5600)
    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        #applique la fonction d'activation relu aux sorties  des neurones de flat.Permet de modéliser des relations non-linéaires entre les différentes caractéristiques extraites par les couches de convolution précédentes.
        #est une couche dense(fully connected layer) avec 64 neurones et une fonction d'activation relu.
        dens1 = layers.Dense(64, activation='relu')(flat)
        #couche de dropout qui désactive aléatoirement la moitié des neurones de la couche précédente (dens1 ) pendant l'entraînement.technique  utilisée pour éviter le surapprentissage
        drop = layers.Dropout(0.5)(dens1)#0.5 c'est la proba de désactivation.

        # utilise la fonction d'activation "sigmoid" pour produire des valeurs de sortie entre 0 et 1. Le nombre de neurones dans cette couche=  nombre de symboles(num_symbols)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)
        #relu est utilisée dans dens1 pour la non-linéarité dans le réseau, alors que le sigmoid est utilisée dans la couche de sortie (res) pour  une prédiction binaire(comprise entre 0 et 1).


        #liste qui contiendra les sorties de chaque couche de classification.il y a 5 couches de classification, donc outs aura une longueur de 5.
        # Chaque élément de outs sera un multi-mtrx contenant les sorties de la couche de classification correspondante, qui seront des probabilités pour chaque carac possible.
        # Ces sorties seront utilisées pour dire quelle est la prédiction finale de l'ensemble du modèle.
        outs.append(res)
    #crée un modèle qui prend en entrée img (l'image) et renvoie une liste de outs qui contiennent les prédictions pour chaque caractère dans l'image
    model = Model(img, outs)

    #configurer les paramètres d'apprentissage du modèle
    #loss-> fonction de perte utilisée pour évaluer l'écart entre les prédictions du modèle et les vraies valeurs. Ici,la perte est la "cross-entropy catégorielle"
    #optimizer-> l'algorithme d'optimisation utilisé pour mettre à jour les poids du modèle pendant l'apprentissage. Ici, l'optimisateur est "Adam",  un algorithme d'optimisation adaptatif couramment utilisé.
    #metrics-> la métrique utilisée pour évaluer la performance du modèle.Ici, la métrique est l'exactitude (accuracy), qui est une mesure couramment utilisée pour la classification multiclasse.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


    #envoie le modèle compilé
    return model

import os
import cv2
import numpy as np
import string


# Init main values
#symbols = string.ascii_lowercase + "0123456789"  # All symbols captcha can contain
#num_symbols = len(symbols)

symbols = "bcdefgmnpwxy" + "2345678"
num_symbols = len(symbols)
#taille des captchas, 200x50, le 1 est le profondeur car c'est un gris, pas rgb
img_shape = (50, 200, 1)

def preprocess_data():
    n_samples = len(os.listdir('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples'))
    X = np.zeros((n_samples, 50, 200, 1))  # matrice des 0's, crée un tableau numpy(matrice) de dimension 1040*50*200*1
    y = np.zeros((5, n_samples, num_symbols))  # matrice des 0's, de taille 5*1040*36, utilisé pour stocker les étiquettes de chaque image captcha

    for i, pic in enumerate(os.listdir('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples')):
        # Read image as grayscale
        img = cv2.imread(
            os.path.join('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples', pic),
            cv2.IMREAD_GRAYSCALE)   # crée une image niveau de gris
        pic_target = pic[:-4] #ne prend pas le '.png'
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0  #normalisation-> chaque pixel est divisée par 255, donc on a un pixel soit 0 soit 1.
            img = np.reshape(img, (50, 200, 1)) # transforme l'image 2D en une matrice avec prof 1, expliqué avant pour X et Y
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))#matrice utilisé pour stocker la classification,chaque ligne représente une position dans le code captcha à prédire, et chaque colonne représente un symbole possible pour cette position.
            for j, l in enumerate(pic_target):
                ind = symbols.find(l) #si carac dans la liste des symboles
                targs[j, ind] = 1 # on aura un vecteur de type  one-hot p.e. [0,0,0,1,0] etc.
            X[i] = img # on sauvegarde l'image normalisée et transformé
            y[:, i] = targs # on ramplie la colonne i de y avec les valeurs targs

    # Return final data
    return X, y

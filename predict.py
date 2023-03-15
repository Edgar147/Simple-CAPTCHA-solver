import string

import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model("captcha_model.h5")

# Define symbols
#symbols = string.ascii_lowercase + "0123456789"
symbols = "bcdefgmnpwxy" + "2345678"

# Define image shape
img_shape = (50, 200, 1)


def predict_captcha(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Image not detected.")
        return

    # Reshape image
    img = np.reshape(img, (1, *img_shape))

    # Predict captcha
    prediction = model.predict(img)

    # Get indexes of predicted symbols
    symbol_indexes = [np.argmax(prediction[i]) for i in range(5)]

    # Decode predicted captcha
    predicted_captcha = ''.join([symbols[symbol_indexes[i]] for i in range(5)])

    return predicted_captcha

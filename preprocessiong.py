import os
import cv2
import numpy as np
import string


# Init main values
symbols = string.ascii_lowercase + "0123456789"  # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

def preprocess_data():
    n_samples = len(os.listdir('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples'))
    X = np.zeros((n_samples, 50, 200, 1))  # 1070*50*200
    y = np.zeros((5, n_samples, num_symbols))  # 5*1070*36

    for i, pic in enumerate(os.listdir('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples')):
        # Read image as grayscale
        img = cv2.imread(
            os.path.join('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples', pic),
            cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs

    # Return final data
    return X, y

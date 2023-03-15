import matplotlib as mpl
from predict import predict_captcha
import cv2
import matplotlib.pyplot as plt
mpl.use('TkAgg')


img = cv2.imread('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\c.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print("Predicted Captcha =", predict_captcha('C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\c.png'))


plt.show()


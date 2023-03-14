import os
from predict import predict_captcha
#je viens de tester
#Correct Predictions: 806
#Incorrect Predictions: 234
#77.5% correct contre 22.5% incrrect



# Set the path to the directory containing the images
image_dir = "C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples"


correct_count = 0
incorrect_count = 0

# Loop through all images in the directory
for image_file in os.listdir(image_dir):
    # Get the full path to the image file
    image_path = os.path.join(image_dir, image_file)

    # Solve the captcha using the solve function
    predicted_text = predict_captcha(image_path)

    # Get the filename without the extension
    filename = os.path.splitext(image_file)[0]

    # Check if the predicted text matches the filename
    if predicted_text == filename:
        print(f"Prediction for {image_file}: {predicted_text} (CORRECT)")
        correct_count +=1
    else:
        print(f"Prediction for {image_file}: {predicted_text} (INCORRECT)")
        incorrect_count +=1

print("Correct Predictions: {}".format(correct_count))
print("Incorrect Predictions: {}".format(incorrect_count))
print("Correct->{}".format((correct_count/1040)*100))
print("Incorrecte->{}".format((incorrect_count/1040)*100))
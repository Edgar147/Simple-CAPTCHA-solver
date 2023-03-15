import os
from predict import predict_captcha
#je viens de tester
#Correct Predictions: 806
#Incorrect Predictions: 234
#77.5% correct contre 22.5% incrrect

# Set the path to the directory containing the images
image_dir = "C:\\Users\\karap\\Desktop\\M1\\IR\\CAPTCHA-Solver-master\\Noisy Arc\\samples"

# Initialize dictionaries to keep track of correct and incorrect predictions for each character
correct_counts = {char: 0 for char in "02345678abcdefgkmnpqrstuvwxyz"}
incorrect_counts = {char: 0 for char in "02345678abcdefgkmnpqrstuvwxyz"}


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
        correct_count += 1
    else:
        print(f"Prediction for {image_file}: {predicted_text} (INCORRECT)")
        incorrect_count += 1

    # Loop through each character in the predicted text and compare it with the corresponding character in the filename
    for pred_char, true_char in zip(predicted_text, filename):
        if pred_char == true_char:
            correct_counts[pred_char] += 1
        else:
            incorrect_counts[pred_char] += 1

# Print the overall accuracy
total_count = correct_count + incorrect_count
accuracy = correct_count / total_count
print("Overall Accuracy: {:.2f}%".format(accuracy * 100))

# Print the results for each character
for char in "2345678bcdefgmnpwxy":
    total_count = correct_counts[char] + incorrect_counts[char]
    if total_count > 0:
        accuracy = correct_counts[char] / total_count
        print("{}: {} correct, {} incorrect, accuracy={:.2f}%".format(char, correct_counts[char], incorrect_counts[char], accuracy * 100))
    else:
        print("{}: No predictions for this character".format(char))

# Print the total number of correct and incorrect predictions
print("Correct Predictions: {}".format(correct_count))
print("Incorrect Predictions: {}".format(incorrect_count))
print("Correct -> {:.2f}%".format((correct_count/1040)*100))
print("Incorrect -> {:.2f}%".format((incorrect_count/1040)*100))

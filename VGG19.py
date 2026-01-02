import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------------------------------------
# Load Pretrained VGG-19 Model (ImageNet)
# -----------------------------------------------------------
vgg_model = VGG19(weights='imagenet')

# -----------------------------------------------------------
# Function: Classify Image
# -----------------------------------------------------------
def classify_image(img_path):

    # Load image (OpenCV)
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Could not load image.")
        return

    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to VGG19 input size
    resized = cv2.resize(img, (224, 224))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # Predict using VGG19
    predictions = vgg_model.predict(arr)
    decoded = decode_predictions(predictions, top=3)[0]

    # Prepare text result
    result_text = ""
    for i, (imagenet_id, label, score) in enumerate(decoded):
        result_text += f"{i+1}. {label} ({score*100:.2f}%)\n"

    # Display original image + predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(original)
    plt.title("VGG-19 Image Classification")
    plt.axis("off")
    plt.show()

    print("\nTop Predictions:")
    print(result_text)


# -----------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------
def main():
    img_path = input("Enter image filename: ")

    while True:
        print("\n******** VGG-19 FEATURE CLASSIFICATION ********")
        print("1. Classify Image")
        print("2. Exit")

        ch = int(input("Enter choice: "))

        if ch == 1:
            classify_image(img_path)
        elif ch == 2:
            print("Exiting...")
            break
        else:
            print("Invalid choice!")


main()


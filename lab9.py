# Program 9: to implement Feature Detection.
# Menu:
# 1. Harris Corner 2. MSER 3. SIFT

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Helper: Display Original + Result
# -----------------------------------------------------------
def display_two(title1, img1, title2, img2):
    plt.figure(figsize=(11, 6))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.show()


# -----------------------------------------------------------
# 1. Harris Corner Detection
# -----------------------------------------------------------
def harris_corner(img, gray):
    gray32 = np.float32(gray)

    harris = cv2.cornerHarris(gray32, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    result = img.copy()
    result[harris > 0.01 * harris.max()] = [0, 0, 255]  # red corners

    display_two("Original Image", img, "Harris Corner Detection", result)


# -----------------------------------------------------------
# 2. MSER Detection
# -----------------------------------------------------------
def mser_detection(img, gray):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    result = img.copy()

    for p in regions:
        hull = cv2.convexHull(p.reshape(-1, 1, 2))
        cv2.polylines(result, [hull], True, (0, 255, 0), 2)  # green lines

    display_two("Original Image", img, "MSER Regions", result)


# -----------------------------------------------------------
# 3. SIFT Detection
# -----------------------------------------------------------
def sift_detection(img, gray):
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    result = cv2.drawKeypoints(img, keypoints, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    display_two("Original Image", img, "SIFT Keypoints", result)


# -----------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------
def main():
    img_path = input("Enter image filename: ")
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Unable to load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        print("\n******** FEATURE DETECTION MENU ********")
        print("1. Harris Corner Detection")
        print("2. MSER Detection")
        print("3. SIFT Feature Detection")
        print("4. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            harris_corner(img, gray)

        elif choice == 2:
            mser_detection(img, gray)

        elif choice == 3:
            sift_detection(img, gray)

        elif choice == 4:
            print("Exiting Program...")
            break

        else:
            print("Invalid choice, try again.")


# Run Program
main()

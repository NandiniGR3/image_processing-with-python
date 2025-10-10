# contrast stretching of image with normalization and histogram

import cv2
import numpy as np
import matplotlib.pyplot as plt



def display_image(title, image):
    """Display an image in a window."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_histogram(image, title):
    """Plot histogram for grayscale image."""
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue')
    plt.show()

def contrast_stretching(image):
    """Apply contrast stretching transformation."""
    # Convert to float for accurate computation
    img_float = image.astype('float')
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    
    stretched = (img_float - min_val) * (255.0 / (max_val - min_val))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype('uint8')

def normalization(image):
    """Apply normalization using OpenCV."""
    normalized = np.zeros(image.shape)
    normalized = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype('uint8')

def histogram_equalization(image):
    """Apply histogram equalization (for grayscale images)."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def display_combined_results(original, stretched, normalized, equalized):
    """Show all images and histograms together."""
    titles = ['Original', 'Contrast Stretched', 'Normalized', 'Histogram Equalized']
    images = [original, stretched, normalized, equalized]
    
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(4, 2, 2*i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

        plt.subplot(4, 2, 2*i+2)
        plt.hist(images[i].ravel(), bins=256, range=[0,256], color='blue')
        plt.title(titles[i] + " Histogram")
    plt.tight_layout()
    plt.show()



def main():
    image_path = input("Enter the image file path: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image. Check the file path.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    while True:
        print("\nMenu:")
        print("1. Display Original Image")
        print("2. Apply Contrast Stretching")
        print("3. Apply Normalization")
        print("4. Apply Histogram Equalization")
        print("5. Display Combined Images with Histograms")
        print("6. Exit")

        choice = input("Choose an option: ")

        if choice == '1':
            display_image("Original Image", image)
            plot_histogram(gray, "Original Histogram")

        elif choice == '2':
            stretched = contrast_stretching(gray)
            display_image("Contrast Stretched Image", stretched)
            plot_histogram(stretched, "Contrast Stretched Histogram")

        elif choice == '3':
            normalized = normalization(gray)
            display_image("Normalized Image", normalized)
            plot_histogram(normalized, "Normalized Histogram")

        elif choice == '4':
            equalized = histogram_equalization(gray)
            display_image("Histogram Equalized Image", equalized)
            plot_histogram(equalized, "Equalized Histogram")

        elif choice == '5':
            stretched = contrast_stretching(gray)
            normalized = normalization(gray)
            equalized = histogram_equalization(gray)
            display_combined_results(image, stretched, normalized, equalized)

        elif choice == '6':
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

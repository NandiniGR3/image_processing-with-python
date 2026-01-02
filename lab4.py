import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter

# ---------------- DISPLAY FUNCTION ----------------
def show_image(title, image):
    plt.figure(figsize=(5, 5))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# ---------------- CHANNEL-WISE HANDLING ----------------
def apply_channel_wise(img, func, *args, **kwargs):
    """Apply a filter to grayscale OR each channel of a color image."""
    if len(img.shape) == 2:  # Grayscale
        return func(img, *args, **kwargs)
    else:
        b, g, r = cv2.split(img)
        fb = func(b, *args, **kwargs)
        fg = func(g, *args, **kwargs)
        fr = func(r, *args, **kwargs)
        return cv2.merge([fb, fg, fr])


# ---------------- LINEAR FILTERS ----------------
def mean_filter(img, k=3):
    return cv2.blur(img, (k, k))

def gaussian_filter(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)

def harmonic_mean(img, k=3):
    img = img.astype(float) + 1e-8
    result = np.zeros_like(img)
    m, n = img.shape
    pad = k // 2
    padded = np.pad(img, pad, mode='reflect')

    for i in range(m):
        for j in range(n):
            region = padded[i:i+k, j:j+k]
            result[i, j] = (k*k) / np.sum(1.0 / region)

    return np.uint8(result)

def geometric_mean(img, k=3):
    img = img.astype(float) + 1e-8
    result = np.zeros_like(img)
    m, n = img.shape
    pad = k // 2
    padded = np.pad(img, pad, mode='reflect')

    for i in range(m):
        for j in range(n):
            region = padded[i:i+k, j:j+k]
            result[i, j] = np.exp(np.mean(np.log(region)))

    return np.uint8(result)


# ---------------- NON-LINEAR FILTERS ----------------
def median_filter(img, k=3):
    return cv2.medianBlur(img, k)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def min_filter(img, k=3):
    return cv2.erode(img, np.ones((k, k), np.uint8))

def max_filter(img, k=3):
    return cv2.dilate(img, np.ones((k, k), np.uint8))

def percentile_filter_custom(img, percentile=50, k=3):
    return percentile_filter(img, percentile=percentile, size=k)


# ---------------- MAIN MENU ----------------
def main():
    path = input("Enter image path: ")
    img = cv2.imread(path)
    if img is None:
        print("Error loading image.")
        return

    while True:
        print("\nChoose an option:")
        print("1. Original Image")
        print("2. Linear Filters")
        print("3. Non-Linear Filters")
        print("4. Exit")

        choice = input("Enter your choice: ")

        # Show Original
        if choice == '1':
            show_image("Original Image", img)

        # LINEAR FILTERS
        elif choice == '2':
            print("\nLinear Filters:")
            print("1. Mean Filter")
            print("2. Gaussian Filter")
            print("3. Harmonic Mean Filter")
            print("4. Geometric Mean Filter")
            sub = input("Enter your choice: ")

            if sub == '1':
                show_image("Mean Filter", apply_channel_wise(img, mean_filter))
            
            elif sub == '2':
                show_image("Gaussian Filter", apply_channel_wise(img, gaussian_filter))
            
            elif sub == '3':
                show_image("Harmonic Mean", apply_channel_wise(img, harmonic_mean))
            
            elif sub == '4':
                show_image("Geometric Mean", apply_channel_wise(img, geometric_mean))


        # NON-LINEAR FILTERS
        elif choice == '3':
            print("\nNon-linear Filters:")
            print("1. Median Filter")
            print("2. Bilateral Filter")
            print("3. Min & Max Filter")
            print("4. Percentile Filter")
            sub = input("Enter your choice: ")

            if sub == '1':
                show_image("Median Filter", apply_channel_wise(img, median_filter))

            elif sub == '2':
                show_image("Bilateral Filter", apply_channel_wise(img, bilateral_filter))

            elif sub == '3':
                print("a. Min Filter")
                print("b. Max Filter")
                mm = input("Choose a/b: ")
                if mm.lower() == 'a':
                    show_image("Min Filter", apply_channel_wise(img, min_filter))
                else:
                    show_image("Max Filter", apply_channel_wise(img, max_filter))

            elif sub == '4':
                p = int(input("Enter percentile (0-100): "))
                show_image("Percentile Filter", apply_channel_wise(img, percentile_filter_custom, percentile=p))

        elif choice == '4':
            print("Exiting program…")
            break

        else:
            print("Invalid input! Try again.")


if __name__ == "__main__":
    main()

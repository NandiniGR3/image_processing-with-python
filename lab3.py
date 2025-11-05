import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    plt.figure(figsize=(5,5))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. IMAGE TRANSFORMATIONS
def negative_image(img):
    return 255 - img

def log_transformation(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    return np.array(log_img, dtype=np.uint8)

def power_law_transformation(img, gamma=1.2):
    c = 255 / (np.power(np.max(img), gamma))
    power_img = c * np.power(img, gamma)
    return np.array(power_img, dtype=np.uint8)

# 2. PIECEWISE LINEAR TRANSFORMATIONS
def binary_threshold(img, thresh=128):
    _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary_img

def intensity_level_slicing(img, lower=100, upper=200):
    sliced = np.where((img >= lower) & (img <= upper), 255, 0)
    return np.array(sliced, dtype=np.uint8)

# 3. LINEAR FILTERS
def mean_filter(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))

def gaussian_filter(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def harmonic_mean_filter(img, ksize=3):
    img = img.astype(np.float64) + 1e-8
    kernel_size = ksize * ksize
    result = np.zeros_like(img)
    for i in range(img.shape[0]-ksize+1):
        for j in range(img.shape[1]-ksize+1):
            region = img[i:i+ksize, j:j+ksize]
            result[i+ksize//2, j+ksize//2] = kernel_size / np.sum(1.0 / region)
    return np.array(result, dtype=np.uint8)

def geometric_mean_filter(img, ksize=3):
    img = img.astype(np.float64) + 1e-8
    result = np.zeros_like(img)
    kernel_size = ksize * ksize
    for i in range(img.shape[0]-ksize+1):
        for j in range(img.shape[1]-ksize+1):
            region = img[i:i+ksize, j:j+ksize]
            gm = np.prod(region)**(1.0/kernel_size)
            result[i+ksize//2, j+ksize//2] = gm
    return np.array(result, dtype=np.uint8)

# 4. FREQUENCY DOMAIN FILTERS
def frequency_filter(img, filter_type, cutoff=30, band=(10, 60)):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)

    if filter_type == 'low':
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    elif filter_type == 'high':
        mask[:, :] = 1
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
    elif filter_type == 'band':
        mask[crow - band[1]:crow + band[1], ccol - band[1]:ccol + band[1]] = 1
        mask[crow - band[0]:crow + band[0], ccol - band[0]:ccol + band[0]] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    return img_back

# MAIN MENU
def main():
    path = input("Enter image path: ")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Original Image")
        print("2. Spatial Domain Filters")
        print("3. Frequency Domain Filters")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            show_image("Original Image", img)

        elif choice == '2':
            print("\n-- SPATIAL DOMAIN FILTERS --")
            print("1. Image Transformations")
            print("2. Piecewise Linear Transformations")
            print("3. Linear Filters")
            print("4. Exit")
            sub = input("Enter your choice: ")

            if sub == '1':
                print("\n-- Image Transformations --")
                print("1. Negative\n2. Log Transformation\n3. Power Law Transformation")
                opt = input("Enter your choice: ")
                if opt == '1': show_image("Negative Image", negative_image(img))
                elif opt == '2': show_image("Log Transformation", log_transformation(img))
                elif opt == '3': show_image("Power Law Transformation", power_law_transformation(img))

            elif sub == '2':
                print("\n-- Piecewise Linear Transformations --")
                print("1. Binary Thresholding\n2. Intensity Level Slicing")
                opt = input("Enter your choice: ")
                if opt == '1': show_image("Binary Threshold", binary_threshold(img))
                elif opt == '2': show_image("Intensity Level Slicing", intensity_level_slicing(img))

            elif sub == '3':
                print("\n-- Linear Filters --")
                print("1. Mean Filter\n2. Gaussian Filter\n3. Harmonic Mean\n4. Geometric Mean")
                opt = input("Enter your choice: ")
                if opt == '1': show_image("Mean Filter", mean_filter(img))
                elif opt == '2': show_image("Gaussian Filter", gaussian_filter(img))
                elif opt == '3': show_image("Harmonic Mean Filter", harmonic_mean_filter(img))
                elif opt == '4': show_image("Geometric Mean Filter", geometric_mean_filter(img))

        elif choice == '3':
            print("\n-- FREQUENCY DOMAIN FILTERS --")
            print("1. Low-pass Filter\n2. High-pass Filter\n3. Band-pass Filter\n4. Exit")
            opt = input("Enter your choice: ")
            if opt == '1': show_image("Low-pass Filter", frequency_filter(img, 'low'))
            elif opt == '2': show_image("High-pass Filter", frequency_filter(img, 'high'))
            elif opt == '3': show_image("Band-pass Filter", frequency_filter(img, 'band'))

        elif choice == '4':
            print("Exiting program...")
            break

        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()

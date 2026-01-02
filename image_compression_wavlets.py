import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Function: Wavelet Compression for Grayscale
# -------------------------------------------------------------
def compress_grayscale(img, threshold):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply thresholding (compression step)
    LH = pywt.threshold(LH, threshold, mode='soft')
    HL = pywt.threshold(HL, threshold, mode='soft')
    HH = pywt.threshold(HH, threshold, mode='soft')

    # Reconstruct image
    compressed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    return compressed


# -------------------------------------------------------------
# Function: Wavelet Compression for Color
# -------------------------------------------------------------
def compress_color(img, threshold):
    b, g, r = cv2.split(img)

    def compress_channel(channel):
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        # Apply threshold to details
        LH = pywt.threshold(LH, threshold, mode='soft')
        HL = pywt.threshold(HL, threshold, mode='soft')
        HH = pywt.threshold(HH, threshold, mode='soft')

        return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    # Compress each channel
    cb = compress_channel(b)
    cg = compress_channel(g)
    cr = compress_channel(r)

    # Merge channels
    compressed = cv2.merge((
        np.clip(cb, 0, 255).astype(np.uint8),
        np.clip(cg, 0, 255).astype(np.uint8),
        np.clip(cr, 0, 255).astype(np.uint8)
    ))

    return compressed


# -------------------------------------------------------------
# Display Original & Compressed Image
# -------------------------------------------------------------
def display_results(original, compressed, is_color):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    if is_color:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Compressed Image (Wavelet)")
    if is_color:
        plt.imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(compressed, cmap='gray')
    plt.axis('off')

    plt.show()


# -------------------------------------------------------------
# MAIN PROGRAM MENU
# -------------------------------------------------------------
def main():
    image_path = input("Enter image filename: ")
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image!")
        return

    is_color = True if len(img.shape) == 3 else False

    while True:
        print("\n******** IMAGE COMPRESSION USING WAVELETS ********")
        print("1. Compress Image")
        print("2. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            threshold = float(input("Enter threshold value (10-50 recommended): "))

            if is_color:
                compressed = compress_color(img, threshold)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                compressed = compress_grayscale(gray, threshold)

            # Show original + compressed
            display_results(img if is_color else gray, compressed, is_color)

            print("Compression Completed Successfully.")

        elif choice == 2:
            print("Exiting Program...")
            break

        else:
            print("Invalid choice. Try again.")


# Run program
main()


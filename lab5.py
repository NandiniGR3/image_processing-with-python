# to implement homomorphic filtering
# Menu:
# 1. Grayscale Image 2. Color Image


import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to display images
def show(title, img):
    plt.figure(figsize=(5, 5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def homomorphic_filter(img, low_gain=0.5, high_gain=2.0, cutoff=30):
    """
    img: input grayscale image
    """
    img = img.astype(np.float32) / 255.0

    # Step 1: Log Transformation
    log_img = np.log1p(img)

    # Step 2: FFT
    dft = np.fft.fft2(log_img)
    dft_shift = np.fft.fftshift(dft)

    # Step 3: Construct Homomorphic Filter (Butterworth)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    n = 2  # filter order

    H = (high_gain - low_gain) * (1 - np.exp(-(D**2) / (2 * (cutoff**2)))) + low_gain

    # Step 4: Apply filter
    filtered = dft_shift * H

    # Step 5: Inverse FFT
    ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(ishift)
    img_back = np.real(img_back)

    # Step 6: Inverse log
    img_exp = np.expm1(img_back)

    # Step 7: Normalize
    img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)

    return img_norm.astype(np.uint8)


def main():

    print("\n=== HOMOMORPHIC FILTER IMPLEMENTATION ===")
    path = input("Enter image path: ")

    img = cv2.imread(path)
    if img is None:
        print("⚠ Error: Image not found!")
        return

    while True:
        print("\n-------------------------------")
        print("Menu:")
        print("1. Grayscale Image")
        print("2. Color Image")
        print("3. Exit")
       

        ch = input("Enter your choice: ")

        # ----------- GRAYSCALE MODE -----------
        if ch == "1":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = homomorphic_filter(gray)
            show("Original Grayscale", gray)
            show("Homomorphic Filter Output", out)

        # ----------- COLOR MODE -----------
        elif ch == "2":
            b, g, r = cv2.split(img)

            hb = homomorphic_filter(b)
            hg = homomorphic_filter(g)
            hr = homomorphic_filter(r)

            color_out = cv2.merge([hb, hg, hr])

            show("Original Color Image", img)
            show("Homomorphic Filter Output (Color)", color_out)

        # ----------- EXIT -----------
        elif ch == "3":
            print("Exiting...")
            break

        else:
            print(" Invalid choice! Try again.")


# Run the program
if __name__ == "__main__":
    main()

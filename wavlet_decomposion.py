import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Function: Decompose Image (Displays original + subbands)
# -----------------------------------------------------------
def decompose_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
        return None

    is_color = False if len(img.shape) == 2 else True

    # -----------------------------------------------------------
    # GRAYSCALE IMAGE PROCESSING
    # -----------------------------------------------------------
    if not is_color:
        coeffs = pywt.dwt2(img, 'haar')
        LL, (LH, HL, HH) = coeffs

        plt.figure(figsize=(12, 8))
        plt.suptitle("Wavelet Decomposition (Grayscale)", fontsize=14)

        # Original
        plt.subplot(2, 3, 1)
        plt.title("Original")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        # LL
        plt.subplot(2, 3, 2)
        plt.title("LL (Approximation)")
        plt.imshow(LL, cmap='gray')
        plt.axis('off')

        # LH
        plt.subplot(2, 3, 3)
        plt.title("LH (Horizontal Detail)")
        plt.imshow(LH, cmap='gray')
        plt.axis('off')

        # HL
        plt.subplot(2, 3, 4)
        plt.title("HL (Vertical Detail)")
        plt.imshow(HL, cmap='gray')
        plt.axis('off')

        # HH
        plt.subplot(2, 3, 5)
        plt.title("HH (Diagonal Detail)")
        plt.imshow(HH, cmap='gray')
        plt.axis('off')

        plt.show()

        print("Grayscale image decomposed successfully.")
        return (coeffs, "gray")

    # -----------------------------------------------------------
    # COLOR IMAGE PROCESSING
    # -----------------------------------------------------------
    b, g, r = cv2.split(img)

    coeffs_b = pywt.dwt2(b, 'haar')
    coeffs_g = pywt.dwt2(g, 'haar')
    coeffs_r = pywt.dwt2(r, 'haar')

    LLb, (LHb, HLb, HHb) = coeffs_b
    LLg, (LHg, HLg, HHg) = coeffs_g
    LLr, (LHr, HLr, HHr) = coeffs_r

    # Merge LL (approximation)
    LL_color = cv2.merge((
        LLb.astype(np.uint8),
        LLg.astype(np.uint8),
        LLr.astype(np.uint8)
    ))

    plt.figure(figsize=(12, 8))
    plt.suptitle("Wavelet Decomposition (Color)", fontsize=14)

    # Original
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # LL
    plt.subplot(2, 3, 2)
    plt.title("LL (Approximation)")
    plt.imshow(cv2.cvtColor(LL_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # LH (Blue)
    plt.subplot(2, 3, 3)
    plt.title("LH (Horizontal detail)")
    plt.imshow(LHb, cmap='gray')
    plt.axis('off')

    # HL (Green)
    plt.subplot(2, 3, 4)
    plt.title("HL (vertical detail)")
    plt.imshow(HLg, cmap='gray')
    plt.axis('off')

    # HH (Red)
    plt.subplot(2, 3, 5)
    plt.title("HH (diagonal detail)")
    plt.imshow(HHr, cmap='gray')
    plt.axis('off')

    plt.show()

    print("Color image decomposed successfully.")
    return (coeffs_b, coeffs_g, coeffs_r, "color")


# -----------------------------------------------------------
# RECONSTRUCT IMAGE
# -----------------------------------------------------------
def reconstruct_image(saved_coeffs):
    if saved_coeffs is None:
        print("Error: No decomposition found! Decompose first.")
        return

    if saved_coeffs[-1] == "gray":
        coeffs = saved_coeffs[0]
        reconstructed = pywt.idwt2(coeffs, 'haar')
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        plt.imshow(reconstructed, cmap='gray')
        plt.title("Reconstructed Image (Grayscale)")
        plt.axis('off')
        plt.show()

    else:
        coeffs_b, coeffs_g, coeffs_r, _ = saved_coeffs

        rec_b = pywt.idwt2(coeffs_b, 'haar')
        rec_g = pywt.idwt2(coeffs_g, 'haar')
        rec_r = pywt.idwt2(coeffs_r, 'haar')

        reconstructed = cv2.merge((
            np.clip(rec_b, 0, 255).astype(np.uint8),
            np.clip(rec_g, 0, 255).astype(np.uint8),
            np.clip(rec_r, 0, 255).astype(np.uint8)
        ))

        plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Image (Color)")
        plt.axis('off')
        plt.show()

    print("Image successfully reconstructed.")


# -----------------------------------------------------------
# MAIN MENU
# -----------------------------------------------------------
image_path = input("Enter image filename (with extension): ")

saved_coeffs = None

while True:
    print("\n******** MULTI-RESOLUTION IMAGE MENU ********")
    print("1. Decompose Image (Show original + subbands)")
    print("2. Reconstruct Image")
    print("3. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        saved_coeffs = decompose_image(image_path)

    elif choice == 2:
        reconstruct_image(saved_coeffs)

    elif choice == 3:
        print("Exiting...")
        break

    else:
        print("Invalid choice! Try again.")

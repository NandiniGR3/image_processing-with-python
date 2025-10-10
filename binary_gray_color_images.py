# binary image,grayscale,color images along with resize(upscale,downscale,horizontal scaling,vertical scaling),rotate(all type of rotates) for all


import cv2
import os

def validate_image(image_path):
    """Check if the image exists and is valid."""
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return False
    if os.path.getsize(image_path) == 0:
        print(f"Error: '{image_path}' is empty.")
        return False
    return True


def resize_image(img, option):
    """Perform resize based on user option."""
    h, w = img.shape[:2]

    if option == '1':  # Upscale
        scale_percent = int(input("Enter upscale percentage (e.g., 150 for 1.5x): "))
        new_dim = (int(w * scale_percent / 100), int(h * scale_percent / 100))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Upscaled Image", resized)

    elif option == '2':  # Downscale
        scale_percent = int(input("Enter downscale percentage (e.g., 50 for half size): "))
        new_dim = (int(w * scale_percent / 100), int(h * scale_percent / 100))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Downscaled Image", resized)

    elif option == '3':  # Vertical Scaling
        new_height = int(input("Enter new height (in pixels): "))
        resized = cv2.resize(img, (w, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Vertical Scaled Image", resized)

    elif option == '4':  # Horizontal Scaling
        new_width = int(input("Enter new width (in pixels): "))
        resized = cv2.resize(img, (new_width, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("Horizontal Scaled Image", resized)

    elif option == '5':  # Exit
        return

    else:
        print("Invalid option.")
        return

    print("Resizing completed. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(img):
    """Rotate image based on user-entered angle."""
    while True:
        angle = input("Enter rotation angle (or type 'exit' to stop): ")
        if angle.lower() == 'exit':
            break

        try:
            angle = float(angle)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rot_matrix, (w, h))
            cv2.imshow(f"Rotated {angle}° Image", rotated)
            print(f"Rotated by {angle} degrees. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ValueError:
            print("Please enter a valid numeric angle.")


def process_image(img, img_type):
    """Menu for Resize and Rotate options for a given image type."""
    while True:
        print(f"\n--- {img_type} Image Menu ---")
        print("1. Show Original Image")
        print("2. Resize")
        print("   1: Upscale  2: Downscale  3: Vertical Scaling  4: Horizontal Scaling  5: Exit")
        print("3. Rotate (enter custom angle)")
        print("4. Back to Image Selection")

        choice = input("Enter your choice: ")

        if choice == '1':
            cv2.imshow(f"Original {img_type} Image", img)
            print("Displaying original image. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif choice == '2':
            print("\nResize Options:")
            print("1. Upscale\n2. Downscale\n3. Vertical Scaling\n4. Horizontal Scaling\n5. Exit")
            sub_choice = input("Enter your resize choice: ")
            resize_image(img, sub_choice)

        elif choice == '3':
            rotate_image(img)

        elif choice == '4':
            break

        else:
            print("Invalid option. Try again.")


# ---------- Main Function ----------
def main():
    while True:
        print("\n=== Image Processing Menu ===")
        print("1. Binary Image")
        print("2. Grayscale Image")
        print("3. Color Image")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            img_path = input("Enter the path of the Binary image: ")
            if validate_image(img_path):
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                process_image(binary_img, "Binary")

        elif choice == '2':
            img_path = input("Enter the path of the Grayscale image: ")
            if validate_image(img_path):
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                process_image(gray_img, "Grayscale")

        elif choice == '3':
            img_path = input("Enter the path of the Color image: ")
            if validate_image(img_path):
                color_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                process_image(color_img, "Color")

        elif choice == '4':
            print("Exiting program.")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()

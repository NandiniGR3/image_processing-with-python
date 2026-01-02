# Program 6: to extract frames from videos analyzing the frames.
# Menu:
# 1. Play Video
# 1. Color
# 2. GrayScale
# 2. Extract Frames from Video
# 3. Slow Down Video
# 4. Speed Up Video
# 5. Reverse Video
# 6. Loop Video
# 7. Display Video Info
# 8. Exit

import cv2
import os

# --------------------------------------------------------
# Function: Play Video (Color or Grayscale)
# --------------------------------------------------------
def play_video(path, mode="color"):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode.lower() == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------
# Function: Extract Frames
# --------------------------------------------------------
def extract_frames(path, output_folder="frames"):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(filename, frame)
        count += 1

    print(f"{count} frames extracted to '{output_folder}' folder.")
    cap.release()


# --------------------------------------------------------
# Function: Slow Down / Speed Up
# --------------------------------------------------------
def play_modified_speed(path, speed_factor=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    delay = int(25 * speed_factor)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Modified Speed Video", frame)
        if cv2.waitKey(delay) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------
# Function: Reverse Video
# --------------------------------------------------------
def reverse_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    for frame in reversed(frames):
        cv2.imshow("Reversed Video", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# --------------------------------------------------------
# Function: Loop Video
# --------------------------------------------------------
def loop_video(path, loops=2):
    for i in range(loops):
        print(f"Loop {i+1}")
        play_video(path)


# --------------------------------------------------------
# Function: Display Video Info
# --------------------------------------------------------
def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0

    print("\n--- VIDEO INFORMATION ---")
    print(f"Resolution     : {int(width)} x {int(height)}")
    print(f"FPS            : {int(fps)}")
    print(f"Total Frames   : {int(frames)}")
    print(f"Duration (sec) : {duration:.2f}")
    print("--------------------------\n")

    cap.release()


# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------

# Step 1: Ask user for video input
while True:
    video_path = input("Enter the path of your video file: ")

    if os.path.exists(video_path):
        print("Video file found. Loading menu...")
        break
    else:
        print("File not found! Try again.\n")

# Step 2: Menu
while True:
    print("\n***** VIDEO PROCESSING MENU *****")
    print("1. Play Video (Color)")
    print("2. Play Video (Grayscale)")
    print("3. Extract Frames from Video")
    print("4. Slow Down Video")
    print("5. Speed Up Video")
    print("6. Reverse Video")
    print("7. Loop Video")
    print("8. Display Video Info")
    print("9. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        play_video(video_path, "color")

    elif choice == 2:
        play_video(video_path, "gray")

    elif choice == 3:
        extract_frames(video_path)

    elif choice == 4:
        play_modified_speed(video_path, speed_factor=2)   # slower

    elif choice == 5:
        play_modified_speed(video_path, speed_factor=0.5) # faster

    elif choice == 6:
        reverse_video(video_path)

    elif choice == 7:
        loops = int(input("Enter number of loops: "))
        loop_video(video_path, loops)

    elif choice == 8:
        video_info(video_path)

    elif choice == 9:
        print("Exiting program...")
        break

    else:
        print("Invalid choice. Try again.")

import cv2
from tkinter import Tk, Button, filedialog, Label, Canvas, Toplevel
from ultralytics import YOLO
from PIL import Image, ImageTk
import random

# Load the YOLO model
model = YOLO("yolo11x.pt")

# Class-color mapping
class_colors = {}

def get_class_color(class_name):
    """Assign a unique color to each class name."""
    if class_name not in class_colors:
        class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_colors[class_name]

def resize_image(image, max_width, max_height):
    """Resize the image to fit within max_width and max_height, maintaining aspect ratio."""
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width > max_width or original_height > max_height:
        if aspect_ratio > 1:  # Wider than tall
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:  # Taller than wide
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
    else:
        new_width, new_height = original_width, original_height

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def launch_camera():
    """Launch the webcam for live detection."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Launching Camera. Press 'P' to exit.")

    # Create the window
    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Get screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to the screen size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    # While loop for video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Resize the frame to fit the window size
        frame_resized = cv2.resize(frame, (screen_width, screen_height))

        # Run inference on the frame
        results = model(frame_resized)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                color = get_class_color(class_name)

                # Draw thicker bounding box and bolder text
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # Thickness = 4
                cv2.putText(frame_resized, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)  # Bigger and bolder text

        # Display the video feed
        cv2.imshow(window_name, frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    """Upload an image and perform object detection."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Read and process the image
    image = cv2.imread(file_path)
    results = model(image)

    # Draw results on the image
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            color = get_class_color(class_name)

            # Draw thicker bounding box and bolder text
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # Thickness = 4
            cv2.putText(image, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)  # Bigger and bolder text

    # Convert image for Tkinter display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Resize image to fit within the current window size (updated dynamically)
    current_width = root.winfo_width()
    current_height = root.winfo_height()

    image_resized = resize_image(image_pil, current_width, current_height)

    image_tk = ImageTk.PhotoImage(image_resized)

    # Display the processed image in a new window
    top = Toplevel(root)
    top.title("Detection Result")
    top.geometry(f"{image_tk.width()}x{image_tk.height()}+0+0")  # Maximize window size
    canvas = Canvas(top, width=image_tk.width(), height=image_tk.height())
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk

# Create the main UI
root = Tk()
root.title("Object Detection App")
root.geometry("400x300")  # Initial window size
root.configure(bg="#282c34")

# Add a title label
title_label = Label(root, text="Object Detection App", font=("Helvetica", 18, "bold"), bg="#282c34", fg="white")
title_label.pack(pady=20)

# Add buttons with improved styling
launch_button = Button(root, text="Launch Camera", command=launch_camera, width=20, height=2, bg="#61afef", fg="white", font=("Helvetica", 12))
launch_button.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_image, width=20, height=2, bg="#98c379", fg="white", font=("Helvetica", 12))
upload_button.pack(pady=10)

exit_button = Button(root, text="Exit", command=root.quit, width=20, height=2, bg="#e06c75", fg="white", font=("Helvetica", 12))
exit_button.pack(pady=10)

# Allow the window to be resizable
root.resizable(True, True)

# Run the UI
root.mainloop()

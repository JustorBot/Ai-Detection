import cv2
from tkinter import Tk, Button, filedialog, Label, Canvas, Toplevel, Scale
from tkinter.colorchooser import askcolor
from ultralytics import YOLO
from PIL import Image, ImageTk
import random
import time

# Load the YOLO model
model = YOLO("yolo11x.pt")

# Class-color mapping
class_colors = {}

# Global variable to store the selected person bounding box color
person_color = (0, 255, 0)  # Default color (green)

def get_class_color(class_name):
    """Assign a unique color to each class name or use the user-defined color for 'person'."""
    if class_name == 'person':
        return person_color
    if class_name not in class_colors:
        class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_colors[class_name]

def resize_image(image, max_width=800, max_height=800):
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

    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    # Set the interval for updating FPS (in seconds)
    fps_update_interval = 1  # Update FPS every 1 second

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

        # Get the confidence threshold value
        confidence_threshold = threshold_slider.get() / 100  # Convert to percentage

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]  # Get confidence of the detection
                if confidence >= confidence_threshold:  # Only display if above threshold
                    x1, y1, x2, y2 = box.xyxy[0]
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    color = get_class_color(class_name)

                    # Draw thicker bounding box and bolder text
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # Thickness = 4
                    cv2.putText(frame_resized, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)  # Bigger and bolder text

        # Increment the frame count
        frame_count += 1
        elapsed_time = time.time() - start_time

        # Update FPS every fps_update_interval seconds
        if elapsed_time >= fps_update_interval:
            fps = frame_count / elapsed_time  # Calculate FPS
            frame_count = 0  # Reset the frame count
            start_time = time.time()  # Reset the start time

            # Display FPS on the frame in red color (BGR color code for red is (0, 0, 255))
            cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

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
    original_image = cv2.imread(file_path)
    results = model(original_image)

    # Get the confidence threshold value
    confidence_threshold = threshold_slider.get() / 100  # Convert to percentage

    # Draw results on the image
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]  # Get confidence of the detection
            if confidence >= confidence_threshold:  # Only display if above threshold
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                color = get_class_color(class_name)

                # Draw thicker bounding box and bolder text
                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # Thickness = 4
                cv2.putText(original_image, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)  # Bigger and bolder text

    # Convert image for Tkinter display
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_pil = Image.fromarray(original_image_rgb)

    # Create the display window
    top = Toplevel(root)
    top.title("Detection Result")

    # Create a canvas
    canvas = Canvas(top)
    canvas.pack(fill="both", expand=True)

    def resize_event(event):
        """Handle resizing of the Toplevel window and update the image display."""
        new_width = min(event.width, 800)
        new_height = min(event.height, 800)
        resized_image = resize_image(original_image_pil, new_width, new_height)
        image_tk = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
        canvas.image = image_tk

    # Bind the resize event
    top.bind("<Configure>", resize_event)

    # Set initial size
    top.geometry("800x800")

def pick_person_color():
    """Allow the user to pick a color for the bounding boxes of 'person' class."""
    global person_color
    color = askcolor()[0]  # Ask for color and get the RGB tuple
    if color:
        person_color = tuple(map(int, color))  # Convert the color to a tuple of integers

# Create the main UI
root = Tk()
root.title("Object Detection App")
root.geometry("400x500")  # Initial window size
root.configure(bg="#282c34")

# Add a title label
title_label = Label(root, text="Object Detection App", font=("Helvetica", 18, "bold"), bg="#282c34", fg="white")
title_label.pack(pady=20)

# Add buttons with improved styling
launch_button = Button(root, text="Launch Camera", command=launch_camera, width=20, height=2, bg="#61afef", fg="white", font=("Helvetica", 12))
launch_button.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_image, width=20, height=2, bg="#98c379", fg="white", font=("Helvetica", 12))
upload_button.pack(pady=10)

# Add a button to pick color for 'person' bounding boxes
pick_color_button = Button(root, text="Pick Person Color", command=pick_person_color, width=20, height=2, bg="#e0e0e0", fg="black", font=("Helvetica", 12))
pick_color_button.pack(pady=10)

# Add confidence threshold slider
threshold_label = Label(root, text="Confidence Threshold:", font=("Helvetica", 12), bg="#282c34", fg="white")
threshold_label.pack(pady=10)

threshold_slider = Scale(root, from_=0, to=100, orient="horizontal", length=300, tickinterval=10, bg="#282c34", fg="white", sliderlength=20)
threshold_slider.set(50)  # Default value of 50%
threshold_slider.pack(pady=10)

exit_button = Button(root, text="Exit", command=root.quit, width=20, height=2, bg="#e06c75", fg="white", font=("Helvetica", 12))
exit_button.pack(pady=10)

# Allow the window to be resizable
root.resizable(True, True)

# Run the UI
root.mainloop()

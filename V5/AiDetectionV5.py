import cv2
from tkinter import Tk, Button, filedialog, Label, Canvas, Toplevel, Scale
from tkinter.colorchooser import askcolor
from ultralytics import YOLO
from PIL import Image, ImageTk
import random
import time

# Load the YOLO model
model = YOLO("yolov8s-worldv2.pt")

# Class-color mapping
class_colors = {}

# Global variable to store the selected person bounding box color
person_color = (0, 255, 0)  # Default color (green)

# Global variable to store the selected person bounding box color
def get_class_color(class_name):
    """Assign a unique color to each class name or use the user-defined color for 'person'."""
    global person_color  # Make sure person_color is globally accessible
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

def show_loading_indicator():
    """Show a loading indicator during the image processing."""
    loading_window = Toplevel(root)
    loading_window.title("Processing...")
    loading_window.geometry("300x150")
    loading_window.configure(bg="#282c34")

    label = Label(loading_window, text="Processing image... Please wait.", font=("Helvetica", 12), bg="#282c34", fg="white")
    label.pack(pady=40)

    loading_window.update_idletasks()
    return loading_window

def hide_loading_indicator(loading_window):
    """Hide the loading indicator after processing is complete."""
    loading_window.destroy()

def launch_camera():
    """Launch the webcam for live detection."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Launching Camera. Press 'P' to exit. Press 'R' to pause/resume. Press 'T' to toggle overlays.")

    # Create the window
    window_name = "Live Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Get screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to fit the screen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    detection_count = 0
    detection_summary = {}  # Dictionary to store class-wise count of detected objects
    paused = False
    show_overlays = True  # Flag to toggle overlays

    while True:
        if not paused:
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

            # Reset detection summary for the frame
            detection_summary.clear()

            # Extract inference time
            inference_time = results[0].speed["inference"]  # Inference time in milliseconds
            fps = 1000 / inference_time if inference_time > 0 else 0

            if show_overlays:  # Only process overlays if enabled
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

                            # Increment class detection count
                            detection_summary[class_name] = detection_summary.get(class_name, 0) + 1

                            # Draw bounding box and class label
                            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)  # Thickness = 4
                            cv2.putText(frame_resized, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)  # Bigger and bolder text
                            detection_count += 1

                # Display FPS on the frame
                cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

                # Display real-time detection count
                cv2.putText(frame_resized, f"Detections: {detection_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                # Display class summary
                y_offset = 150
                for class_name, count in detection_summary.items():
                    cv2.putText(frame_resized, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 50

            # Display the video feed
            cv2.imshow(window_name, frame_resized)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):  # Exit the application
            break
        elif key == ord('r'):  # Pause/Resume the video feed
            paused = not paused
        elif key == ord('t'):  # Toggle display of bounding boxes and labels
            show_overlays = not show_overlays  # Toggle the flag

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Display stats in a popup
    stats_window = Toplevel(root)
    stats_window.title("Session Statistics")
    stats_window.geometry("400x300")
    stats_window.configure(bg="#282c34")

    Label(stats_window, text="Session Statistics", font=("Helvetica", 16, "bold"), bg="#282c34", fg="white").pack(pady=10)
    Label(stats_window, text=f"Total Detections: {detection_count}", font=("Helvetica", 12), bg="#282c34", fg="white").pack(pady=5)
    Label(stats_window, text=f"Average FPS: {fps:.2f}", font=("Helvetica", 12), bg="#282c34", fg="white").pack(pady=5)

    # Display per-class detection summary
    if detection_summary:
        for class_name, count in detection_summary.items():
            Label(stats_window, text=f"{class_name}: {count}", font=("Helvetica", 12), bg="#282c34", fg="white").pack(pady=2)

    Button(stats_window, text="Close", command=stats_window.destroy, bg="#e06c75", fg="white", font=("Helvetica", 12)).pack(pady=10)

def upload_image():
    """Upload an image and perform object detection."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Show loading indicator
    loading_window = show_loading_indicator()

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

    # Save function
    def save_image():
        """Allow the user to save the processed image."""
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")])
        if save_path:
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(save_path, original_image_bgr)
            print(f"Image saved to {save_path}")

    # Create Save button
    save_button = Button(top, text="Save Image", command=save_image, bg="#61afef", fg="white", font=("Helvetica", 12))
    save_button.pack(pady=10)

    # Bind the resize event
    top.bind("<Configure>", resize_event)

    # Set initial size
    top.geometry("800x800")

    # Hide loading indicator after processing is complete
    hide_loading_indicator(loading_window)

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

# Function to update the status bar
def update_status(message):
    """Update the status bar with a given message."""
    status_bar.config(text=message)
    status_bar.update_idletasks()  # Ensure the update is reflected immediately

# Add a title label
title_label = Label(root, text="Object Detection App", font=("Helvetica", 18, "bold"), bg="#282c34", fg="white")
title_label.pack(pady=20)

# Add buttons with improved styling
launch_button = Button(root, text="Launch Camera", command=lambda: [update_status("Launching Camera..."), launch_camera()], width=20, height=2, bg="#61afef", fg="white", font=("Helvetica", 12))
launch_button.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=lambda: [update_status("Uploading Image..."), upload_image()], width=20, height=2, bg="#98c379", fg="white", font=("Helvetica", 12))
upload_button.pack(pady=10)

# Add a button to pick color for 'person' bounding boxes
pick_color_button = Button(root, text="Pick Person Color", command=lambda: [update_status("Picking Color..."), pick_person_color()], width=20, height=2, bg="#e0e0e0", fg="black", font=("Helvetica", 12))
pick_color_button.pack(pady=10)

# Add confidence threshold slider
threshold_label = Label(root, text="Confidence Threshold:", font=("Helvetica", 12), bg="#282c34", fg="white")
threshold_label.pack(pady=10)

threshold_slider = Scale(root, from_=0, to=100, orient="horizontal", length=300, tickinterval=10, bg="#282c34", fg="white", sliderlength=20)
threshold_slider.set(50)  # Default value of 50%
threshold_slider.pack(pady=10)

exit_button = Button(root, text="Exit", command=lambda: [update_status("Exiting..."), root.quit()], width=20, height=2, bg="#e06c75", fg="white", font=("Helvetica", 12))
exit_button.pack(pady=10)

# Add a status bar at the bottom
status_bar = Label(root, text="Ready", font=("Helvetica", 12), bg="#1e1e1e", fg="white", anchor="w")
status_bar.pack(side="bottom", fill="x")

# Allow the window to be resizable
root.resizable(True, True)

# Run the UI
root.mainloop()

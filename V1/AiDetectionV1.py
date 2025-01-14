import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Run inference on the current frame
    results = model(frame)  # Perform inference on the frame
    
    # Process the results
    for result in results:
        # Extract bounding boxes and other outputs
        boxes = result.boxes  # Bounding boxes object
        for box in boxes:
            # Get the box coordinates (x1, y1, x2, y2) and class ID
            x1, y1, x2, y2 = box.xyxy[0]  # Extract the coordinates of the bounding box
            class_id = int(box.cls[0])  # Get the class ID (integer)
            
            # Get the class name using the class ID (from the YOLO model)
            class_name = result.names[class_id]
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw rectangle
            
            # Add the class label text above the bounding box
            cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # If segmentation masks or keypoints exist, they can also be drawn here
    
    # Display the frame with bounding boxes (or other results)
    cv2.imshow('Camera Feed', frame)
    
    # Wait for key press to exit (if 'p' is pressed, close the feed)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

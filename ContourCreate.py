import cv2
import numpy as np

# Load the image
image = cv2.imread('images/President_Barack_Obama1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use a face detector (Haar Cascade for this example)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop through detected faces
for (x, y, w, h) in faces:
    # Extract the face region from the image
    face_roi = image[y:y+h, x:x+w]
    
    # Convert the face region to grayscale
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(face_gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image (for visualization)
    cv2.drawContours(face_roi, contours, -1, (0, 255, 0), 2)
    
    # Optional: Split the face into sections based on contours
    # Here we assume simple sections like top, middle, bottom thirds of the face
    height, width = face_roi.shape[:2]
    section_height = height // 3
    
    sections = [
        face_roi[0:section_height, :],        # Top third
        face_roi[section_height:2*section_height, :],  # Middle third
        face_roi[2*section_height:height, :]  # Bottom third
    ]
    
    # Display each section
    for i, section in enumerate(sections):
        cv2.imshow(f'Section {i+1}', section)

    # Save each section (optional)
    for i, section in enumerate(sections):
        cv2.imwrite(f'section_{i+1}.jpg', section)

# Display the result
cv2.imshow('Detected Face Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

# Load the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hand_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # Replace with the actual path

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangles around faces and display the text
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the text "Face Detected" near the top-left corner of the face rectangle
        cv2.putText(img, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Detect hands
    hands = hand_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangles around hands and display the text
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the text "Hand Detected" near the top-left corner of the hand rectangle
        cv2.putText(img, 'eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()

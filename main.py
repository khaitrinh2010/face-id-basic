import cv2

# Load the cascade for eye detection
detector = cv2.CascadeClassifier("models\\haarcascade_eye.xml")

# Start video capture from the first camera device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if no frame is captured

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the image
    eyes = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Scale factor between the window searches
        minNeighbors=20,  # How many neighbors each candidate rectangle should have
        minSize=(30, 30),  # Minimum object size to detect
    )

    # Draw rectangles around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()

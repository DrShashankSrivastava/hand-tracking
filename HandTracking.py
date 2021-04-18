# Imports
import cv2
import mediapipe as mp
import time

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize hand tracking object
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Utility to draw on hand  image
mpDraw = mp.solutions.drawing_utils

# Temporal variables
pTime = 0      # Previous time
cTime = 0      # Current Time

# Start recording
while True:
    success, image = cap.read()

    # Convert image to RGB format for 'hands' object
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(imageRGB)

    # Check for multiple hands and process each of them
    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)

    # Calculate and display fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (75,75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)

    cv2.imshow("Feed image", image)
    cv2.waitKey(1)

import cv2
import numpy as np

# Load the Haar cascade for detecting waste items
from keras.saving.save import load_model

cascade = cv2.CascadeClassifier('waste_cascade.xml')

# Load the machine learning model for classifying waste items
model = load_model('D:/SHANTANU/TECH HACKS 3.0/keras_model.h5')

# Segregate the waste items into different categories
categories = {'paper': [], 'plastic': [], 'metals': [], 'others': []}

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect waste items in the frame
    waste_items = cascade.detectMultiScale(gray, 1.1, 3)

    # Classify and segregate the waste items
    for (x, y, w, h) in waste_items:
        # Extract the waste item from the frame
        waste_item = frame[y:y + h, x:x + w]

        # Predict the class of the waste item using the machine learning model
        prediction = model.predict(waste_item)

        # Segregate the waste item into the appropriate category
        if prediction == 'paper':
            categories['paper'].append(waste_item)
        elif prediction == 'plastic':
            categories['plastic'].append(waste_item)
        elif prediction == 'metal':
            categories['metals'].append(waste_item)
        else:
            categories['others'].append(waste_item)

    # Display the frame with waste items highlighted
    for (x, y, w, h) in waste_items:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Waste Segregation', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and destroy all windows
cap.release()
cv2.destroyAllWindows()

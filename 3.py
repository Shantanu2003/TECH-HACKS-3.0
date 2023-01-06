import cv2
import numpy as np

# Load the positive and negative images
def load_images(param):
    pass


glass = load_images('D:/SHANTANU/TECH HACKS 3.0/dataset-resized/glass/')
paper = load_images('D:/SHANTANU/TECH HACKS 3.0/dataset-resized/paper/')
metal = load_images('D:/SHANTANU/TECH HACKS 3.0/dataset-resized/metal/')
others = load_images('D:/SHANTANU/TECH HACKS 3.0/dataset-resized/trash/')

# Create the training data
X = []
y = []

for image in glass:
    X.append(image)
    y.append(1)

for image in paper:
    X.append(image)
    y.append(0)

for image in metal:
    X.append(image)
    y.append(0)

for image in others:
    X.append(image)
    y.append(0)

# Convert the training data to numpy arrays
X = np.array(X)
y = np.array(y)

# Create the Haar cascade classifier
classifier = cv2.CascadeClassifier()

# Train the classifier on the training data
classifier.train(X, y)

# Save the classifier to an XML file
classifier.save('waste_cascade.xml')

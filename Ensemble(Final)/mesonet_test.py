import cv2
import numpy as np
from modules.MesoNet.classifiers import Meso4

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('Ensemble(Final)/modules/MesoNet/weights/Meso4_DF.h5')

# 2 - Load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Resize the image to 256x256 and scale pixel values to [0, 1]
    image = cv2.resize(image, (256, 256)) / 255.0
    return image

# Replace 'path_to_your_image.jpg' with the path to your input image
input_image_path = 'Ensemble(Final)/modules/MesoNet/test_images/df/df01254.jpg'
image = preprocess_image(input_image_path)

# 3 - Predict
image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 256, 256, 3)
prediction = classifier.predict(image)
print(prediction)
print('Predicted:', prediction[0][0])

import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

img=cv2.imread("resources/test2.jpeg")
location = detector.detect_faces(img)
if len(location) > 0:
    for face in location:
        x, y, width, height = face['box']

cropped_image = img[y:y+height, x:x+width]
# cropped_image = cv2.resize(cropped_image, (224, 224))
cv2.imwrite("resources/Output2.jpg",cropped_image)
print("The Image was successfully saved")

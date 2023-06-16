import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

img=cv2.imread("C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/MTCNN_Face_Extraction/resources/test4.jpeg")
location = detector.detect_faces(img)
print(location)
if len(location) > 0:
    for face in location:
        x, y, width, height = face['box']
        print(x,y)

cropped_image = img[y:y+height, x:x+width]
# cropped_image = cv2.resize(cropped_image, (224, 224))
cv2.imwrite(("C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/MTCNN_Face_Extraction/resources/Output4.jpeg"),cropped_image)
print("The Image was successfully saved")

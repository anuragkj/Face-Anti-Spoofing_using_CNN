import os
from mtcnn import MTCNN
from PIL import Image
import cv2
from mtcnn.mtcnn import MTCNN

# Specify the paths for the input folder containing images, output folder, and the file to store the last number
input_folder = "./Input"
output_folder = "./Output"
last_number_file = "num.txt"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize MTCNN
detector = MTCNN()

# Read the last number from the file or use 0 as the default if the file doesn't exist
if os.path.exists(last_number_file):
    with open(last_number_file, "r") as f:
        last_number = int(f.read())
else:
    last_number = 0

# Iterate over the images in the input folder
for image_name in os.listdir(input_folder):
    # Check if the file is an image (you can modify the condition based on your image formats)
    if image_name.endswith((".jpg", ".jpeg", ".png")):
        # Load the image
        image_path = os.path.join(input_folder, image_name)
        print(image_path)
        # image = Image.open(image_path)
        img=cv2.imread(image_path)
        
        # Perform face detection using MTCNN
        try:
            
            location = detector.detect_faces(img)
            if len(location) > 0:
                for face in location:
                    x, y, width, height = face['box']

            cropped_image = img[y:y+height, x:x+width]
            output_image_name = f"image_{last_number}.jpg"
            output_image_path = os.path.join(output_folder, output_image_name)
            # cropped_image.save(output_image_path)
            cv2.imwrite(output_image_path,cropped_image)
            print("The Image was successfully saved")
            last_number += 1
        except: print("Error")
    # Delete the processed image file from the input folder
    os.remove(image_path)

# Store the last number in the file
with open(last_number_file, "w") as f:
    f.write(str(last_number))
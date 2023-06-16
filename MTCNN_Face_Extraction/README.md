# Face-detection-using-MTCNN
face detection using mtcnn complete implementation step by step

# Installation
Python version should be **3.7**. Run the following commands one by one sequently.

     pip install opencv-python==4.4.0.46
     
     pip install keras==2.3.1
     
     pip install tensorflow==2.2.1
     
     pip install mtcnn==0.1.0
     
     pip install numpy==1.16.6

# Face Detection in Image
Run the *"mtcnn_image.py"* file and keep your input image in the **"input & output"** folder and set your image path (or name) in the *"mtcnn_image.py"* file. The output save in the **"input & output"** folder with *"Outputfile.jpg"* name (you can also change output file name).
Test image is shown below:

![Outputfile](https://user-images.githubusercontent.com/57293126/104318957-f9404880-5501-11eb-8601-179d266d2540.jpg)

# Face Detection in .MP4 file
Run the *"mtcnn_video.py"* file and keep your input video in the **"input & output"** folder and set your image path (or name) in the *"mtcnn_video.py"* file. The output save in the **"input & output"** folder with *"filename.avi"* name (you can also change output file name).
Test Video is shown below:

![ezgif com-gif-maker](https://user-images.githubusercontent.com/57293126/104320928-c481c080-5504-11eb-99cd-1a10bdfdcfb1.gif)

# Face Detection using live camera
Run the *"mtcnn_live_cam.py"* file or run the following code for live face detection:

    import cv2
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()

    video = cv2.VideoCapture(0)

    if (video.isOpened() == False):
        print("Web Camera not detected")
    while (True):
        ret, frame = video.read()
        if ret == True:
            location = detector.detect_faces(frame)
            if len(location) > 0:
                for face in location:
                    x, y, width, height = face['box']
                    x2, y2 = x + width, y + height
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4)
            cv2.imshow("Output",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

## Live Camera Output

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/57293126/104323069-8c2fb180-5507-11eb-9de8-0ace88d632b0.gif)

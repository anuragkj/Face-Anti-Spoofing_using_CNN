import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
# Create an object to read
# from camera
video = cv2.VideoCapture("input & output/file2.mp4")

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('input & output/filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),29, size)
frame_num=0
while (True):
    ret, frame = video.read()
    frame_num += 1
    print(frame_num)
    if ret == True:

        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4)
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break


video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")

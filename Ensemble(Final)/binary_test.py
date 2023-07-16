import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from modules.binary.Model import DeePixBiS
from modules.binary.Loss import PixWiseBCELoss
from modules.binary.Metrics import predict, test_accuracy, test_loss
from modules.patch_depth.lib.processing_utils import get_file_list, FaceDection
import datetime
import cv2

def binary_pixel(test_dir, label, model, faceClassifier, tfms):

    # time_begin = datetime.datetime.now()
    file_list = get_file_list(test_dir)
    count = 0
    true_num = 0
    
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue
        try:
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
            #Add checking code to not crash even if face is not detected
            for x, y, w, h in faces:
                faceRegion = img[y:y + h, x:x + w]
                faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)

                faceRegion = tfms(faceRegion)
                faceRegion = faceRegion.unsqueeze(0)

                mask, binary = model.forward(faceRegion)
                res = torch.mean(mask).item()
                print(res)



            if res < 0.5:
                result = 0
            else:
                result = 1

            if result is None:
                continue
            if result == label:
                count += 1
                true_num += 1
            else:
                # print(file)
                count += 1
        except Exception as e:
            print(e)
    # print(count, true_num, true_num / count)

    # time_end = datetime.datetime.now()
    # time_all = time_end - time_begin
    # print("time_all", time_all.total_seconds())

    
    


if __name__ == '__main__':
    test_dir = "Ensemble(Final)/test_img_folder"
    label = 0
    model = DeePixBiS()
    model.load_state_dict(torch.load('Ensemble(Final)/modules/binary/DeePixBiS.pth'))
    #It is used to set the model in evaluation mode
    model.eval()

    #tfms are the transformations to apply
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    faceClassifier = cv.CascadeClassifier('Ensemble(Final)/modules/binary/Classifiers/haarface.xml')
    binary_pixel(test_dir=test_dir, label=label, model = model, faceClassifier = faceClassifier, tfms = tfms)
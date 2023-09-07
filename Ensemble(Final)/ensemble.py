import sys
import cv2
import numpy as np
from PIL import Image
import torch
import os
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from modules.patch_depth.model.patch_based_cnn import net_baesd_patch, patch_test_transform
from modules.patch_depth.model.depth_based_cnn import net_baesd_depth, depth_test_transform
from modules.patch_depth.lib.processing_utils import get_file_list, FaceDection
from modules.patch_depth.lib.model_develop_utils import deploy_base
from modules.patch_depth.configuration.config_patch import args
from modules.MesoNet.classifiers import Meso4
import torchvision.transforms as ts


import torch.nn as nn
from torchvision import transforms
import numpy as np
from modules.binary.Model import DeePixBiS
from modules.binary.Loss import PixWiseBCELoss
from modules.binary.Metrics import predict, test_accuracy, test_loss
from modules.patch_depth.lib.processing_utils import get_file_list, FaceDection
# import datetime


class rgb2ycrcb(object):
    '''
    自定义transform处理,将rgb图像转ycrcb
    :param object:
    :return:
    '''

    def __call__(self, img):
        img_new = img.convert("YCbCr")
        return img_new


class RandomCrop(object):

    def __init__(self, size, seed, path_dir=None, choice=0):
        self.seed = seed
        self.size = (int(size), int(size))
        self.path_dir = path_dir
        self.choice = choice

    def get_params(self, img, output_size):
        img = np.array(img)
        img_shape = img.shape
        w = img_shape[0]
        h = img_shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        np.random.seed(self.seed)
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):#choice 0 for random, 1 for specific
        if self.choice == 0:
            i, j, h, w = self.get_params(img, self.size)
            img = np.array(img)
            img_new = img[j:j + h, i:i + w]
            try:
                img_new = Image.fromarray(img_new.astype('uint8')).convert('RGB')
            except Exception as e:
                print("Image.fromarray(img.astype('uint8')).convert('RGB')")
                i, j, h, w = self.get_params(img, self.size)
            return img_new
        else:
            cascade_file = cv2.CascadeClassifier(self.path_dir)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            specific_patch = cascade_file.detectMultiScale(gray)
            
            if len(specific_patch) > 0:
                region = random.choice(specific_patch)
                x, y, w, h = region
                ph, tw = self.size
                detected_patch = img[y-((ph-h)//2):y+h+((ph-h)//2), x-((tw-w)//2):x+w+((tw-w)//2)]
                detected_patch = Image.fromarray(detected_patch)
                detected_patch = detected_patch.resize(self.size, Image.BILINEAR)
                return detected_patch
            else:
                return None
    

# Add specific patch detection along with random patches
def patch_cnn_single(model, face_detector, img, isface, classifiers):
    '''

    :param model:
    :param face_detector:
    :param img:
    :param isface: the img is face img or not
    :return:
    '''
    if not isface:
        img = cv2.resize(img, (480, 640))
        # 人脸检测
        face_img = face_detector.face_detect(img)

    else:
        face_img = img

    if face_img is None:
        return None

    # 随机裁剪
    patch_size = 96
    img_Image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    data_len = 8
    seed_arr = np.arange(data_len)

    true_count = 0
    false_count = 0
    for i in range(data_len):
        img_transform = ts.Compose([RandomCrop(size=patch_size, seed=seed_arr[i], path_dir = None, choice = 0)])
        try:
            img_patch = img_transform(img = img_Image)

            result_one = deploy_base(model=model, img=img_patch, transform=patch_test_transform)
            result_one=result_one[0]

            if result_one[0] > result_one[1]:
                false_count += 1
            else:
                true_count += 1
        except Exception as e:
            pass
            # print(e)
    specific_true = 0
    specific_false = 0
    for i in classifiers:
        img_transform = ts.Compose([RandomCrop(size=patch_size, seed=seed_arr[0], path_dir = classifiers[i], choice = 1)])
        try:
            img_patch = img_transform(img = face_img)

            result_one = deploy_base(model=model, img=img_patch, transform=patch_test_transform)
            result_one=result_one[0]

            if result_one[0] > result_one[1]:
                false_count += 1
                specific_false += 1
            else:
                true_count += 1
                specific_true += 1
        except Exception as e:
            pass
            # print(e)

    # 集成判断
    ret_dic = {}
    # print("Patch: ")
    # print("true_count", true_count, "false_count", false_count)
    # print("specific_true_count", specific_true, "specific_false_count", specific_false)
    # print(true_count/(true_count + false_count))
    ret_dic["true_count"] = true_count
    ret_dic["false_count"] = false_count
    ret_dic["specific_true_count"] = "specific_true_count"
    ret_dic["specific_false_count"] = "specific_false_count"
    ret_dic["result"] = true_count/(true_count + false_count)
    return ret_dic


def depth_cnn_single(model, face_detector, img, isface):
    '''

    :param model:
    :param face_detector:
    :param img:
    :param isface: the img is face img or not
    :return:
    '''
    if not isface:
        img = cv2.resize(img, (480, 640))
        # 人脸检测
        face_img = face_detector.face_detect(img)

    else:
        face_img = img

    if face_img is None:
        return None

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img_pil = Image.fromarray(face_img)

    result = deploy_base(model=model, img=face_img_pil, transform=depth_test_transform)

    result_mean = np.mean(result)
    # print("Depth: ")
    # print(result_mean)

    return result_mean

def binary_supervision_single(model, img, faceClassifier, tfms):
    try:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
        #Add checking code to not crash even if face is not detected
        for x, y, w, h in faces:
            faceRegion = img[y:y + h, x:x + w]
            faceRegion = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)

            faceRegion = tfms(faceRegion)
            faceRegion = faceRegion.unsqueeze(0)

            mask, binary = model.forward(faceRegion)
            res = torch.mean(mask).item()
            # print("Binary:")
            # print(res)

        return res
    except Exception as e:
        # print(e)
        return None

def deepfake_detection_single(pre_path_deepfake, image):
    classifier = Meso4()
    classifier.load(pre_path_deepfake)
    image = cv2.resize(image, (256, 256)) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 256, 256, 3)
    prediction = classifier.predict(image)
    return prediction[0][0]

# Function to go over all images and call the single functions for depth, patch and binary analyis.
def ensemble_test(args, pre_path_depth, pre_path_patch_surf, pre_path_patch_fasd, pre_path_binary, pre_path_deepfake_Meso_4_DF, 
                  pre_path_deepfake_Meso_4_F2F, pre_path_deepfake_Meso_Inc_DF, pre_path_deepfake_Meso_Inc_F2F, binary_face_classifier_path, 
                  test_dir, isface, classifiers):
    '''

    :param :
    :param pre_path: road to pretrain model
    :param test_dir: test img dir
    :param isface: img is face img or not, if not need to detect face
    :return:
    '''
    face_detector = FaceDection(model_name='cv2')

    #Preparing all models

    #Preparing depth_model
    model_depth = net_baesd_depth()
    state_dict_depth = torch.load(pre_path_depth, map_location='cpu')
    model_depth.load_state_dict(state_dict_depth)
    model_depth.eval()

    #Preparing patch_model surf
    model_patch_surf = net_baesd_patch(args)
    state_dict_patch_surf = torch.load(pre_path_patch_surf, map_location='cpu')
    model_patch_surf.load_state_dict(state_dict_patch_surf)
    model_patch_surf.eval()
    
    #Preparing patch_model fasd
    model_patch_fasd = net_baesd_patch(args)
    state_dict_patch_fasd = torch.load(pre_path_patch_fasd, map_location='cpu')
    model_patch_fasd.load_state_dict(state_dict_patch_fasd)
    model_patch_fasd.eval()

    #Preparing binary_model
    faceClassifier = cv2.CascadeClassifier(binary_face_classifier_path)
    model_binary_supervision = DeePixBiS()
    model_binary_supervision.load_state_dict(torch.load(pre_path_binary))
    model_binary_supervision.eval()
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Iterating over the files
    file_list = get_file_list(test_dir)
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue

        result_deepfake__Meso_4_DF = deepfake_detection_single(pre_path_deepfake_Meso_4_DF, img)
        result_deepfake__Meso_4_F2F = deepfake_detection_single(pre_path_deepfake_Meso_4_F2F, img)
        result_deepfake__Meso_Inc_DF = deepfake_detection_single(pre_path_deepfake_Meso_Inc_DF, img)
        result_deepfake__Meso_Inc_F2F = deepfake_detection_single(pre_path_deepfake_Meso_Inc_F2F, img)
        result_depth = depth_cnn_single(model=model_depth, face_detector=face_detector, img=img, isface=isface)
        result_patch_surf = patch_cnn_single(model=model_patch_surf, face_detector=face_detector, img=img, isface=isface, classifiers=classifiers)
        result_patch_fasd = patch_cnn_single(model=model_patch_fasd, face_detector=face_detector, img=img, isface=isface, classifiers=classifiers)
        result_binary = binary_supervision_single(model=model_binary_supervision, img = img, faceClassifier = faceClassifier, tfms = tfms)

        print("============================================================")
        print(file)
        print()
        print("Real Image(Not a deepfake) result_deepfake__Meso_4_DF: " + str(result_deepfake__Meso_4_DF))
        print("Real Image(Not a deepfake) result_deepfake__Meso_4_F2F: " + str(result_deepfake__Meso_4_F2F))
        print("Real Image(Not a deepfake) result_deepfake__Meso_Inc_DF: " + str(result_deepfake__Meso_Inc_DF))
        print("Real Image(Not a deepfake) result_deepfake__Meso_Inc_F2F: " + str(result_deepfake__Meso_Inc_F2F))
        print("Depth: " + str(result_depth))
        print("Patch Surf: " + str(result_patch_surf["result"])) #Can use and print other dictionary keys also
        print("Patch Fasd: " + str(result_patch_fasd["result"])) #Can use and print other dictionary keys also
        print("Binary Supervision: " + str(result_binary))
        print()
        print("============================================================")        



if __name__ == '__main__':
    test_dir = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/test_img_folder'
    pre_path_patch_surf = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/output/models/patch_surf.pth'
    pre_path_patch_fasd = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/output/models/patch_fasd.pth'
    pre_path_depth = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/output/models/depth_patch.pth'
    pre_path_binary = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/binary/DeePixBiS.pth'
    pre_path_deepfake_Meso_4_DF = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/MesoNet/weights/Meso4_DF.h5'
    pre_path_deepfake_Meso_4_F2F = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/MesoNet/weights/Meso4_F2F.h5'
    pre_path_deepfake_Meso_Inc_DF = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/MesoNet/weights/Meso4_F2F.h5'
    pre_path_deepfake_Meso_Inc_F2F = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/MesoNet/weights/Meso4_F2F.h5'


    isface = True
    classifiers = {
        'left_ear' : 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_leftear.xml',
        'left_eye' : 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_lefteye.xml',
        'right_eye': 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_righteye.xml',
        'right_ear': 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_rightear.xml',
        'nose'     : 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_nose.xml',
        'mouth'    : 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/patch_depth/Classifiers/haarcascade_mcs_mouth.xml',
    }

    binary_face_classifier_path = 'C:/Users/anura/Documents/Github/Face-Anti-Spoofing_using_CNN/Ensemble(Final)/modules/binary/Classifiers/haarface.xml'
    ensemble_test(args, pre_path_depth, pre_path_patch_surf, pre_path_patch_fasd, pre_path_binary, pre_path_deepfake_Meso_4_DF, 
                  pre_path_deepfake_Meso_4_F2F, pre_path_deepfake_Meso_Inc_DF, pre_path_deepfake_Meso_Inc_F2F, binary_face_classifier_path, 
                  test_dir, isface, classifiers)

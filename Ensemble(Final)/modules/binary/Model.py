import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torch.sigmoid as sigmoid


# Semantic segmentation is a technique used in computer vision to classify each pixel in an image into different classes 
# or categories. In other words, it involves dividing an image into 
# regions or segments and assigning a label to each segment based on its content. 


#defining a neural network DeePixBiS for image segmentation
class DeePixBiS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        dense = models.densenet161(pretrained=pretrained)#popular pretrained model for image segmentation
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])#extracting the first 8 layers of its feature extraction module
        #output of the feature extraction module passed through convolution layer with given kernel and stride to get feature map
        
        #single pixel at a time, and moves pixel by pixel, channel=384 = no of feature maps obtained
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        #The feature map is then flattened and passed through a fully connected linear layer to generate the final output.
        #its 14*14 since at the end of 8th layer in densenet161 the spatial res is 14*14
        self.linear = nn.Linear(14 * 14, 1)

#A linear layer in a Convolutional Neural Network (CNN) is typically used to perform a final classification
#feature map is a representation of the important features that the CNN has detected in the input image., here it has size 14*14
    def forward(self, x):
        #nput image is passed through the feature extraction module 
        enc = self.enc(x)
        #resulting feature map is passed through the convolutional layer
        dec = self.dec(enc)
        #sigmoid activation function to generate an output map.
        out_map = F.sigmoid(dec)
        # print(out_map.shape)
        #out_map tensor from a 4-dimensional tensor of shape (batch_size, num_channels, height, width) 
        #The output map is then flattened and passed through the linear layer 
        # and sigmoid activation function to generate the final output.
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out



'''
USING XCEPTION
https://github.com/chaymabh/Transfer-Learning-with-Xception

import torch
import torch.nn as nn
import torchvision.models as models

class XceptionModel(nn.Module):
    def __init__(self, pretrained=True, pixelwise_output=False):
        super(XceptionModel, self).__init__()

        # Load the Xception base model
        base_model = models.xception(pretrained=pretrained)

        # Create a Sequential model for feature extraction
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Create a Global Average Pooling layer to obtain feature maps of size 14x14
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Create a model for pixel-wise output prediction if enabled
        if pixelwise_output:
            self.pixelwise_output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 14 * 14),  # 2048 is the number of output features from the base model
                nn.Sigmoid(),
                nn.Unflatten(1, (14, 14))
            )

    def forward(self, x):
        # Pass input through the feature extractor
        features = self.feature_extractor(x)
        features = self.global_avg_pooling(features)
        features = features.view(features.size(0), -1)

        # If pixel-wise output is enabled, predict it
        if hasattr(self, 'pixelwise_output'):
            pixelwise_out = self.pixelwise_output(features)
            return features, pixelwise_out
        else:
            return features
'''


'''
Explanation:
The proposed model, denoted as XceptionModel, is a deep neural network architecture designed for facial 
anti-spoofing, which plays a pivotal role in securing biometric systems from fraudulent access attempts. 
XceptionModel leverages the power of transfer learning by utilizing a pre-trained Xception base model, 
initially developed for image classification tasks. This base model is used for feature extraction from facial 
images, allowing the network to capture discriminative representations that distinguish between genuine and 
spoofed faces.

The XceptionModel primarily focuses on two critical aspects: feature extraction and pixel-wise output prediction.
For feature extraction, the base model is employed to generate high-level features from the input images. 
Subsequently, a global average pooling layer condenses these features into a fixed-size representation, 
ensuring compatibility with the succeeding stages of the model. Notably, the use of global average pooling 
results in feature maps of size 14x14, which provides valuable spatial information for anti-spoofing.

The model's flexibility is highlighted by its capacity for pixel-wise output prediction. If enabled, the network 
can predict a pixel-wise mask that indicates the likelihood of each pixel being part of a spoofed region in the 
input image. This pixel-wise output further enhances the model's sensitivity to partial spoofing attacks, 
allowing it to account for a wide range of presentation attack scenarios.

In the context of facial anti-spoofing, the XceptionModel excels by harnessing deep learning techniques and 
leveraging a well-established base model. Its ability to extract meaningful features and, when required, 
provide pixel-wise spoofing predictions empowers anti-spoofing systems to detect fraudulent attempts with 
enhanced accuracy and adaptability, contributing significantly to the security of face recognition systems.

The XceptionModel represents a significant advancement in facial anti-spoofing, a vital component of biometric
security systems. This model is founded upon transfer learning, capitalizing on a pre-trained Xception base
model initially designed for image classification. The primary objective of the XceptionModel is to 
discriminate between genuine and spoofed faces by extracting highly discriminative features from facial 
images.

   


At its core, the XceptionModel is designed to accomplish two fundamental tasks: feature extraction and 
pixel-wise output prediction. Feature extraction is pivotal for capturing essential facial characteristics, 
and for this purpose, the model employs the pre-trained Xception base model to generate high-level features. 
Subsequent to feature extraction, a global average pooling layer is employed to create a compact fixed-size 
representation. This reduction in feature dimensions is critical for ensuring compatibility with the later 
stages of the model.

One of the key technical aspects of the XceptionModel is its capability to provide pixel-wise output p
redictions. If activated, the model predicts a pixel-wise mask indicating the probability of each pixel's 
inclusion in a spoofed region within the input image. This pixel-wise prediction is instrumental in dealing 
with partial spoofing attacks, as it affords the model the ability to account for these nuanced forms of 
presentation attacks.

In a technical context, the XceptionModel significantly enhances facial anti-spoofing systems by effectively 
leveraging deep learning techniques and a well-established base model. Its feature extraction prowess, combined 
with its potential for precise pixel-wise spoofing predictions, equips anti-spoofing systems to detect a broader 
spectrum of fraudulent attempts with heightened accuracy and adaptability. Consequently, the XceptionModel 
serves as a crucial tool in augmenting the security and robustness of face recognition systems.


COPLIOT EXPLANATION:
The active selection is a Python class named XceptionModel that inherits from nn.Module, a base class for all 
neural network modules provided by PyTorch. This class is designed to implement a deep learning model using the 
Xception architecture, which is a convolutional neural network model proposed by Google.

In the __init__ method, a pre-trained Xception model is loaded using the torchvision.models module. The feature 
extraction part of the Xception model is separated from the final fully connected layer by creating a sequential 
model self.feature_extractor with all layers of the Xception model except the last one.

Next, a Global Average Pooling layer self.global_avg_pooling is defined. This layer is used to reduce the 
spatial dimensions of the output feature maps from the feature extractor to 1x1.

If pixel-wise output is enabled, a sequential model self.pixelwise_output is defined for pixel-wise output 
prediction. This model consists of a Flatten layer, a Linear layer with 2048 input features and 14*14 output 
features, a Sigmoid activation function, and an Unflatten layer that reshapes the output to a 14x14 grid.

In the forward method, the input is passed through the feature extractor and the Global Average Pooling layer. 
The output feature maps are then reshaped to a 2D tensor with the first dimension being the batch size and the 
second dimension being the number of features.

If pixel-wise output is enabled, the reshaped feature maps are passed through the pixel-wise output prediction 
model, and the output of this model is returned along with the reshaped feature maps. If pixel-wise output is 
not enabled, only the reshaped feature maps are returned.
'''
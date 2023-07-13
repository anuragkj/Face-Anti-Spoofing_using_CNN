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

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, RandomRotation
import numpy as np
from torch.utils.data import DataLoader
from Dataset import PixWiseDataset
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
from Trainer import Trainer
#training pipeline for the DeePixBiS algorithm
model = DeePixBiS()
#state dic loaded from pretrained model save
model.load_state_dict(torch.load('./DeePixBiS.pth'))
#Loss class loaded from loss.py
loss_fn = PixWiseBCELoss()
#optimizer (Adam) is initialized to update the parameters of the model during training.
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

#Two sets of image transformations are defined: one for training data and another for validation/test data. 
# The training transformations include resizing, random horizontal flipping, random rotation, conversion to tensor, and normalization. 
# The test transformations include resizing, conversion to tensor, and normalization.
#Compose composes several transformations together
train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = PixWiseDataset('./train_data.csv', transform=train_tfms)
train_ds = train_dataset.dataset()

val_dataset = PixWiseDataset('./test_data.csv', transform=test_tfms)
val_ds = val_dataset.dataset()

batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)#Find num_workers and pin_memory
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

# for x, y, z in val_dl:
# 	_, zp = model(x)
# 	print(zp)
# 	print (z)
# 	break

# print(test_accuracy(model, train_dl))
# print(test_loss(model, train_dl, loss_fn))

# 5 epochs ran
#The trainer fits the model by iterating over each epoch and calling the train_one_epoch method.
trainer = Trainer(train_dl, val_dl, model, 20, opt, loss_fn)

print('Training Beginning\n')
trainer.fit()
print('\nTraining Complete')
torch.save(model.state_dict(), './DeePixBiS.pth')

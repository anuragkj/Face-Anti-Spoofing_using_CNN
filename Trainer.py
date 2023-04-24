import torch
import torch.nn as nn
from Metrics import test_accuracy, test_loss

#trained for 1 epoch
class Trainer():
    def __init__(self, train_dl, val_dl, model, epochs, opt, loss_fn, device='cpu'):
        self.train_dl = train_dl
        self.val_dl = val_dl
        #to(device) is a method that is used to move a model or a tensor to a specified device, 
        # which could be either CPU or GPU
        self.model = model.to(device)
        self.epochs = epochs
        self.opt = opt
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, num):
        #Takes epoch no. as input and prints it
        print(f'\nEpoch ({num+1}/{self.epochs})')
        print('----------------------------------')
        # self.model.train()
        #Loops through each batch in training dataloader
        for batch, (img, mask, label) in enumerate(self.train_dl):
            # For each batch, it loads the batch of images, masks, and labels into the GPU or CPU
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            #Getting the predicted value
            net_mask, net_label = self.model(img)
            loss = self.loss_fn(net_mask, net_label, mask, label)

            # Train
            #After calculating the loss, it sets the optimizer gradients to zero (self.opt.zero_grad()), 
            # backpropagates the loss through the model (loss.backward()), 
            # and updates the model parameters using the optimizer (self.opt.step()).
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if batch % 9 == 0:
                print(f'Loss : {loss}')

        # self.model.eval()
        test_acc = test_accuracy(self.model, self.val_dl)
        test_los = test_loss(self.model, self.val_dl, self.loss_fn)

        print(f'Test Accuracy : {test_acc}  Test Loss : {test_los}')
        return test_acc, test_los

    def fit(self):
        training_loss = []
        training_acc = []
        self.model.train()
        #For each epoch, the model is trained for one epoch
        for epoch in range(self.epochs):
            train_acc, train_loss = self.train_one_epoch(epoch)
            training_acc.append(train_acc)
            training_loss.append(train_loss)

        return training_acc, training_loss

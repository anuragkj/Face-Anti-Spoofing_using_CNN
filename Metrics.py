import torch
import numpy


def predict(mask, label, threshold=0.5, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            #The mask tensor is of shape [batch_size, num_channels, height, width]. 1 2 and 3 give the spatial dimensions
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == 'binary':
            score = label
        else:
            score = (torch.mean(mask, axis=(1, 2, 3)) + label) / 2

        preds = (score > threshold).type(torch.FloatTensor)

        return preds, score

#For predicting accuracy
def test_accuracy(model, test_dl):
    acc = 0
    total = len(test_dl.dataset)
    for img, mask, label in test_dl:
        #net mask and net label are the predicted values
        net_mask, net_label = model(img)
        #we send the predicted mask and label values into the predict function, 
        # the returned value consist of the binary classified prediction and the score
        preds, _ = predict(net_mask, net_label)
        ac = (preds == label).type(torch.FloatTensor)
        acc += torch.sum(ac).item()
    return (acc / total) * 100

#For predicting loss
def test_loss(model, test_dl, loss_fn):
    loss = 0
    total = len(test_dl)
    for img, mask, label in test_dl:
        net_mask, net_label = model(img)
        losses = loss_fn(net_mask, net_label, mask, label)
        loss += torch.mean(losses).item()
    return loss / total


#########################
def evaluate(model, test_dl, threshold=0.5): #threshold which is a threshold value for the decision function(0.5 default)
    total = len(test_dl.dataset)
    apcer_total = 0
    bpcer_total = 0
    for img, mask, label in test_dl:
        with torch.no_grad():#ensures that PyTorch does not keep track of gradients since we are not training the model
            net_mask, net_label = model(img)
            preds, score = predict(net_mask, net_label, threshold)

        genuine_scores = score[label == 0]
        attack_scores = score[label == 1]

        #Here, preds is a tensor of predicted labels (0 for genuine and 1 for attack) and label is a tensor of true labels. 
        #APCER is the proportion of attack samples that were incorrectly classified as genuine by the model, 
        # so we should sum up the number of attack samples that were misclassified as genuine. We can do this 
        # by counting the number of times the model predicted "0" (genuine) for an attack sample with label "1"

        # Calculate APCER
        #apcer_total += torch.sum(preds[label == 1]).item()#For APCER, it sums up the number of predictions that are 1 and their corresponding labels are also 1 using
        apcer_total += torch.sum((preds == 0) & (label == 1)).item()

        # Calculate BPCER
        #bpcer_total += torch.sum(1 - preds[label == 0]).item()#For BPCER, it sums up the number of predictions that are 0 and their corresponding labels are also 0 using
        bpcer_total += torch.sum((preds == 1) & (label == 0)).item()
    # Calculate APCER and BPCER
    apcer = apcer_total / total
    bpcer = bpcer_total / total

    # Calculate HTER
    hter = (apcer + bpcer) / 2

    return apcer, bpcer, hter, apcer_total, bpcer_total, total
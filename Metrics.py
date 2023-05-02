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
def evaluate(model, test_dl, threshold=0.5):
    with torch.no_grad():
        total = len(test_dl.dataset)
        apcer = 0
        bpcer = 0
        for img, mask, label in test_dl:
            net_mask, net_label = model(img)
            preds, score = predict(net_mask, net_label, threshold)

            genuine_scores = score[label == 0]
            attack_scores = score[label == 1]

            # Calculate APCER
            apcer += torch.sum(attack_scores > threshold).item() / len(attack_scores)

            # Calculate BPCER
            bpcer += torch.sum(genuine_scores < threshold).item() / len(genuine_scores)

        apcer /= total
        bpcer /= total
        hter = (apcer + bpcer) / 2

    return apcer, bpcer, hter
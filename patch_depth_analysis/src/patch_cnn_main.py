import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
import sys

sys.path.append('../')

from model.patch_based_cnn import net_baesd_patch, patch_data_loader
from lib.model_develop_utils import train_base
from configuration.config_patch import args



def patch_cnn_train(args):
    '''

    :return:
    '''
    train_loader = patch_data_loader(args,train=True)
    test_loader = patch_data_loader(args,train=False)

    model = net_baesd_patch(args=args)
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    args.retrain = False
    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    patch_cnn_train(args)

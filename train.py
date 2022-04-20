import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#this file for training the models with the unattacked dataset

import torch
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

import pdb




def run_code_for_training(net,loaders,net_save_path):
    train_data_loader = loaders['train']
    validation_data_loader = loaders['test']

    net = copy.deepcopy(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.train()
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-3, betas=(0.9, 0.999))
    epochs = 50
    Loss_runtime = []
    accc = []
    for epoch in range(epochs):
        start_time = time.time()
        print("---------starting epoch %s ---------" % (epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1,3,1,1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                Loss_runtime.append(running_loss / float(500))
                running_loss = 0.0

        acc = run_code_for_validation_direct(net, validation_data_loader)
        accc.append(acc)
        print("---------time executed for training in epoch %s : %s seconds---------" % (epoch + 1, (time.time() - start_time)))

    torch.save(net, net_save_path)

    return Loss_runtime, accc

def run_code_for_validation_direct(net, validation_data_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = copy.deepcopy(net)
    net.eval()
    net = net.to(device)
    correct = 0
    total = 0
    for i, data in enumerate(validation_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for label, prediction in zip(labels, predicted):
            total += 1
            if label == prediction:
                correct += 1
    print("accuracy: ", correct/total)
    return correct/total



if __name__ == "__main__":
    
    model1 = models.resnet50()
    model2 = models.mobilenet_v2()
    model3 = models.efficientnet_b0()
    # googlenet = models.googlenet(pretrained=True)

    transform = tvt.Compose([tvt.ToTensor()])
    # train_data = datasets.MNIST(root = 'data', train = True, transform = tvt.ToTensor(), download = True)
    # test_data = datasets.MNIST(root = 'data', train = False, transform = tvt.ToTensor())
    train_data = datasets.CIFAR10(root = 'data', train = True, transform = transform, download = True)
    test_data = datasets.CIFAR10(root = 'data', train = False, transform = transform)
    loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)}



    loss1,acc1 = run_code_for_training(model1,loaders,"net1.pth")
    loss2,acc2 = run_code_for_training(model2,loaders,"net2.pth")
    loss3,acc3 = run_code_for_training(model3,loaders,"net3.pth")

    plt.figure()
    plt.plot(loss1, label="Resnet50 Training Loss")
    plt.plot(loss2, label="Mobilenet V2 Training Loss")
    plt.plot(loss3, label = "Efficientnet B0 Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("train_loss.jpg")

    plt.figure()
    plt.plot(acc1, label="Resnet50 Runtime Validation Acc")
    plt.plot(acc2, label="Mobilenet V2 Runtime Validation Acc")
    plt.plot(acc3, label = "Efficientnet B0 Runtime Validation Acc")
    plt.xlabel("Iteration")
    plt.ylabel("Acc")
    plt.title("Runtime Accuracy")
    plt.legend()
    plt.savefig("runtime_acc.jpg")
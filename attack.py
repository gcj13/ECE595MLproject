import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#this file for attacking the dataset

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
from PIL import Image
import PIL

import pdb

def generate_clean_dataset():
    train_data = datasets.CIFAR10(root = 'data', train = True, transform = None, download = True)
    test_data = datasets.CIFAR10(root = 'data', train = False, transform = None)
    train_label = []
    for i in range(50000):
        train_data[i][0].save('unattacked_dataset/cifar10/train/'+str(i)+'.jpg')
        train_label.append(train_data[i][1])
    np.savetxt('unattacked_dataset/cifar10/train_label.txt', train_label, fmt='%i')
    test_label = []
    for i in range(10000):
        test_data[i][0].save('unattacked_dataset/cifar10/test/'+str(i)+'.jpg')
        test_label.append(test_data[i][1])
    np.savetxt('unattacked_dataset/cifar10/test_label.txt', test_label, fmt='%i')

    train_data = datasets.MNIST(root = 'data', train = True, transform = None, download = True)
    test_data = datasets.MNIST(root = 'data', train = False, transform = None)
    train_label = []
    for i in range(60000):
        train_data[i][0].save('unattacked_dataset/mnist/train/'+str(i)+'.jpg')
        train_label.append(train_data[i][1])
    np.savetxt('unattacked_dataset/mnist/train_label.txt', train_label, fmt='%i')
    test_label = []
    for i in range(10000):
        test_data[i][0].save('unattacked_dataset/mnist/test/'+str(i)+'.jpg')
        test_label.append(test_data[i][1])
    np.savetxt('unattacked_dataset/mnist/test_label.txt', test_label, fmt='%i')



if __name__ == "__main__":

    #first generate the dataset to data with images

    # generate_clean_dataset()

    #generate attacked dataset.

    
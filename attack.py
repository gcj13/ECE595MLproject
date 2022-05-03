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
import torchattacks
import statistics
import cv2


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

def generate_attacked_dataset(eps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load models
    model1 = torch.load('net1.pth')
    model2 = torch.load('net2.pth')
    model3 = torch.load('net3.pth')

    #load dataset
    images = []
    labels = []
    # train_data = datasets.CIFAR10(root = 'data', train = True, transform = None, download = True)
    test_data = datasets.CIFAR10(root='data', train=False, transform=tvt.ToTensor())
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle = False, num_workers=0)
    # pdb.set_trace()
    #generate adv images

    # for i in test_data:
    #     image,label = i
    #     #image (H,W,C)->(C,H,W)
    #
    #     images.append(np.transpose(np.array(image), (1,2,0)))            #
    #     labels.append(label)
    adv1 = []
    adv2 = []
    adv3 = []

    attack1 = torchattacks.PGD(model1, eps=eps, alpha=1 / 255, steps=40, random_start=True)
    attack2 = torchattacks.PGD(model2, eps=eps, alpha=1 / 255, steps=40, random_start=True)
    attack3 = torchattacks.PGD(model3, eps=eps, alpha=1 / 255, steps=40, random_start=True)


    for i, data in enumerate(test_data_loader):
        images, labels = data
        adv_images1 = attack1(images, labels)
        adv_images2 = attack2(images, labels)
        adv_images3 = attack3(images, labels)
        for j in range(len(adv_images1)):
            adv1.append((adv_images1[i],labels[i]))
            adv2.append((adv_images2[i],labels[i]))
            adv3.append((adv_images3[i],labels[i]))
            # adv_images1[i].save('cifar10_adv/adv_images1/' + str(j+len(adv_images1)*i) + '.jpg')
            # adv_images2[i].save('cifar10_adv/adv_images2/' + str(j+len(adv_images1)*i) + '.jpg')
            # adv_images3[i].save('cifar10_adv/adv_images3/' + str(j+len(adv_images1)*i) + '.jpg')
    return adv1,adv2,adv3


def ensemble_prediction(model1,model2,model3,test_data,mode):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model1 = copy.deepcopy(model1)
    model1.eval()
    model1 = model1.to(device)
    model2 = copy.deepcopy(model2)
    model2.eval()
    model2 = model2.to(device)
    model3 = copy.deepcopy(model3)
    model3.eval()
    model3 = model3.to(device)
    if mode == 'ensemble':
        total = 0
        correct = 0
        for data, label in test_data:
            data = torch.unsqueeze(data,dim=0)
            data = data.to(device)
            label = label.to(device)
            output1 = model1(data)
            output2 = model2(data)
            output3 = model3(data)
            # pdb.set_trace()
            outputs = output3 + output2 + output1
            _, predicted = torch.max(outputs.data, 1)
            # for label, predicted in zip(label, predicted):
            total += 1
            if label == predicted:
                correct += 1
        print("accuracy: ", correct/total)
        return correct/total
    elif mode == 'seperate':
        total = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        for data, label in test_data:
            data = torch.unsqueeze(data,dim=0)
            data = data.to(device)
            label = label.to(device)
            output1 = model1(data)
            output2 = model2(data)
            output3 = model3(data)
            # pdb.set_trace()
            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            _, predicted3 = torch.max(output3.data, 1)
            # for label, predicted in zip(label, predicted):
            total += 1
            if label == predicted1:
                correct1 += 1
            if label == predicted2:
                correct2 += 1
            if label == predicted3:
                correct3 += 1
        print("accuracy: ", correct1/total, correct2/total, correct3/total)
        return correct1/total,correct2/total,correct3/total
    elif mode == 'ens_vote':
        total = 0
        correct = 0
        for data, label in test_data:
            data = torch.unsqueeze(data,dim=0)
            data = data.to(device)
            label = label.to(device)
            output1 = model1(data)
            output2 = model2(data)
            output3 = model3(data)
            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            _, predicted3 = torch.max(output3.data, 1)
            # for label, predicted in zip(label, predicted):
            pred_l = torch.cat((predicted3,predicted1,predicted2),0).tolist()
            predicted = statistics.mode(pred_l)
            # pdb.set_trace()
            total += 1
            if label.item() == predicted:
                correct += 1
        print("accuracy: ", correct/total)
        return correct/total

if __name__ == "__main__":

    #first generate the dataset to data with images

    # generate_clean_dataset()

    #generate attacked dataset.

    # adv1,adv2,adv3 = generate_attacked_dataset(1/255)

    # pdb.set_trace()
    # adv = adv1 + adv2 + adv3
    model1 = torch.load('net1.pth')
    model2 = torch.load('net2.pth')
    model3 = torch.load('net3.pth')

    # acc = ensemble_prediction(model1,model2,model3,adv,'ensemble')
    # acc1 = ensemble_prediction(model1, model2, model3, adv1,'ensemble')
    # acc2 = ensemble_prediction(model1, model2, model3, adv2,'ensemble')
    # acc3 = ensemble_prediction(model1, model2, model3, adv3,'ensemble')
    #
    # print(acc,acc1,acc2,acc3) #0.11666666666666667 0.12 0.17 0.06 8/255 preturbation | 0.03 0.03 0.06 0.0   0.1 perturbation  |  0.04 0.02 0.09 0.01  16/255 preturbation
    #
    #
    # acc1,acc2,acc3 = ensemble_prediction(model1,model2,model3,adv,'seperate')
    # print(acc1,acc2,acc3) # 0.43333333333333335 0.3333333333333333 0.41333333333333333 8/255 preturbation | 0.37666666666666665 0.19333333333333333 0.23333333333333334    0.1 perturbation  |  0.4033333333333333 0.21 0.2966666666666667  16/255 preturbation
    # acc1,acc2,acc3 = ensemble_prediction(model1, model2, model3, adv1,'seperate')
    # print(acc1,acc2,acc3) # 0.02 0.42 0.54  8/255 preturbation |  0.02 0.19 0.26   0.1 perturbation  |  0.02 0.23 0.31  16/255 preturbation
    # acc1,acc2,acc3 = ensemble_prediction(model1, model2, model3, adv2,'seperate')
    # print(acc1,acc2,acc3) # 0.62 0.0 0.7  8/255 preturbation | 0.53 0.0 0.44   0.1 perturbation  |  0.61 0.0 0.58  16/255 preturbation
    # acc1,acc2,acc3 = ensemble_prediction(model1, model2, model3, adv3,'seperate')
    # print(acc1,acc2,acc3) #0.66 0.58 0.0  8/255 preturbation | 0.58 0.39 0.0   0.1 perturbation  |  0.58 0.4 0.0  16/255 preturbation
    #
    # acc = ensemble_prediction(model1,model2,model3,adv,'ens_vote')
    # acc1 = ensemble_prediction(model1, model2, model3, adv1,'ens_vote')
    # acc2 = ensemble_prediction(model1, model2, model3, adv2,'ens_vote')
    # acc3 = ensemble_prediction(model1, model2, model3, adv3,'ens_vote')
    #
    # print(acc,acc1,acc2,acc3) # 0.48 0.42 0.58 0.44  8/255 preturbation | 0.2733333333333333 0.19 0.34 0.29   0.1 perturbation  |  0.3466666666666667 0.21 0.53 0.3  16/255 preturbation
#ens vote
    acc1 = []
    acc2 = []
    acc3 = []
    acc = []
#original
    oacc1 = []
    oacc2 = []
    oacc3 = []
    oacc = []

    aaa = []
    for i in [1/255, 2/255, 4/255, 8/255, 16/255]:
        adv1, adv2, adv3 = generate_attacked_dataset(i)
        aaa.append(adv1[0][0].cpu().numpy().transpose(1, 2, 0) * 255)
        print('{}done'.format(i))
    h1 = cv2.hconcat(aaa)
    cv2.imwrite('images.png', h1)
    bigimg = cv2.resize(h1,(5*32*4,32*4))
    cv2.imwrite('bigimages.png', bigimg)
    #
    #     acc1.append(ensemble_prediction(model1, model2, model3, adv1,'ens_vote'))
    #     acc2.append(ensemble_prediction(model1, model2, model3, adv2, 'ens_vote'))
    #     acc3.append(ensemble_prediction(model1, model2, model3, adv3, 'ens_vote'))
    #
    #     a1,a2,a3 = ensemble_prediction(model1, model2, model3, adv1, 'seperate')
    #     b1,b2,b3 = ensemble_prediction(model1, model2, model3, adv2, 'seperate')
    #     c1,c2,c3 = ensemble_prediction(model1, model2, model3, adv3, 'seperate')
    #
    #     oacc1.append((a1+a2+a3)/3)
    #     oacc2.append((b1+b2+b3)/3)
    #     oacc3.append((c1+c2+c3)/3)
    #
    # for i in range(len(acc1)):
    #     acc.append((acc1[i]+acc2[i]+acc3[i])/3)
    #     oacc.append((oacc1[i]+oacc2[i]+oacc3[i])/3)
    #
    # print(acc1)
    # print(acc2)
    # print(acc3)
    # print(acc)
    # # [0.81, 0.73, 0.57, 0.42, 0.2]
    # # [0.78, 0.73, 0.65, 0.63, 0.49]
    # # [0.65, 0.57, 0.45, 0.4, 0.35]
    # # [0.7466666666666667, 0.6766666666666666, 0.5566666666666666, 0.4833333333333334, 0.3466666666666667]
    # print(oacc1)
    # print(oacc2)
    # print(oacc3)
    # print(oacc)
    # # [0.66, 0.58, 0.43, 0.32666666666666666, 0.18666666666666668]
    # # [0.6133333333333333, 0.53, 0.48, 0.4666666666666666, 0.38000000000000006]
    # # [0.6, 0.5033333333333333, 0.42999999999999994, 0.4033333333333333, 0.3466666666666667]
    # # [0.6244444444444445, 0.5377777777777778, 0.4466666666666666, 0.39888888888888885, 0.30444444444444446]
    # plt.figure()
    # plt.plot([1/255, 2/255, 4/255, 8/255, 16/255], acc, label="Ensemble voting")
    # plt.plot([1/255, 2/255, 4/255, 8/255, 16/255], oacc, label="Undefended")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy Comparison against PGD attack")
    # plt.legend()
    # plt.savefig("Accuracy Comparison.jpg")
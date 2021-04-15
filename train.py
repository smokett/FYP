import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import random
from torchvision import transforms
from dataset import RGBDataSet, OpticalFlowDataSet
from torch.utils.data import DataLoader
from torch.nn.init import normal, constant
from basic_ops import SegmentConsensus
from RGBNet import RGBNet
from OpticalFlowNet import OpticalFlowNet

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def marginrankingloss_with_similarity(input1, input2, target, margin=1.0, reduction='mean'):
    target_is_rank = torch.where(target != 0, one, zero)
    target_is_similar = torch.where(target != 0, zero, one)
    output = 0.5*target_is_rank*((-target * (input1 - input2) + margin).clamp(min=0)) + 0.5*target_is_similar*((torch.abs(input1 - input2) - margin).clamp(min=0))
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


NUM_EPOCHS = 1
LEARNING_RATE = 0.001
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = RGBNet()
#net = OpticalFlowNet()
net = net.to(gpu)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
dataset = RGBDataSet()
#dataset = OpticalFlowDataSet()
dataLoader = DataLoader(dataset,batch_size=64,shuffle=True) 
iteration = 0
zero = torch.zeros(64).type('torch.FloatTensor').to(gpu)
one = torch.ones(64).type('torch.FloatTensor').to(gpu)


print("Start training")
for epoch in range(NUM_EPOCHS):
    for i, (data,label) in enumerate(dataLoader):

        (d1,d2) = data
        d1 = d1.type('torch.FloatTensor').to(gpu)
        d2 = d2.type('torch.FloatTensor').to(gpu)
        label = label.type('torch.FloatTensor').to(gpu)
        f1 = net(d1)
        f2 = net(d2)    

        loss = marginrankingloss_with_similarity(f1,f2,label)

        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        # f1 = f1.detach().numpy()
        # f2 = f2.detach().numpy()
        # label = label.detach().numpy()
        # right = 0.0
        # for k in range(0,np.shape(f1)[0]):
        #     if((f1[k]-f2[k] > 0 and label[k] > 0) or (f1[k]-f2[k] < 0 and label[k] < 0)):
        #         right += 1.0
        # count = float(np.shape(f1)[0])
        # acc = right / count
        # print("current accurancy: " + str(acc))
            
        iteration += 1
        print(str(iteration))
        if (iteration+1)%1500 == 0:
            LEARNING_RATE /= 10
            update_lr(optimizer, LEARNING_RATE)
    break

torch.save(net.state_dict(),"RGB.pt")
print("finished")


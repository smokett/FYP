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

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = RGBNet()
#net = OpticalFlowNet()
net = net.to(gpu)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
dataset = RGBDataSet()
val_dataset = RGBDataSet("val.txt")
#dataset = OpticalFlowDataSet()
dataLoader = DataLoader(dataset,batch_size=128,shuffle=True)
valLoader = DataLoader(val_dataset,batch_size=128,shuffle=False)
iteration = 0
zero = torch.zeros(128).type('torch.FloatTensor').to(gpu)
one = torch.ones(128).type('torch.FloatTensor').to(gpu)

losst = []
lossv = []

print("Start training")
for epoch in range(NUM_EPOCHS):
    for i, ((data,label),(data1,label1)) in enumerate(zip(dataLoader,valLoader)):
        net.train()
        (d1,d2) = data
        d1 = d1.type('torch.FloatTensor').to(gpu)
        d2 = d2.type('torch.FloatTensor').to(gpu)
        label = label.type('torch.FloatTensor').to(gpu)
        f1 = net(d1)
        f2 = net(d2)
        if(list(f1.shape)[0] < 128 or list(f2.shape)[0] < 128):
            break     

        loss = marginrankingloss_with_similarity(f1,f2,label)

        print("t " + str(loss.item()))
        losst.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            (d1,d2) = data1
            d1 = d1.type('torch.FloatTensor').to(gpu)
            d2 = d2.type('torch.FloatTensor').to(gpu)
            label1 = label1.type('torch.FloatTensor').to(gpu)
            f1 = net(d1)
            f2 = net(d2)    

            loss = marginrankingloss_with_similarity(f1,f2,label1)
            print("v " + str(loss.item()))
            lossv.append(loss.item())

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
        if iteration == 700:
            break

f = open("losst.txt",'w')
for line in losst:
    f.write(str(line)+"\n")
f.close()
f = open("lossv.txt",'w')
for line in lossv:
    f.write(str(line)+"\n")
f.close()
torch.save(net.state_dict(),"temp1.pt")
print("finished")


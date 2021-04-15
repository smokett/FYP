from RGBNet import RGBNet
from OpticalFlowNet import OpticalFlowNet
from dataset import RGBDataSet, OpticalFlowDataSet
import torch
from torch.utils.data import DataLoader
import time

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

spatial = RGBNet()
spatial.load_state_dict(torch.load("RGB_no_background_movements80%_980.pt"))
spatial.eval()
spatial = spatial.to(gpu)
temporal = OpticalFlowNet()
temporal.load_state_dict(torch.load("OpticalFlow_movements80%_980.pt"))
temporal.eval()
temporal = temporal.to(gpu)


s_dataset = RGBDataSet("valt.txt")
t_dataset = OpticalFlowDataSet("valt.txt")

s_dataLoader = DataLoader(s_dataset,batch_size=7,shuffle=False)
t_dataLoader = DataLoader(t_dataset,batch_size=7,shuffle=False) 

s_list = []
t_list = []

for i, (data,label) in enumerate(s_dataLoader):
    s_score = 0.0
    (d1,d2) = data
    d1 = d1.type('torch.FloatTensor').to(gpu)
    f1 = spatial(d1)
    f1 = f1.cpu().detach().numpy()
    for i in range(0,7):
        s_score += f1[i]
    s_list.append(s_score)
    print(str(len(s_list)))

for i, (data,label) in enumerate(t_dataLoader):
    t_score = 0.0
    (d1,d2) = data
    d1 = d1.type('torch.FloatTensor').to(gpu)
    f2 = temporal(d1)
    f2 = f2.cpu().detach().numpy()
    for i in range(0,7):
        t_score += f2[i]
    t_list.append(t_score)
    print(str(len(t_list)))


rf = open("test1_rgb.txt",'w')
of = open("test1_opticalflow.txt",'w')
for i in range(0,len(s_list)):
    rf.write(str(s_list[i])+"\n")
    of.write(str(t_list[i])+"\n")
rf.close()
of.close()

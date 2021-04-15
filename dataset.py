import torch.utils.data as data
import cv2
import torch
from torchvision import transforms
import numpy as np

class RGBDataSet(data.Dataset):
    def __init__(self,list_file="dataset.txt"):
        self.list_file = list_file
        self.pairs_list = list()
        self._read_list()
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    def _read_list(self):
        f = open(self.list_file,"r")
        for line in f:
            self.pairs_list.append(line)
        f.close()
        print("finished reading data list.")
    
    def __getitem__(self,index):
        imgs_0 = []
        imgs_1 = []
        for i in range(0,3):
            if(self.list_file=="dataset.txt"):
                path_0 = ("./"+self.pairs_list[index].split(",")[0]+"/RGB_no_background_crop/"+self.pairs_list[index].split(",")[1]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+".jpg")
                path_1 = ("./"+self.pairs_list[index].split(",")[2]+"/RGB_no_background_crop/"+self.pairs_list[index].split(",")[3]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+".jpg")
            else:
                path_0 = ("./"+self.pairs_list[index].split(",")[0]+"/RGB_no_background/"+self.pairs_list[index].split(",")[1]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+".jpg")
                path_1 = ("./"+self.pairs_list[index].split(",")[2]+"/RGB_no_background/"+self.pairs_list[index].split(",")[3]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+".jpg")                
            img = [cv2.imread(path_0)]
            imgs_0.extend(img)
            img = [cv2.imread(path_1)]
            imgs_1.extend(img)
        label = self.pairs_list[index].split(",")[4]
        label = torch.tensor(float(label))
        result_0 = np.concatenate(imgs_0, axis=2)
        tensor_0 = torch.from_numpy(result_0).permute(2, 0, 1).contiguous()
        tensor_0 = tensor_0.float().div(255)
        result_1 = np.concatenate(imgs_1, axis=2)
        tensor_1 = torch.from_numpy(result_1).permute(2, 0, 1).contiguous()
        tensor_1 = tensor_1.float().div(255)
        rep_mean = self.mean * (tensor_0.size()[0]//len(self.mean))
        rep_std = self.std * (tensor_0.size()[0]//len(self.std))
        for t, m, s in zip(tensor_0, rep_mean, rep_std):
            t.sub_(m).div_(s)
        for t, m, s in zip(tensor_1, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return [tensor_0,tensor_1], label

    def __len__(self):
        return len(self.pairs_list)

class OpticalFlowDataSet(data.Dataset):
    def __init__(self,list_file="dataset.txt"):
        self.list_file = list_file
        self.pairs_list = list()
        self._read_list()
        self.mean=[0.5]
        self.std=[np.mean([0.229, 0.224, 0.225])]
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    def _read_list(self):
        f = open(self.list_file,"r")
        for line in f:
            self.pairs_list.append(line)
        f.close()
        print("finished reading data list.")
    
    def __getitem__(self,index):
        imgs_0 = []
        imgs_1 = []
        for i in range(0,3):
            for k in range(0,5):
                path_0_x = ("./"+self.pairs_list[index].split(",")[0]+"/OpticalFlow_crop/"+self.pairs_list[index].split(",")[1]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+"_"+str(k)+"_x.jpg")
                path_0_y = ("./"+self.pairs_list[index].split(",")[0]+"/OpticalFlow_crop/"+self.pairs_list[index].split(",")[1]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+"_"+str(k)+"_y.jpg")
                path_1_x = ("./"+self.pairs_list[index].split(",")[2]+"/OpticalFlow_crop/"+self.pairs_list[index].split(",")[3]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+"_"+str(k)+"_x.jpg")
                path_1_y = ("./"+self.pairs_list[index].split(",")[2]+"/OpticalFlow_crop/"+self.pairs_list[index].split(",")[3]+"_"+str(int(self.pairs_list[index].split(",")[5])*3+i)+"_"+str(k)+"_y.jpg")
                img_0_x = cv2.imread(path_0_x,cv2.IMREAD_GRAYSCALE)
                img_0_y = cv2.imread(path_0_y,cv2.IMREAD_GRAYSCALE)
                img_1_x = cv2.imread(path_1_x,cv2.IMREAD_GRAYSCALE)
                img_1_y = cv2.imread(path_1_y,cv2.IMREAD_GRAYSCALE)
                img_0 = [img_0_x, img_0_y]
                img_1 = [img_1_x, img_1_y]
                imgs_0.extend(img_0)
                imgs_1.extend(img_1)
                

        label = self.pairs_list[index].split(",")[4]
        label = torch.tensor(float(label))
        result_0 = np.concatenate([np.expand_dims(x, 2) for x in imgs_0], axis=2)
        tensor_0 = torch.from_numpy(result_0).permute(2, 0, 1).contiguous()
        tensor_0 = tensor_0.float().div(255)
        result_1 = np.concatenate([np.expand_dims(x, 2) for x in imgs_1], axis=2)
        tensor_1 = torch.from_numpy(result_1).permute(2, 0, 1).contiguous()
        tensor_1 = tensor_1.float().div(255)
        rep_mean = self.mean * (tensor_0.size()[0]//len(self.mean))
        rep_std = self.std * (tensor_0.size()[0]//len(self.std))
        for t, m, s in zip(tensor_0, rep_mean, rep_std):
            t.sub_(m).div_(s)
        for t, m, s in zip(tensor_1, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return [tensor_0,tensor_1], label

    def __len__(self):
        return len(self.pairs_list)

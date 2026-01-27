import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale

def blazepose(x):
    '''
        Input: x (T x V x C)  
       //33 body keypoints
    {0,  "Nose"},
    {1,  "LEyeI"},
    {2,  "LEye"},
    {3,  "LEyeO"},
    {4,  "REyeI"},
    {5,  "REye"},
    {6,  "REyeO"},
    {7,  "LEar"},
    {8,  "REar"},
    {9,  "MouthL"},
    {10, "MouthR"},
    {11, "LShoulder"},
    {12, "RShoulder"},
    {13, "LElbow"},
    {14, "RElbow"},
    {15, "LWrist"},
    {16, "RWrist"},
    {17, "LPinky"}, 
    {18, "RPinky"}, 
    {19, "LIndex"}, 
    {20, "RIndex"},
    {21, "LThumb"}, 
    {22, "RThumb"},
    {23, "LHip"},
    {24, "RHip"},
    {25, "LKnee"},
    {26, "Rknee"},
    {27, "LAnkle"},
    {28, "RAnkle"},
    {29, "LHeel"},
    {30, "RHeel"},
    {31, "LFoot"},
    {32, "RFoot"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = (x[:,23,:] + x[:,24,:]) * 0.5  # Hip
    y[:,1,:] = x[:,24,:]
    y[:,2,:] = x[:,26,:]
    y[:,3,:] = x[:,28,:]
    y[:,4,:] = x[:,23,:]
    y[:,5,:] = x[:,25,:]
    y[:,6,:] = x[:,27,:]
    y[:,7,:] = (((x[:,9,:] + x[:,10,:]) * 0.5) + ((x[:,23,:] + x[:,24,:]) * 0.5)) * 0.5  # Spine (Neck + Hip)
    y[:,8,:] = (x[:,9,:] + x[:,10,:]) * 0.5  # Neck
    y[:,9,:] = x[:,0,:] 
    y[:,10,:] = (x[:,2,:] + x[:,5,:]) * 0.5  # Head
    y[:,11,:] = x[:,11,:]
    y[:,12,:] = x[:,13,:]
    y[:,13,:] = x[:,15,:]
    y[:,14,:] = x[:,12,:]
    y[:,15,:] = x[:,14,:]
    y[:,16,:] = x[:,16,:]
    return y
    
def read_input(npy_path, scale_range):
    kpts_all = np.load(npy_path)
    kpts_all = blazepose(kpts_all)
    if scale_range:
        motion = crop_scale(kpts_all, scale_range) 
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, npy_path, clip_len=243, scale_range=None):
        self.npy_path = npy_path
        self.clip_len = clip_len
        self.npy_all = read_input(npy_path, scale_range)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.npy_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.npy_all))
        return self.npy_all[st:end]
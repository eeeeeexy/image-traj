import sys
import os.path as osp
import time
from PIL import Image
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch


def load_data(pixel_size):

    # filename = '/home/xieyuan/Transportation-mode/Traj2Image/Geolife/trips_traj2image_1s_trip20_shift_4class_pixelsize%d_normalize_acc+speed.pickle'%pixel_size
    filename = '/home/xieyuan/Transportation-mode/Traj2Image/Geolife/trips_traj2image_1s_trip20_shift_4class_pixelsize%d_normalize_acc+speed+time.pickle'%pixel_size
    

    with open(filename, 'rb') as f:
        kfold_dataset = pickle.load(f)
    dataset = kfold_dataset

    
    # print('dataset:', dataset)
        
    train_x_geolife = np.array(dataset[0])# [141,248,4]
    train_y_geolife = np.array(dataset[1])

    print(train_x_geolife.shape)
    print(train_y_geolife.shape)

    test_X_geolife = np.array(dataset[2])
    test_Y_geolife = np.array(dataset[3])

    print(test_X_geolife.shape)
    print(test_Y_geolife.shape)

    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x_geolife).to(torch.float),
        torch.from_numpy(train_y_geolife)
    )

    test_dataset_geolife = TensorDataset(
        torch.from_numpy(test_X_geolife).to(torch.float),
        torch.from_numpy(test_Y_geolife),
    )



    return train_dataset_geolife, test_dataset_geolife, train_y_geolife
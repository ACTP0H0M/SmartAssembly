#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import cv2
import numpy as np
import glob
import pandas as pd
import random

import rospy # only used for printing warning

from vision.train_feature_extractor.train_pin_detection import CroppedDataset
from vision.train_feature_extractor.train_pin_detection import train_frozen, train_custom_cnn, train_not_frozen


if __name__ == "__main__":
    """ This script train a frozen resnet 18 model """
    random.seed(10)
    data_transforms = {
        'train': transforms.Compose([ #when resize adapt the label!
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #date_time = ""
    #if date_time == "":
    #    print("date_time is empty")
    #    exit(0)
    data_dir = 'dataset_feature_extractor'
    # data_dir = 'test_dataset_featureextractor'



    # id_list was not used because there were more pngs than yaml (contains labels)
    # for each yaml take same name png

    #id_list = [f for f in glob.glob(data_dir+'/**/*.png', recursive=True)]

    #id_list = [f for f in id_list if "cam_1" in f]
    # print("id_list", id_list)

    #print("id_list", id_list)
    labels_list_names = [f for f in glob.glob(data_dir+'/**/*.yaml', recursive=True)]
    labels_list_names = [f for f in labels_list_names if "cam_2" in f]
    # labels_list_names = [f for f in labels_list_names if "sess_33" not in f]
    labels_list = []
    for f in labels_list_names:
        if os.stat(f).st_size == 0:
            print("file "+f+" is empty")
        else:
            df = pd.read_csv(f, header=None, sep=": ")
            df.iloc[:,0] = f.split("/**")[-1][:-5]
            labels_list.append(df)
    labels_list = pd.concat(labels_list)
    # print('label list:', np.unique(labels_list.iloc[:,1],return_counts=True))
    labels_list = dict(zip(labels_list.iloc[:,0], labels_list.iloc[:,1]))
    # TODO check if all ids are in labels_list
    keys_labels = list(labels_list.keys()) # TODO change code here! also in cropped_dataset because keys unknown
    l = int( len(keys_labels) * 0.8 )
    random.shuffle(keys_labels)
    train_data = keys_labels[:l] # traindata contain l items of labels_list
    val_data = keys_labels[l:]
    val_labels = []
    for k in val_data:
        val_labels.append(labels_list[k])
    # print(np.unique(val_labels, return_counts=True))
    if np.unique(val_labels).shape[0] < 3:
        rospy.logerr("Missing classes in validation set")

    data_dir = ''
    image_datasets = {'train': CroppedDataset(train_data, labels_list, data_dir),
                      'val': CroppedDataset(val_data, labels_list, data_dir)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #class_names = image_datasets['train'].classes

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    inputs, classes = next(iter(dataloaders['train'])) # Get a batch of training data
    # print(inputs.shape)
    #out = torchvision.utils.make_grid(inputs) # Make a grid from batch
    #imshow(out, title="show grid")


    #visualize_model(model_ft, dataloaders)


    train_not_frozen(dataloaders, dataset_sizes)
    # train_custom_cnn(dataloaders, dataset_sizes)

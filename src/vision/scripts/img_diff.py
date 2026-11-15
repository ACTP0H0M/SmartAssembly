#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch as th
import torch.nn as nn
def applPooling(img):
    print("apply pooling")
    print(img.shape)
    torch_img = th.Tensor(img).unsqueeze(0)
    print(torch_img.shape)
    modulePooling = nn.MaxPool2d(4)
    processed_img = modulePooling(torch_img)
    return processed_img.squeeze().numpy()

def applConv(img):
    moduleConv = nn.Sequential(nn.MaxPool2d(4),
                  nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
                  nn.ReLU(),
                  nn.Flatten(start_dim=1))
    print("apply conv")
    print(img.shape)
    torch_img = th.Tensor(img).unsqueeze(0).unsqueeze(0)
    print(torch_img.shape)
    processed_img = moduleConv(torch_img).reshape((8,8))
    print(processed_img.shape)
    return processed_img.detach().squeeze().numpy()

if __name__ == "__main__":
    path = "/home/ips/Desktop/"
    filename1 = 'cropped_img2_003_no_pin.png'
    filename2 = 'cropped_img2_002.png'

    #self.img.append(cv2.cvtColor(self.image_to_numpy(args[x]), cv2.COLOR_BGR2RGB))


    img1 = cv2.imread(path + filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + filename2, cv2.IMREAD_GRAYSCALE)
    print('Image Dimensions :', img1.shape, np.max(np.array(img1)), np.max(np.array(img2)))
    diff = img1.astype("int16") - img2.astype("int16")
    print(np.min(diff), np.max(diff))
    d = np.array(diff)
    #d = applPooling(d)
    d = applConv(d)
    print(np.unique(d,return_counts=True))
    plt.imshow(d)
    plt.title('Image')
    plt.show()


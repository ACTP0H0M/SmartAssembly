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

def scale_width(x, width = 1280.0):
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return width
    return x * width

def normalize_scale_width(x, width = 1280.0):
    return x / width

def scale_height(y, height = 720.0):
    if y < 0.0:
        return 0.0
    elif y > 1.0:
        return height
    return y * height

def normalize_scale_height(y, height = 720.0):
    return y / height

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and running_loss < best_loss:  # deep copy the best epoch's model
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            print("outputs:", outputs)


            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {outputs[j]}')  # draw red point on regressed points
                ax.plot(scale_width(outputs[j][0]), scale_height(outputs[j][1]), marker=',', color="red")  # TODO plot in subplots
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'


    def __init__(self, list_IDs, labels, img_dir):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.img_dir = img_dir
        self.tensor_module = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        print("len", len(self.list_IDs))
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # TODO delete last line of labeling
        # Select sample
        print("getitem")
        ID = self.list_IDs[index]
        print("ID",ID)
        # Load data and get label
        print("img_dir",self.img_dir)
        img_path = os.path.join(self.img_dir, ID.split("img")[0], ID + ".png")
        print("img_path",img_path)
        img = cv2.imread(img_path)
        if os.stat(img_path).st_size == 0:
            print("file "+img_path+" is empty")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # TODO /255
        X = self.tensor_module(img)
        y = list(eval(self.labels[ID]))  # convert str to tuple to list
        y = [normalize_scale_width(y[0]),normalize_scale_height(y[1])]
        y = torch.tensor(y,dtype=torch.float32)
        print("y", y)
        return X, y

if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([ # when resize adapt the label!
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #date_time = ""
    #if date_time == "":
    #    print("date_time is empty")
    #    exit(0)
    data_dir = 'C:\\Users\\boiar\\Documents\\Data\\SmartAssembly\\imgs_training'




    id_list = [f for f in glob.glob(data_dir+'\\**\\*.png', recursive=True)]
    print("id_list", id_list)
    labels_list_names = [f for f in glob.glob(data_dir+'\\**\\*.txt', recursive=True)]
    labels_list = []
    for f in labels_list_names:
        if os.stat(f).st_size == 0:
            print("file "+f+" is empty")
        else:
            labels_list.append(pd.read_csv(f, header=None, sep=";"))
    labels_list = pd.concat(labels_list)
    labels_list = dict(zip(labels_list.iloc[:,0], labels_list.iloc[:,1]))
    print("labels_list",labels_list)
    # TODO check if all ids are in labels_list
    keys_labels = list(labels_list.keys())
    l = int( len(keys_labels) * 0.8 )
    random.shuffle(keys_labels)
    train_data = keys_labels[:l]
    val_data = keys_labels[l:]
    image_datasets = {'train': Dataset(train_data, labels_list, data_dir),
                      'val': Dataset(val_data, labels_list, data_dir)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    inputs, classes = next(iter(dataloaders['train'])) # Get a batch of training data
    print(inputs.shape)
    #out = torchvision.utils.make_grid(inputs) # Make a grid from batch
    #imshow(out, title="show grid")





    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #
    def train_not_frozen():
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)
        criterion = nn.MSELoss()
        # Observe that all parameters are being optimized!!!!!!!!!
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=2)

        ######################################################################
        #

        visualize_model(model_ft)



    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad = False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    def train_frozen():
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False # freeze weights

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        model_conv = model_conv.to(device)
        criterion = nn.MSELoss()
        # Observe that only parameters of final layer are being optimized as opposed to before.
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs


        ######################################################################
        # Train and evaluate
        # ^^^^^^^^^^^^^^^^^^
        #
        # On CPU this will take about half the time compared to previous scenario.
        # This is expected as gradients don't need to be computed for most of the
        # network. However, forward does need to be computed.
        #

        model_conv = train_model(model_conv, criterion, optimizer_conv,
                                 exp_lr_scheduler, num_epochs=2)

        ######################################################################
        #

        visualize_model(model_conv)
        plt.show()
    train_frozen()
## importing data related libraries
import h5py
import numpy as np
from scipy import stats
import pandas as pd
import random
import sys 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize
import numpy as np


# model related packages
import torch.nn as nn
import torch
from torch.optim import Adam
from sklearn.metrics import f1_score , precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.autograd import Variable
import pickle 
from torch.optim import lr_scheduler

from model_multi import UTime


# visulization packages
from seaborn import heatmap
import matplotlib.pyplot as plt


n_epochs = 5
num_channels = 8


## preprocessing :  normalizing across channels using the l2 norm
def normalize_channels(X):
    for i in range(8):
        X[:,9000*i:9000*(i+1)] = normalize(X[:,9000*i:9000*(i+1)],norm='l2')
    return X


## making dataloader for training, testing and validation
def make_loader(X,y,batch_size=8,shuffle=True):
  X = torch.Tensor(X).reshape((X.shape[0],num_channels,-1))
  X = torch.Tensor(X)
  print(X.shape)
  print(y.shape)
  y = torch.Tensor(y).unsqueeze(1)
  data = TensorDataset(X,y)
  dataloader = DataLoader(data,batch_size=batch_size,shuffle=shuffle)
  return dataloader


## Training for one epoch
def train(epoch):
  utime.train()
  losses=[]
  
  with tqdm(train_loader,unit="batch") as tepoch:
    for data,target in tepoch:
        optimizer.zero_grad()
        tepoch.set_description(f'epoch {epoch}')
        output = utime(data.to(device))
        # loss = loss_fct(output,target.long().to(device),binary=True)
        loss = loss_fct(output,target.to(device))
        # loss = loss_fct(output,target.long().to(device))
        # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
        loss.backward()
        # losses.append(loss.item())
        optimizer.step()
        tepoch.set_postfix(loss = loss.item())
        del data
        del target


## Evaluation of the model 
def eval():
    utime.eval()
    losses=[]
    out = []
    y_target = []

    with tqdm(val_loader) as tepoch:
        for data,target in tepoch:
            tepoch.set_description('evaluation')
            output = utime(data.to(device))
            # loss = loss_fct(output,target.long().to(device),binary=True)
            loss = loss_fct(output.view((-1,90,1)),target.to(device).view((-1,90,1)))
            # print(output.shape,target.shape)
            # loss = loss_fct(output,target.long().to(device))
            # loss = loss_fct(output.unsqueeze(2),target.long().to(device))
            losses.append(loss.item())
            out.append(output)
            
            y_target.append(target.long().numpy())
            tepoch.set_postfix(loss = np.average(losses))

            del loss
            del data
            del target
    
        out = torch.cat(out,dim=0).squeeze(1)
        out = 1*(out.cpu().detach().numpy()>=0)
        tepoch.set_postfix(loss=np.average(losses),f1_score =f1_score(y_val.flatten(),out.flatten(),average='binary'))
        return out,y_target
        



if __name__ == "__main__":

    PATH_TO_TRAINING_DATA = "/content/drive/MyDrive/dreem_files/X_train.h5" ## the training dataset is 2Gb so I couldn't include it in the repository
    PATH_TO_TRAINING_TARGET = "/content/drive/MyDrive/UTime-pytorch/y_train.csv"
    h5_file = h5py.File(PATH_TO_TRAINING_DATA,'r')
    
    X = h5_file['data'][:,2:]
    y = pd.read_csv(PATH_TO_TRAINING_TARGET,index_col=0).to_numpy()

    # normalizing across channels
    for i in range(8):
        X[:,9000*i:9000*(i+1)] = normalize(X[:,9000*i:9000*(i+1)],norm='l2')

    # splitting into train and validation sets
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.05)

    # creating data loaders
    train_loader = make_loader(X_train,y_train)
    val_loader = make_loader(X_val,y_val)

    # loading the model
    utime = UTime()
    device = torch.device('cuda')
    utime.to(device)
    
    # setting the optimizer and the scheduler
    optimizer=Adam(utime.parameters(),lr=1e-2,)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, min_lr=1e-8) 

    # setting the loss function
    loss_fct = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([2]).cuda())

    # first evaluation to benchmark the performance of the model before any training
    eval()

    # training and outputting the performance metrics on the validation set
    for epoch in range(1,n_epochs+1):
        train(epoch)
        out,y_target = eval()
        y_target = np.concatenate(y_target).squeeze(axis=1)
        print('precision {0}, recall {1}, f1 score {2}'.format(*precision_recall_fscore_support(out.flatten(),y_target.flatten(),average='binary')))
        scheduler.step(f1_score(out.flatten(),y_target.flatten()))
        


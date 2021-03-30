

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable


num_channels = 8 ## number of channels / signal types
num_classes = 2 ## number of classes to predict


## Defining the encoder block 
class EncoderLayer(nn.Module):

  def __init__(self,maxpool=10,cf=1):
    super(EncoderLayer,self).__init__()
    self.conv1 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf*2,kernel_size=5,padding=2)
    self.batch_norm1 = nn.BatchNorm1d(num_channels*cf*2)
    self.conv2 = nn.Conv1d(in_channels=num_channels*cf*2,out_channels=num_channels*cf*2,kernel_size=5,padding=2)
    self.batch_norm2 = nn.BatchNorm1d(num_channels*cf*2)
    self.maxpool = nn.MaxPool1d(maxpool)
    self.relu = nn.ReLU()
  
  def forward(self,input):
    output = self.conv1(input)
    output = self.relu(self.batch_norm1(output))
    output = self.conv2(output)
    output = self.relu(self.batch_norm2(output))
    output_features = output
    output = self.maxpool(output)
    return output,output_features



## defining the decoder block
class DecoderLayer(nn.Module):

  def __init__(self,kernel_size=10,cf=16):
    super(DecoderLayer,self).__init__()
    self.upsample = nn.Upsample(scale_factor=kernel_size)
    self.conv3 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*(cf//2),kernel_size=kernel_size,padding=(kernel_size-1)//2)
    self.batch_norm3 = nn.BatchNorm1d(num_channels*(cf//2))
    self.conv4 = nn.Conv1d(in_channels=num_channels*(cf//2),out_channels=num_channels*(cf//4),kernel_size=kernel_size,padding=(kernel_size+1)//2)
    self.batch_norm4 = nn.BatchNorm1d(num_channels*(cf//4))
    self.relu = nn.ReLU()

  def forward(self,input,encoder_output):
    
    output = self.upsample(input)
    diff = encoder_output.shape[2] - output.shape[2]
    output = nn.functional.pad(output,(diff//2,diff//2))

    output = torch.cat((output,encoder_output),dim=1)

    output = self.conv3(output)
    output = self.relu(self.batch_norm3(output))
    output = self.conv4(output)
    output = self.relu(self.batch_norm4(output))

    return output


## defining the bridge, which takes the output of the last encoder block as input, and its output is used as the input of the first decoder block
class Bridge(nn.Module):

    def __init__(self,cf=16):
        super(Bridge,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf,kernel_size=5,padding=2)
        self.batch_norm1 = nn.BatchNorm1d(num_channels*cf)
        self.conv2 = nn.Conv1d(in_channels=num_channels*cf,out_channels=num_channels*cf,kernel_size=5,padding=2)
        self.batch_norm2 = nn.BatchNorm1d(num_channels*cf)
        self.relu = nn.ReLU()
    
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu(self.batch_norm1(output))
        output = self.conv2(output)
        output = self.relu(self.batch_norm2(output))
        
        return output



class UTime(nn.Module):
  
  def __init__(self):
    super(UTime,self).__init__()

    ## the four encoder blocks
    self.encoder1 = EncoderLayer(10,cf=1)
    self.encoder2 = EncoderLayer(8,cf=2)
    self.encoder3 = EncoderLayer(6,cf=4)
    self.encoder4 = EncoderLayer(4,cf=8)

    ## the four decoder blocks
    self.decoder1 = DecoderLayer(10,cf=4)
    self.decoder2 = DecoderLayer(8,cf=8)
    self.decoder3 = DecoderLayer(6,cf=16)
    self.decoder4 = DecoderLayer(4,cf=32)

    ## the bridge connecting the encoder and the decoder
    self.bridge = Bridge()

    ## the segment classifier
    self.avg_pool = nn.AvgPool1d(kernel_size=100,stride=100)
    self.conv3 = nn.Conv1d(in_channels=num_channels,out_channels=num_classes-1,kernel_size=1)
    
    ## the activation function used multiple times
    self.relu = nn.ReLU()

  def forward(self,input):

    output,output_features1 = self.encoder1(input)
    output,output_features2 = self.encoder2(output)
    output,output_features3 = self.encoder3(output)
    output,output_features4 = self.encoder4(output)

    output = self.bridge(output)

    output = self.decoder4(output,output_features4)
    output = self.decoder3(output,output_features3)
    output = self.decoder2(output,output_features2)
    output = self.decoder1(output,output_features1)

    output = self.avg_pool(output)
    output = self.conv3(output)

    return output

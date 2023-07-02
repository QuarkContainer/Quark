import multiprocessing as mp
import time
import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
import csv

import asyncio
import json
import hashlib
import sys

import qserverless

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import torch.utils.data as tdata
from torch.nn.functional import nll_loss, cross_entropy

from torch import optim
from datetime import datetime
import shutil

import warnings
warnings.filterwarnings("ignore")

#experimental setting
experiment_name ="convNet"
path = os.getcwd()

batch_size = 256

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def save_model(model, i):
    storage_key = f'model_{i}'
    filename = f"{path}/tmp/{storage_key}.pt"
    torch.save(model.state_dict(), filename)
    in_file = open(filename, "rb") # opening for [r]eading as [b]inary
    data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
    return data

def load_model(data, i):
    storage_key = f'model_{i}'
    filename = f"{path}/tmp/{storage_key}.pt"
    out_file = open(filename, "wb") # open for [w]riting as [b]inary
    out_file.write(data)
    out_file.close()
    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load(f"{path}/tmp/{storage_key}.pt"))
        
    return loaded_model


async def train(context, blob, state, device, epoch, i, parallelism, batch_size):
    """Loop used to train the network"""
    torch.manual_seed(42) 
    print("train 1", state);

    trainStart = datetime.now()
    model = ConvNet()
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    # create optimizer
    if epoch > 0:
        # load the global averaged model
        (data, err) = await context.BlobReadAll(blob)
        model = load_model(data, 'average')
        (statedata, err) = await context.BlobReadAll(state)
 
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    if epoch > 0:
        load_state(statedata, optimizer, i)

    criterion = nn.CrossEntropyLoss().to(device)
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=parallelism, rank=i)
    print("train 2");
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    
    print("train 3");
    model.train()
    loss, tot = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)


        loss = criterion(output, target)
        tot += loss.item()
        

        loss.backward()
        optimizer.step()
        

        if batch_idx % 30 == 0:
            print('Process: {}, Device: {} Train Epoch: {} Step: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, device, epoch, batch_idx, len(train_loader.batch_sampler),
                   100. * batch_idx / len(train_loader), loss.item()))
            
    state_data = save_state(optimizer, i)
    model_data = save_model(model, str(i))

    return (state_data, model_data)

async def trainrunner(context, blob, state, epoch, i, parallelism):
    device = "cpu" 
    print("trainrunner ....1")
    (state_data, model_data) = await train(context, blob, state, device, epoch, i, parallelism, batch_size)
    print("trainrunner ....2")
    blobs = context.NewBlobAddrVec(2)
    (addr, err) = await context.BlobWriteAll(blobs[0], state_data)
    if err is not None :
        return (None, err)
    blobs[0] = addr
    
    (addr, err) = await context.BlobWriteAll(blobs[1], model_data)
    if err is not None :
        return (None, err)
    blobs[1] = addr
    return (json.dumps(blobs), None)
    
def save_state(optimizer, i):
    isExist = os.path.exists(f'{path}/tmp/')
    if not isExist:
        os.makedirs(f'{path}/tmp/') 
    filename = f'{path}/tmp/optimizer_state_{i}.pkl'
    with open(filename, 'wb') as f:
        #print(optimizer.state_dict()['state'])
        pickle.dump(optimizer.state_dict(), f)
    in_file = open(filename, "rb") # opening for [r]eading as [b]inary
    data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
    return data
        
    #print('saving optimizer state done, time is: ', datetime.now() - start)
def load_state(data, optimizer, i):
    filename = f'{path}/tmp/optimizer_state_{i}.pkl'
    out_file = open(filename, "wb") # open for [w]riting as [b]inary
    out_file.write(data)
    out_file.close()
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            optimizer.load_state_dict(state)
            #update_state(optimizer, state)
        #print('loading optimizer state done, time is: ', datetime.now() - start)
    else:
        print('no state found')

async def model_weight_average_runner(context, blobs: qserverless.BlobAddrVec, parallelism):
    model = ConvNet()

    print("model_weight_average_runner 1");
    cur_model = ConvNet()

    sd_avg = model.state_dict()

    beta = 1.0/parallelism 
    for i in range(parallelism):
        (data, err) = await context.BlobReadAll(blobs[i])
        if err is not None :
            print("model_weight_average_runner 2");
            return (None, err)
        cur_model = load_model(data, i)
        for key in cur_model.state_dict():
            if i == 0:
                sd_avg[key] = (cur_model.state_dict()[key]) / parallelism
            else:
                sd_avg[key] += (cur_model.state_dict()[key]) / parallelism
    print("model_weight_average_runner 3");
    model.load_state_dict(sd_avg)
    model_data = save_model(model, 'average')
    blob = context.NewBlobAddr()
    (addr, err) = await context.BlobWriteAll(blob, model_data)
    print("model_weight_average_runner 4");
    return (json.dumps(addr), err)

async def handwritingClassification(context):
    epochs = 4
    parallelism = 2
    blob = None
    states = []
    for i in range(0, parallelism):  
        states.append(None)
    
    for epoch in range(epochs):
        results = await asyncio.gather(
                    *[context.RemoteCall(
                        packageName = "pypackage2",
                        funcName = "trainrunner",
                        blob = blob,
                        state = states[i],
                        epoch = epoch,
                        i = i,
                        parallelism = parallelism
                    ) for i in range(0, parallelism)]
                )
        blobMatrix = list();
        for res, err in results:
            if err is not None:
                return (None, qserverless.QErr(err))
            blobVec = json.loads(res)
            blobMatrix.append(blobVec)
        
        shuffBlobs = qserverless.TransposeBlobMatrix(blobMatrix)
        states = shuffBlobs[0]
        (res, err) = await context.RemoteCall(
                packageName = "pypackage2",
                funcName = "model_weight_average_runner",
                blobs = shuffBlobs[1],
                parallelism = parallelism
        )
        if err is not None:
            return (None, qserverless.QErr(err))
        blob = json.loads(res)
        print("success ", epoch)
    
    (res, err) = await context.RemoteCall(
                packageName = "pypackage2",
                funcName = "validaterunner",
                blob = blob,
                batch_size = batch_size
    )
  
    print("finish.....")
    
    return (res, None)


def validate(model, device, batch_size) -> (float, float):
    """Loop used to validate the network"""

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    valset = datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    val_loader= torch.utils.data.DataLoader(valset, batch_size=batch_size)

    criterion =nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            #print(data.dtype, type(target))
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_loss += cross_entropy(output, target).item()  # sum up batch loss
            correct += predicted.eq(target).sum().item()

    test_loss /= len(val_loader)

    accuracy = 100. * correct / len(val_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return accuracy, test_loss

async def validaterunner(context, blob, batch_size):
    device = "cpu"
    (data, err) = await context.BlobReadAll(blob)
    
    model = ConvNet()
    model = load_model(data, 'average')
    (accu, loss) = validate(model, device, batch_size)
    print("accu is ", accu, "loss is ", loss)
    return ("accus is {} loss is {}".format(accu, loss), None)
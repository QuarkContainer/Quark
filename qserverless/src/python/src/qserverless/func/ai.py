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
    filename = f"{path}/tmp/{storage_key}.pt"
    out_file = open(filename, "wb") # open for [w]riting as [b]inary
    out_file.write(data)
    out_file.close()
    loaded_model = ConvNet()
    storage_key = f'model_{i}'
    loaded_model.load_state_dict(torch.load(f"{path}/tmp/{storage_key}.pt"))
        
    return loaded_model


async def train(context, blob, device, epoch, i,  args):
    """Loop used to train the network"""
    torch.manual_seed(42) 

    trainStart = datetime.now()
    model = ConvNet()
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    # create optimizer
    if epoch > 0:
        # load the global averaged model
        (data, err) = await context.BlobReadAll(blob)
        model = load_model(data, 'average')
 
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    if epoch > 0:
        load_state(optimizer, i)

    criterion = nn.CrossEntropyLoss().to(device)
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=args.parallelism, rank=i)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    
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

async def trainrunner(context, blob, epoch, i):
    device = "cpu" 
    print("training ....")
    #assignProcssToCPU()
    (state_data, model_data) = await train(context, blob, device, epoch, i, 
        batch_size = 256,
        learning_rate = 0.1,
        parallelism = 4,
        epochs = 5,
    )
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

    cur_model = ConvNet()

    sd_avg = model.state_dict()

    beta = 1.0/parallelism 
    for i in range(parallelism):
        (data, err) = await context.BlobReadAll(blobs[i])
        if err is not None :
            return (None, err)
        cur_model = load_model(data, i)
        for key in cur_model.state_dict():
            if i == 0:
                sd_avg[key] = (cur_model.state_dict()[key]) / parallelism
            else:
                sd_avg[key] += (cur_model.state_dict()[key]) / parallelism
    model.load_state_dict(sd_avg)
    model_data = save_model(model, 'average')
    blob = context.NewBlobAddr()
    (addr, err) = await context.BlobWriteAll(blob, model_data)
    if err is not None :
        return (None, err)
    return (addr, err)

async def handwritingClassification(context):
    epochs = 2
    parallelism = 2
    blob = None
    print("handwritingClassification 1")
    for epoch in range(epochs):
        print("handwritingClassification 2 ", epoch)
        results = await asyncio.gather(
                    *[context.RemoteCall(
                        packageName = "pypackage1",
                        funcName = "trainrunner",
                        blob = blob,
                        epoch = epoch,
                        i = i
                    ) for i in range(0, parallelism)]
                )
        print("handwritingClassification 3 ", epoch, results)
        blobMatrix = list();
        for res, err in results:
            if err is not None:
                return (None, qserverless.QErr(err))
            blobVec = json.loads(res)
            blobMatrix.append(blobVec)
        
        shuffBlobs = qserverless.TransposeBlobMatrix(blobMatrix)
        (res, err) = context.RemoteCall(
                packageName = "pypackage1",
                funcName = "model_weight_average_runner",
                blobs = blobMatrix[1],
                parallelism = parallelism
        )
        if err is not None:
            return (None, qserverless.QErr(err))
        blob = json.loads(res)
        print("success ", epoch)
        
    print("finish.....")
    
    return ("sucess", None)

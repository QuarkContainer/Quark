import multiprocessing as mp
import time
import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import io
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
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    data = buffer.read()
    return data

def load_model(data, i):
    buffer = io.BytesIO()
    buffer.write(data)
    buffer.seek(0)
    loaded_model = ConvNet()   
    loaded_model.load_state_dict(torch.load(buffer))
    return loaded_model

async def trainer(context, blob, state, epoch, i, parallelism):
    device = "cpu" 
    """Loop used to train the network"""
    torch.manual_seed(42) 
    print("iternate_trainer 1", state);

    trainStart = datetime.now()
    model = ConvNet()
    
    data = await context.BlobReadAll(blob)
    model = load_model(data, 'average')
    statedata = await context.BlobReadAll(state)
 
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
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
    
    print("trainrunner ....2")
    blobs = list()
    
    state_blob = await context.BlobNew(state_data)
    blobs.append(state_blob)
    
    model_blob = await context.BlobNew(model_data)
    blobs.append(model_blob)
    
    return json.dumps(blobs)

def save_state(optimizer, i):
    data = pickle.dumps(optimizer.state_dict())
    return data
        
    #print('saving optimizer state done, time is: ', datetime.now() - start)
def load_state(data, optimizer, i):
    state = pickle.loads(data)
    optimizer.load_state_dict(state)

async def model_weight_average_runner(context, blobs: qserverless.BlobAddrVec, parallelism):
    model = ConvNet()

    print("model_weight_average_runner 1");
    cur_model = ConvNet()

    sd_avg = model.state_dict()

    beta = 1.0/parallelism 
    for i in range(parallelism):
        data = await context.BlobReadAll(blobs[i])
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
    addr = await context.BlobWriteAll(blob, model_data)
    print("model_weight_average_runner 4");
    return json.dumps(addr)

async def handwritingClassification(context):
    epochs = 2
    parallelism = 2
    
    model = ConvNet()
    model_data = save_model(model, 0)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    state_data = save_state(optimizer, 0)
    
    
    model_blob = await context.BlobNew(model_data)
    state_blob = await context.BlobNew(state_data)
    
    blob = model_blob
    
    states = []
    for i in range(0, parallelism):  
        states.append(state_blob)
    
    for epoch in range(epochs):
        results = await asyncio.gather(
                    *[context.CallFunc(
                        packageName = "pypackage1",
                        funcName = "trainer",
                        blob = blob,
                        state = states[i],
                        epoch = epoch,
                        i = i,
                        parallelism = parallelism
                    ) for i in range(0, parallelism)]
                )
        blobMatrix = list();
        for res in results:
            blobVec = json.loads(res)
            blobMatrix.append(blobVec)
        
        shuffBlobs = qserverless.TransposeBlobMatrix(blobMatrix)
        states = shuffBlobs[0]
        res = await context.CallFunc(
                packageName = "pypackage1",
                funcName = "model_weight_average_runner",
                blobs = shuffBlobs[1],
                parallelism = parallelism
        )
        blob = json.loads(res)
        print("success ", epoch)
    
    res = await context.CallFunc(
                packageName = "pypackage1",
                funcName = "validaterunner",
                blob = blob,
                batch_size = batch_size
    )
  
    print("finish.....")
    
    return res


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
    data = await context.BlobReadAll(blob)
    
    model = ConvNet()
    model = load_model(data, 'average')
    (accu, loss) = validate(model, device, batch_size)
    print("accu is ", accu, "loss is ", loss)
    return "accus is {} loss is {}".format(accu, loss)
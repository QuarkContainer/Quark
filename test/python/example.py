
import torch
import torch.nn as nn
import torch.nn.functional as F
import QActor
from http.server import BaseHTTPRequestHandler, HTTPServer


class ModelActor:
        # Constructor
    def __init__(self, name):
        self.name = name
    
    def deploy(self, devId):
        ...

    def send(self, target, *arguments, **kwarguments):
        ...

# compute V100 == 10000
@qactor(type = "xpu", devMemory = "6GB", memory = "1GB", compute = "1200")
class a1(nn.Model, ModelActor):
    def __init__(self, name):
        super(a1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def step(self, reqid, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        super.send("a2.step", reqid, x)
    
    def deploy(self, device):
        super.to(device)

@qactor(type = "xpu", devMemory = "2GB", memory = "1GB", compute = "1200")
class a2(nn.Model, ModelActor):
    def __init__(self, name):
        super(a2, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def step(self, reqid, x):
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        super.send("gateway.resp", reqid, x)

    def deploy(self, device):
        super.to(device)

@qactor(type = "cpu", memory = "10GB", compute = "1200")
class Gateway(ModelActor, HTTPServer):
    def __init__(self, name):
        super(Gateway, self).__init__()
        nextReqId = 0
    
    def nextId():
        nextReqId += 1
        return nextReqId

    def get(self, httpGet):
        reqId = self.nextId()
        self.requests[reqId, httpGet]
        prompt = httpGet.Req;
        super.send("f1.step", reqId, prompt)

    def resp(self, reqId, x):
        httpReq = self.requests[reqId]
        del(self.requests, reqId)
        httpReq.response(x)

if __name__ == "__main__":
    system = QActorSystem()
    system.new_actor("gateway", Gateway)
    system.new_actor("a1", a1)
    system.new_actor("a2", a1)
    system.httpbind("gateway", "0.0.0.0", 8080)

    system.deployment()


@qactor(type = "xpu", devMemory = "2GB", memory = "1GB", compute = "1200")
class BaseModel(nn.Model, ModelActor):
    def __init__(self, name):
        super(a2, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        loraReady = False
        baseReady = False


    def step(self, reqid, x):
        super.send("LoraModel.step", reqid, x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.loraReady:
            m = Merge(x, self.loraOutput)
            super.Send("NextBase.step", reqid, m)
        else:
            self.baseOutput = x
            self.baseReady = True
    
    def loraReady(self, reqid, x):
        if self.baseReady:
            m = Merge(x, self.baseOutput)
            super.Send("NextBase.step", reqid, m)
        else:
            self.loraReady = x
            self.loraOutput = True
        

def Merge(x, y):
    return x+y

@qactor(type = "xpu", devMemory = "2GB", memory = "1GB", compute = "1200")
class LoraModel(nn.Model, ModelActor):
    def __init__(self, name):
        super(a2, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def step(self, reqid, x):
        super.send("lora.step", reqid, x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        super.send("BaseModel.step", reqid, x)
#from client import RemoteCall

import json
import common
from common import BlobAddr 

async def wordcount(context, parameters) -> str:
    filenames = json.loads(parameters)
    res = ""
    for filename in filenames:
        (res, err) = await context.RemoteCall(
            packageName= "",
            funcName= "map",
            parameters= filename,
            priority= 1
        )
    return res

async def map(context, parameters) -> str:
    filename = parameters
    word_counts = dict()
    with open(filename,'r') as file:
        contents = file.read()
        words = contents.split()
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1 
    res = json.dumps(word_counts)
    print(res)
    return res

async def add(context, parameters):
    (res, err) = await context.RemoteCall(
        packageName= "",
        funcName= "sub",
        parameters= "call from add",
        priority= 1
    )
    print("add res is ", res, " err is ", err)
    baddr = json.loads(res, object_hook=BlobAddr.from_json)
    (b, err) = await context.BlobOpen(baddr)
    (data, err) = await b.Read(100)
    str = data.decode('utf-8')
    print("func add ", str)
    return "add with sub result "+str

async def sub(context, parameters):
    addr = context.NewBlobAddr()
    (b, ret) = await context.BlobCreate(addr)
    print("sub xxx ", addr)
    await b.Write(bytes("test blob", 'utf-8'))
    print("sub 4")
    await b.Close()
    print("sub 5")
    (b, err) = await context.BlobOpen(b.addr)
    (data, err) = await b.Read(100)
    str = data.decode('utf-8')
    print("func sub ", str)
    return b.addr.toJson()

async def simple1(context, parameters):
    print("simple1 1")
    (res, err) = await context.RemoteCall(
        packageName= "",
        funcName= "simple",
        parameters= "call from simple1",
        priority= 1
    )
    print("simple1 2 res {:?}", res)
    (res, err) = await context.RemoteCall(
        packageName= "",
        funcName= "simple",
        parameters= "call from simple1",
        priority= 1
    )
    print("simple1 3 res {:?}", res)
    return "Simple1 %s"%parameters


async def simple(context, parameters):
    print("simple ....")
    return "Simple with parameter '%s'"%parameters
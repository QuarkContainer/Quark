#from client import RemoteCall

import json
import common
from common import BlobAddr 

async def add(context, parameters):
    res = await context.RemoteCall(
        packageName= "",
        funcName= "sub",
        parameters= "call from add",
        priority= 1
    )
    print("add res is ", res)
    baddr = json.loads(res.res, object_hook=BlobAddr.from_json)
    res = await context.BlobOpen(baddr)
    b = res.res
    res = await b.Read(100)
    str = res.res.decode('utf-8')
    print("func add ", str)
    return "add with sub result "+str

async def sub(context, parameters):
    await context.BlobDelete("local", "testblob5")
    createres = await context.BlobCreate("testblob5")
    b  = createres.res
    await b.Write(bytes("test blob", 'utf-8'))
    print("sub 4")
    await b.Close()
    print("sub 5")
    openres = await context.BlobOpen(b.addr)
    b = openres.res
    res = await b.Read(100)
    data = res.res
    str = data.decode('utf-8')
    print("func sub ", str)
    return b.addr.toJson()

async def simple1(context, parameters):
    print("simple1 1")
    res = await context.RemoteCall(
        packageName= "",
        funcName= "simple",
        parameters= "call from simple1",
        priority= 1
    )
    print("simple1 2 res {:?}", res)
    res = await context.RemoteCall(
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
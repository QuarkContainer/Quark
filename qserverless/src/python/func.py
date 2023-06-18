#from client import RemoteCall
import asyncio
import json
import common
from common import BlobAddr 

async def wordcount(context, filenames: list[str]) -> str:
    pcount = len(filenames)
    blobMatrix = list();

    results = await asyncio.gather(
        *[context.RemoteCall(
            funcName = "map",
            filename = filenames[i],
            pcount = 2
        ) for i in range(0, pcount)]
    )
    
    for res, err in results:
        blobVec = json.loads(res)
        blobMatrix.append(blobVec)
        
    shuffBlobs = common.TransposeBlobMatrix(blobMatrix)
    
    wordCounts = dict()
    
    results = await asyncio.gather(
        *[context.RemoteCall(
            funcName = "reduce",
            blobs = shuffBlobs[i]
        ) for i in range(0, 2)]
    )
    for res, err in results:
        map = json.loads(res)
        wordCounts.update(map)
    
    return json.dumps(wordCounts)

async def map(context, filename: str, pcount: int) -> str:   
    blobs = context.NewBlobAddrVec(pcount)
    word_counts = []
    for i in range(0, pcount):
        word_counts.append(dict())
    with open(filename,'r') as file:
        contents = file.read()
        words = contents.split()
        for word in words:
            idx = hash(word) % pcount
            if word in word_counts[idx]:
                word_counts[idx][word] += 1
            else:
                word_counts[idx][word] = 1 
    
    for i in range(0, pcount):
        str = json.dumps(word_counts[i])
        (addr, err) = await context.BlobWriteAll(blobs[i], bytes(str, 'utf-8'))
        blobs[i] = addr
    
    return json.dumps(blobs)

async def reduce(context, blobs: common.BlobAddrVec) -> str:
    wordcounts = dict()
    for b in blobs :
        (data, err) = await context.BlobReadAll(b)
        str = data.decode('utf-8')
        map = json.loads(str)
        for word, count in map.items():
            if word in wordcounts:
                wordcounts[word] += count
            else:
                wordcounts[word] = count 
    return json.dumps(wordcounts)

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
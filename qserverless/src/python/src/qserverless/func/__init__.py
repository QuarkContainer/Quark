# Copyright (c) 2021 Quark Container Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import hashlib

import qserverless

async def wordcount(context, filenames: list[str]): # -> (str, qserverless.Err):
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
        if err is not None:
            return (None, qserverless.Err(err))
        blobVec = json.loads(res)
        blobMatrix.append(blobVec)
        
    shuffBlobs = qserverless.TransposeBlobMatrix(blobMatrix)
    
    wordCounts = dict()
    
    results = await asyncio.gather(
        *[context.RemoteCall(
            funcName = "reduce",
            blobs = shuffBlobs[i]
        ) for i in range(0, 2)]
    )
    for res, err in results:
        if err is not None :
            return (None, err)
        map = json.loads(res)
        wordCounts.update(map)
    
    return (json.dumps(wordCounts), None)

def hash_to_int(string):
    hash_object = hashlib.shake_128(string.encode('utf-8'))
    hash_digest = hash_object.digest(16)  # 16 bytes for shake_128
    hash_int = int.from_bytes(hash_digest, byteorder='big')
    return hash_int

async def map(context, filename: str, pcount: int): # -> (str, qserverless.Err):   
    blobs = context.NewBlobAddrVec(pcount)
    word_counts = []
    for i in range(0, pcount):
        word_counts.append(dict())
    with open(filename,'r') as file:
        contents = file.read()
        words = contents.split()
        for word in words:
            hashval = hash_to_int(word)
            #idx = hash(word) % pcount
            idx = hashval % pcount
            if word in word_counts[idx]:
                word_counts[idx][word] += 1
            else:
                word_counts[idx][word] = 1 
    # print("hash value ", hashlib.shake_128(b"my ascii string").hexdigest(4));
    # print("map1 ", filename, word_counts[0])
    # print("map2 ", filename, word_counts[1])
    for i in range(0, pcount):
        str = json.dumps(word_counts[i])
        (addr, err) = await context.BlobWriteAll(blobs[i], bytes(str, 'utf-8'))
        if err is not None :
            return (None, err)
        blobs[i] = addr
    
    return (json.dumps(blobs), None)

async def reduce(context, blobs: qserverless.BlobAddrVec): # -> (str, qserverless.Err):
    wordcounts = dict()
    for b in blobs :
        (data, err) = await context.BlobReadAll(b)
        if err is not None :
            return (None, err)
        str = data.decode('utf-8')
        map = json.loads(str)
        for word, count in map.items():
            if word in wordcounts:
                wordcounts[word] += count
            else:
                wordcounts[word] = count 
    print("reduce ", wordcounts)
    return (json.dumps(wordcounts), None)


async def call_echo(context, msg: str): # -> (str, qserverless.Err):   
    (res, err) = await context.RemoteCall(
        packageName= "",
        funcName= "echo",
        msg = msg,
        priority= 1
    )

    return ("call_echo %s"%res, None)


async def echo(context, msg: str): # -> (str, qserverless.Err):   
    print("echo .... get message", msg);
    return (msg, None)
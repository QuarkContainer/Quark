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
import sys

import qserverless
import qserverless.func.ai
from qserverless.func.ai import *

async def AITestIterate(context, test):
    print("AITest ", test)
    return await handwritingClassification2(context)

async def AITest(context, test):
    print("AITest ", test)
    return await handwritingClassification(context)

async def wordcount(context, filenames: list[str]) -> str: 
    pcount = len(filenames)
    blobMatrix = list();

    results = await asyncio.gather(
        *[context.CallFunc(
            packageName = "pypackage1",
            funcName = "map",
            filename = filenames[i],
            pcount = 2
        ) for i in range(0, pcount)]
    )
    
    for res in results:
        blobVec = json.loads(res)
        blobMatrix.append(blobVec)
        
    shuffBlobs = qserverless.TransposeBlobMatrix(blobMatrix)
    
    wordCounts = dict()
    
    results = await asyncio.gather(
        *[context.CallFunc(
            packageName = "pypackage1",
            funcName = "reduce",
            blobs = shuffBlobs[i]
        ) for i in range(0, 2)]
    )
    for res in results:
        map = json.loads(res)
        wordCounts.update(map)
    
    return json.dumps(wordCounts)

def hash_to_int(string):
    hash_object = hashlib.shake_128(string.encode('utf-8'))
    hash_digest = hash_object.digest(16)  # 16 bytes for shake_128
    hash_int = int.from_bytes(hash_digest, byteorder='big')
    return hash_int

async def map(context, filename: str, pcount: int) -> str: 
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
    print("map1 ", filename, word_counts[0])
    print("map2 ", filename, word_counts[1])
    for i in range(0, pcount):
        str = json.dumps(word_counts[i])
        addr = await context.BlobWriteAll(blobs[i], bytes(str, 'utf-8'))
        blobs[i] = addr
    
    return json.dumps(blobs)

async def reduce(context, blobs: qserverless.BlobAddrVec) -> str : 
    wordcounts = dict()
    for b in blobs :
        data = await context.BlobReadAll(b)
        str = data.decode('utf-8')
        map = json.loads(str)
        for word, count in map.items():
            if word in wordcounts:
                wordcounts[word] += count
            else:
                wordcounts[word] = count 
    return json.dumps(wordcounts)


async def call_echo(context, msg: str): # -> (str, qserverless.Err):   
    res = await context.CallFunc(
        packageName= "",
        funcName= "echo",
        msg = msg,
        priority= 1
    )

    return "call_echo %s"%res

async def echo(context, msg: str) -> str: 
    print("echo .... get message", msg);
    print('stderr', file=sys.stderr)
    return msg


async def readfile(context, filename: str):
    directory = os.getcwd()

    print("current path is ", directory)
    
    with open(filename,'r') as file:
        contents = file.read()
        return contents
    
async def IternateCall(context, msg: str) -> str :
    await context.SendToParent(msg + "asdf1")
    print("IternateCall 1")
    msg1 = await context.RecvFromParent()
    print("IternateCall 2")
    await context.SendToParent(msg + "asdf2")
    return "test result"
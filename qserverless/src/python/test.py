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

import os
import sys
import asyncio
import qserverless
import qserverless.func as func

EnvVarNodeAgentAddr     = "qserverless_nodeagentaddr";
DefaultNodeAgentAddr    = "unix:///var/lib/quark/nodeagent/sock";
GatewayAddr            = "127.0.0.1:8889";

def GetNodeAgentAddrFromEnvVar() :
    if GatewayAddr is not None :
        return GatewayAddr
    addr = os.getenv(EnvVarNodeAgentAddr)
    if addr is None :
        return DefaultNodeAgentAddr
    return addr

async def wordcount():
    # Start the background task
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    filenames = ["../test.py", "../sync_test.py"]
    res = await func.wordcount(jobContext, filenames)
    print("res is ", res)
    

async def remote_wordcount():
    # Start the background task
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    filenames = ["../test.py", "../sync_test.py"]
    res = await jobContext.CallFunc(
            #packageName = "pypackage1",
            funcName = "wordcount",
            filenames = filenames
        )
    print("res is ", res)

async def readfile():
    # Start the background task
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    filename = "qserverless/func/__init__.py"
    res = await jobContext.CallFunc(
            funcName = "readfile",
            filename = filename
        )
    print("res is ", res)

async def CallFuncEcho():
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    res = await jobContext.CallFunc(
            funcName = "echo",
            msg = "hello world"
        )
    print("CallFuncecho result ", res)

async def CallFuncCallEcho():
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    res = await jobContext.CallFunc(
            funcName = "call_echo",
            msg = "hello world"
        )
    print("CallFuncCallEcho result ", res)

async def ai():
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    res = await func.AITest(jobContext, "testai")
    print("res is ", res)

async def remote_ai():
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    res = await jobContext.CallFunc(
            funcName = "AITest",
            test = "hello world"
        )
    print("res is ", res)

async def call_none(): # -> (str, qserverless.Err):   
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    results = await asyncio.gather(
        *[jobContext.CallFunc(
            packageName = "pypackage1",
            funcName = "None",
        ) for i in range(0, 2)]
    )

    for res in results:
        print("res ", res)

async def IternateCallTest():
    qserverless.Register(GetNodeAgentAddrFromEnvVar(), "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    jobContext = qserverless.NewJobContext()
    fi = await jobContext.StartFunc(
            packageName = "pypackage1",
            funcName = "IternateCall",
            msg = "test"
    )
    print("IternateCallTest *************** 1 ", fi)
    msg1 = await jobContext.RecvFromChild(fi)
    print("IternateCallTest *************** 2 ", msg1)
    await jobContext.SendToChild(fi, "parent msg")
    msg2 = await jobContext.RecvFromChild(fi)
    print("IternateCallTest *************** 3 ", msg2)
    msg3 = await jobContext.RecvFromChild(fi)
    print("IternateCallTest *************** 4 ", msg3) 
    res = jobContext.Result(fi)
    print("IternateCallTest result ", res)
    

async def main() : 
    test = sys.argv[1]
    print("test is ", test)
    match test:
        case "echo" : 
            await CallFuncEcho()
        case "call_echo" : 
            await CallFuncCallEcho()
        case "wordcount":
            await wordcount()
        case "remote_wordcount":
            await remote_wordcount()
        case "readfile":
            await readfile()   
        case "ai":
            await ai() 
        case "remote_ai":
            await remote_ai() 
        case "call_none":
            await call_none() 
        case "IternateCallTest":
            await IternateCallTest()
asyncio.run(main())


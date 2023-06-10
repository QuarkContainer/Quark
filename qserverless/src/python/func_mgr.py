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
import uuid

import func

EnvVarNodeMgrPodId = "podid.core.qserverless.quarksoft.io";

def GetPodIdFromEnvVar() :
    podId = os.getenv(EnvVarNodeMgrPodId)
    if podId is None:
        podId = uuid.uuid4()
        return uuid.uuid4()
    else :
        return podId

class FuncMgr:
    def __init__(self):
        self.funcPodId = GetPodIdFromEnvVar()
        self.namespace = "ns1"
        packageName = "package1"

class CallResult:
    def __init__(self, res, error):
        self.res = res
        self.error = error
    def __str__(self):
        return f"CallResult: res is '{self.res}', error is '{self.error}'"

async def LocalCall(name, parameters):
    function = getattr(func, name)
    if function is None:
        return CallResult("", "There is no func named {}".format(name))
    result = await function(parameters)
    return CallResult(result, "")
        

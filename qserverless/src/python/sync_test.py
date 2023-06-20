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

import qserverless;

res = qserverless.Call(
    svcAddr = "unix:///var/lib/quark/nodeagent/node1/sock",
    namespace = "ns1", 
    packageName = "pypackage1",
    funcName = "add",
    parameters = "call from client"
)
print("result is ", res)
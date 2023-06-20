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
import qserverless
import qserverless.func as func

async def test():
    
    # Start the background task
    print("test 1");
    qserverless.Register("unix:///var/lib/quark/nodeagent/node1/sock", "ns1", "pypackage1", True)
    background_task_coroutine = asyncio.create_task(qserverless.StartSvc())
    print("test 2");
    jobContext = qserverless.NewJobContext()
    #res = await func.add(jobContext, "asdf")
    filenames = ["./test.py", "./sync_test.py"]
    (res, err) = await func.wordcount(jobContext, filenames)
    print("test 3 error is ", err);
    print("test 3 ", res);
    #await background_task_coroutine
    
asyncio.run(test())


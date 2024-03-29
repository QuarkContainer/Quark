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

import sys
import qserverless

def echo():
    res = qserverless.Call(
        # "unix:///var/lib/quark/nodeagent/node1/sock",
        svcAddr = "127.0.0.1:8889", 
        namespace = "ns1", 
        packageName = "pypackage1",
        funcName = "echo",
        msg = "hello world"
    )
    print("echo result is ", res)

def wordcount():
    filenames = ["../test.py", "../sync_test.py"]
    res = qserverless.Call(
        svcAddr = "127.0.0.1:8889", 
        namespace = "ns1", 
        packageName = "pypackage1",
        funcName = "wordcount",
        filenames = filenames
    )
    print("echo result is ", res)

def ai():
    res = qserverless.Call(
        svcAddr = "127.0.0.1:8889", 
        namespace = "ns1", 
        packageName = "pypackage1",
        funcName = "AITest",
        test = "xyz"
    )
    print("echo result is ", res)

def main() : 
    test = sys.argv[1]
    print("test is ", test)
    match test:
        case "echo" : 
            echo()
        case "wordcount":
            wordcount()
        case "ai":
            ai()
if __name__ == "__main__":
    main()
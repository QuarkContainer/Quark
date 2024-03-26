
import signal
import sys

import worker

class C1:
    def __init__(self):
        self.a = 1
        print("c1.init ...");

    def run(self, reqid, data):
        print("c1.run ...", reqid, data, self.a);
        worker.ActorSystem.send("worker2", "add", reqid, data)
        #worker.ActorSystem.send("httpactor", "send", reqid, data)
        
        self.a += 1
    
class C2:
    def __init__(self):
        self.a = 1

    def add(self, reqid, data):
        print("c2.add ...", reqid, data, self.a);
        worker.ActorSystem.send("httpactor", "send", reqid, data)
        self.a += 1

def main():
    system = worker.ActorSystem()
    system.new_actor("worker1", C1)
    system.new_actor("worker2", C2)
    system.new_http_actor("httpactor", "worker1", "run", 9876)
    system.wait()


main()

import signal
import sys

import worker

class C1:
    def __init__(self):
        self.a = 1
        print("c1.init ...");

    def run(self, reqid, data):
        print("c1.run ...", reqid, data);
        worker.ActorSystem.send("httpactor", "send", reqid, data)
        self.a += 1
        return self.a
    
class C2:
    def __init__(self):
        self.a = 1

    def add(self, a):
        self.a += a
        return self.a

def main():
    signal.signal(signal.SIGINT, signal_handler)
    system = worker.ActorSystem()
    print("main 1")
    system.new_actor("worker1", C1)
    print("main 2")
    system.new_http_actor("httpactor", "worker1", "run", 9876)
    print("main 3")
    system.wait()

def signal_handler(signal, frame):
    print("Closing main-thread.This will also close the background thread because is set as daemon.")
    sys.exit(0)


main()
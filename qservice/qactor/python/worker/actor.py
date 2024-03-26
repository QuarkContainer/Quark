from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from functools import wraps, partial
import threading
import queue
import qactor

class ActorSystem:
    def __init__(self):
        signal.signal(signal.SIGINT, signal_handler)
        self.tasks = []
        self.actorInstances = dict()
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def new_actor(self, actorId, cls):
        moduleName = cls.__module__
        className = cls.__qualname__
    
        myqueue = queue.Queue()
        qactor.new_py_actor(actorId, moduleName, className, myqueue)
        actorInst = ActorProxy(actorId, cls, myqueue)
        thread = threading.Thread(target = actorInst.process, args=[])
        self.tasks.append(thread)

    def new_http_actor(self, actorId, gatewayActorId, gatewayFunc, httpPort):
        qactor.new_http_actor(actorId, gatewayActorId, gatewayFunc, httpPort)

    def send(target, funcName, reqId, data):
        qactor.sendto(target, funcName, reqId, data)

    def wait(self):
        qactor.depolyment()
        for t in self.tasks:
            t.start()
        for t in self.tasks:
            t.join()
        

class ActorProxy:
    def __init__(self, actorName, cls, queue):
        self.actorName = actorName
        self.actorInst = cls()
        self.queue = queue

    def process(self):
        while True:
            (func, req_id, data) = self.queue.get()
            func = getattr(self.actorInst, func)
            run = partial(func, req_id, data)
            run()
            
def signal_handler(signal, frame):
    sys.exit(0)


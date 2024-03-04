from concurrent.futures import ThreadPoolExecutor

from functools import wraps, partial
import threading
import qactor

class ActorSystem:
    def __init__(self):
        self.tasks = []
        self.actorInstances = dict()
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def new_actor(self, actorId, cls):
        moduleName = cls.__module__
        className = cls.__qualname__
        qactor.new_py_actor(actorId, moduleName, className)
        actorInst = ActorProxy(actorId, cls)
        thread = threading.Thread(target = actorInst.process, args=[])
        self.tasks.append(thread)

    def new_http_actor(self, actorId, gatewayActorId, gatewayFunc, httpPort):
        qactor.new_http_actor(actorId, gatewayActorId, gatewayFunc, httpPort)

    def send(self, target, funcName, reqId, data):
        qactor.sendto(target, funcName, reqId, data)

    def wait(self):
        qactor.depolyment()
        for t in self.tasks:
            t.start()
        for t in self.tasks:
            t.join()
        

class ActorProxy:
    def __init__(self, actorName, cls):
        self.actorName = actorName
        self.actorInst = cls()

    def process(self):
        while True:
            tell = qactor.recvfrom(self.actorName)
            func = getattr(self.actorInst, tell.func)
            run = partial(func, tell.reqId, tell.data)
            run()




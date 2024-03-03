from concurrent.futures import ThreadPoolExecutor

from functools import wraps, partial
import asyncio

class ActorSystem:
    def __init__(self):
        self.actorList = dict()
        self.actorInstances = dict()
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def new_actor(self, actorName, cls):
        self.actorList[actorName, cls]
        actorInst = cls()
        self.actorInstances[actorName] = actorInst

    def send(self, target, funcName, args):
        actorInst = self.actorInstances[target]
        func = getattr(actorInst, funcName)
        self.pool.task(func, args)

    async def handleRequest(self, target, funcName, args):
        actorInst = self.actorInstances[target]
        func = getattr(actorInst, funcName)
        run = partial(func, args)
        asyncio.to_thread(run)



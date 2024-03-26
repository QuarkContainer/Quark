import qactor
import queue

queue = queue.Queue()
qactor.tryput(queue)
a = queue.get()
print(a)
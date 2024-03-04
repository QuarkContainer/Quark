
import worker

class C1:
    def __init__(self):
        self.a = 1
        print("c1.init ...");

    def run(self):
        print("c1.run ...");
        worker.ActorSystem.send("httpactor", "send", 1, "asdf")
        self.a += 1
        return self.a
    
class C2:
    def __init__(self):
        self.a = 1

    def add(self, a):
        self.a += a
        return self.a

def main():
    system = worker.ActorSystem()
    print("main 1")
    system.new_actor("worker1", C1)
    print("main 2")
    system.new_http_actor("httpactor", "worker1", "run", 123)
    print("main 3")
    system.wait()

main()
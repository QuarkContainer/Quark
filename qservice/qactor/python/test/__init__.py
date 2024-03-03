
class C1:
    def __init__(self):
        self.a = 1

    def run(self):
        self.a += 1
        return self.a
    
class C2:
    def __init__(self):
        self.a = 1

    def add(self, a):
        self.a += a
        return self.a
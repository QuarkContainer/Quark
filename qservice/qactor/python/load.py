import cloudpickle as pickle

file = open('important', 'rb')

cls = pickle.load(file)
c = cls()
c.print()
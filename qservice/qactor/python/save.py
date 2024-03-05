import cloudpickle as pickle
import worker

if __name__=="__main__": 
    file = open('important', 'wb')
    pickle.dump(worker.Test, file)
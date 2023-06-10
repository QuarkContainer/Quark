#locals()["hello"]("test out put")*/

from sanic import Sanic
from sanic.response import json

app = Sanic("my-hello-world-app")

def hello(city: str) :
    print(city)

@app.post("/<FuncName>")
async def handler(request, FuncName):
    #locals()["hello"]("test out put")
    hello("rest")
    return json({'a':FuncName})

if __name__ == '__main__':
    app.run()
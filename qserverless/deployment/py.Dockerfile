# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install 'qserverless @ git+https://github.com/QuarkContainer/Quark.git#subdirectory=qserverless/src/python'

COPY . .

CMD ["python3", "client.py"]
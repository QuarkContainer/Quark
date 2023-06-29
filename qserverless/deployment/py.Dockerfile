# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get -y update
RUN apt-get -y install git

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install 'qserverless @ git+https://github.com/QuarkContainer/Quark.git#subdirectory=qserverless/src/python'
RUN pip3 install torch torchvision torchaudio

COPY . .

CMD ["python3", "src/main.py"]
# pip install grpcio grpcio-tools
all: protoc
run:
	python3 ./src/client.py
protoc:
	python -m grpc_tools.protoc -I ../../proto --python_out=./src --grpc_python_out=. func.proto
local:
	pip install -e .
install:
	python3 -m build
	python3 -m twine upload --repository qserverless dist/*
	python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps qserverless>=0.0.3

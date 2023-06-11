# pip install grpcio grpcio-tools
all: protoc
run:
	python3 client.py
protoc:
	python -m grpc_tools.protoc -I ../../proto --python_out=. --grpc_python_out=. func.proto

# pip install grpcio grpcio-tools
all: protoc
protoc:
	python -m grpc_tools.protoc -I ../../proto --python_out=. --grpc_python_out=. func.proto

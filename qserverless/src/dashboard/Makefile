# pip install grpcio grpcio-tools
all: protoc
run:
	python3 ./app.py
protoc:
	python -m grpc_tools.protoc -I ../../proto --python_out=. --grpc_python_out=. qobjs.proto

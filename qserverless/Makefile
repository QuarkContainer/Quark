All: na nm fs qctl
na:
	cargo build --bin node_agent
nm:
	cargo build --bin node_mgr
fp:
	cargo build --bin func_pod
fs:
	cargo build --bin func_svc
qctl:
	cargo build --bin qctl
deploy: fp 
	mkdir -p ./target/fp
	-rm ./target/fp/* -rf
	cp ./target/debug/func_pod ./target/fp
	cp ./fp_logging_config.yaml ./target/fp
	cp ./deployment/fp.Dockerfile ./target/fp/Dockerfile
	-docker image rm func_pod
	-docker image rm localhost:5000/func_pod
	-crictl rmi localhost:5000/func_pod
	docker build -t func_pod:latest ./target/fp
	docker image tag func_pod localhost:5000/func_pod
	docker push localhost:5000/func_pod
	#docker pull localhost:5000/func_pod
	crictl pull localhost:5000/func_pod
	#docker run --network host -it --name fp --rm func_pod
py: 
	mkdir -p ./target/py
	-rm ./target/py/* -rf
	cp ./src/python/* ./target/py -r
	cp ./deployment/py.Dockerfile ./target/py/Dockerfile
	-docker image rm py_pod
	-docker image rm localhost:5000/py_pod
	-crictl rmi localhost:5000/py_pod
	docker build -t py_pod:latest ./target/py
	docker image tag py_pod localhost:5000/py_pod
	docker push localhost:5000/py_pod
	#docker pull localhost:5000/py_pod
	crictl pull localhost:5000/py_pod
test:
	cargo run --bin test
all: rf


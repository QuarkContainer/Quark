All: na nm fp fs
na:
	cargo build --bin node_agent
nm:
	cargo build --bin node_mgr
fp:
	cargo build --bin func_pod
fs:
	cargo build --bin func_svc
deploy: fp 
	mkdir -p ./target/fp
	rm ./target/fp/*
	cp ./target/debug/func_pod ./target/fp
	cp ./fp_logging_config.yaml ./target/fp
	cp ./deployment/fp.Dockerfile ./target/fp/Dockerfile
	-docker image rm func_pod
	docker build -t func_pod:latest ./target/fp
	docker image tag func_pod localhost:5000/func_pod
	docker push localhost:5000/func_pod
	docker pull localhost:5000/func_pod
	crictl pull localhost:5000/func_pod
	#docker run --network host -it --name fp --rm func_pod
test:
	cargo run --bin test
all: rf


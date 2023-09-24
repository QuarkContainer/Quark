all: release debug

release:
	make -C ./qvisor release
	make -C ./qkernel release

debug:
	make -C ./qvisor debug
	make -C ./qkernel debug

clean:
	rm -rf target build

docker:
	sudo systemctl restart docker

install:
#	sudo cp -f ./build/qkernel.bin /usr/local/bin/
	cp -f ./build/qkernel_d.bin ./release/
#	sudo cp -f ./target/release/quark /usr/local/bin/quark
#	sudo cp -f ./target/release/quark /usr/local/bin/containerd-shim-quark-v1
	cp -f ./target/debug/quark ./release/quark_d
#	sudo cp -f ./target/debug/quark /usr/local/bin/containerd-shim-quarkd-v1
	cp -f ./vdso/vdso.so ./release/vdso.so
#	sudo mkdir -p /etc/quark/
#	sudo cp -f ./config.json /etc/quark/


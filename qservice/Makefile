All: na ss cni test

na: 
	cargo +nightly-x86_64-unknown-linux-gnu build --bin na
	cp -f na_logging_config.yaml /etc/quark/

cni:
	cargo build --bin cni
	cp -f cni_logging_config.yaml /etc/quark/
	sudo cp -f target/debug/cni /opt/cni/bin/tsot

ss:
	cargo +nightly-x86_64-unknown-linux-gnu build --bin ss
	cp -f ss_logging_config.yaml /etc/quark/
	cp -f qlet/node1.json /etc/quark/
	cp -f qlet/node2.json /etc/quark/
	
test:
	cargo build --bin test

clean:
	cargo clean
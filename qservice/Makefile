All: na test cni

na: 
	cargo +nightly-x86_64-unknown-linux-gnu build --bin na
	cp -f na_logging_config.yaml /etc/quark/

cni:
	cargo build --bin cni
	cp -f cni_logging_config.yaml /etc/quark/
	sudo cp -f target/debug/cni /opt/cni/bin/tsot

test:
	cargo build --bin test

clean:
	cargo clean
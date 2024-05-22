all: na ss cni gw sched

na:
	cargo +nightly-2023-12-11-x86_64-unknown-linux-gnu build --bin na
	sudo cp -f na_logging_config.yaml /etc/quark/

cni:
	cargo build --bin cni
	sudo cp -f cni_logging_config.yaml /etc/quark/
	sudo cp -f target/debug/cni /opt/cni/bin/tsot

gw:
	cargo +nightly-2023-12-11-x86_64-unknown-linux-gnu build --bin gateway
	sudo cp -f gateway_logging_config.yaml /etc/quark/

ss:
	cargo +nightly-2023-12-11-x86_64-unknown-linux-gnu build --bin ss
	sudo cp -f ss_logging_config.yaml /etc/quark/
	sudo cp -f qlet/node1.json /etc/quark/
	sudo cp -f qlet/node2.json /etc/quark/

sched:
	cargo build --bin scheduler
	sudo cp -f scheduler_logging_config.yaml /etc/quark/

test:
	cargo build --bin test

clean:
	cargo clean

All: na test

na: 
	cargo build --bin na
	cp -f na_logging_config.yaml /etc/quark/

test:
	cargo build --bin test

clean:
	cargo clean
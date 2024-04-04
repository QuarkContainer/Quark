.PHONY: debug release cc_debug cc_release

debug:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build 

release:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --release

cc_debug:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --features cc

cc_release:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --release --features cc
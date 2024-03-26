.PHONY: debug release cuda_debug cuda_release

debug:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build

release:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --release

cuda_debug:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --features cuda

cuda_release:
	CARGO_TARGET_DIR=../target cargo +nightly-2022-08-11-x86_64-unknown-linux-gnu build --release --features cuda

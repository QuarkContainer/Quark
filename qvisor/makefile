.PHONY: debug release cuda_debug cuda_release

debug:
	CARGO_TARGET_DIR=../target cargo +$(RUST_TOOLCHAIN) build

release:
	CARGO_TARGET_DIR=../target cargo +$(RUST_TOOLCHAIN) build --release

cuda_debug:
	CARGO_TARGET_DIR=../target cargo +$(RUST_TOOLCHAIN) build --features cuda

cuda_release:
	CARGO_TARGET_DIR=../target cargo +$(RUST_TOOLCHAIN) build --release --features cuda

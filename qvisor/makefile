.PHONY: debug release cuda_debug cuda_release cc_debug cc_release

debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build

release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release

cuda_debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --features cuda

cuda_release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release --features cuda

cc_debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --features cc

cc_release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release --features cc
.PHONY: debug release cuda_debug cuda_release snp_debug snp_release

ifdef CARGO_FEATURES
FEATURES_FLAG = --features $(CARGO_FEATURES)
endif

debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build $(FEATURES_FLAG)

release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release $(FEATURES_FLAG)

cuda_debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --features cuda $(FEATURES_FLAG)

cuda_release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release --features cuda $(FEATURES_FLAG)


snp_debug:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --features snp $(FEATURES_FLAG)

snp_release:
	CARGO_TARGET_DIR=../target cargo +$(TOOLCHAIN) build --release --features snp $(FEATURES_FLAG)
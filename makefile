#
# OUTPUT PATHS
#
PREFIX          ?= /usr/local
QBIN_DIR        ?= $(PREFIX)/bin
QCONFIG_DIR     ?= /etc/quark
QLOG_DIR        ?= /var/log/quark

#
# BUILD PATHS
#
QKERNEL_BUILD   = build
QTARGET_RELEASE = target/release
QTARGET_DEBUG   = target/debug
QKERNEL_DEBUG   = $(QKERNEL_BUILD)/qkernel_d.bin
QKERNEL_RELEASE = $(QKERNEL_BUILD)/qkernel.bin
QUARK_DEBUG     = $(QTARGET_DEBUG)/quark
QUARK_RELEASE   = $(QTARGET_RELEASE)/quark
VDSO            = vdso/vdso.so

ARCH := ${shell uname -m}
RUST_TOOLCHAIN  = nightly-2024-07-01-$(ARCH)-unknown-linux-gnu


.PHONY: all release debug clean install install-debug qvisor_release qvisor_debug cuda_make cuda_all cleanall

all:: release debug

cuda_all:: cuda_release cuda_debug

snp_all:: snp_release snp_debug

release:: qvisor_release qkernel_release $(VDSO)

debug:: qvisor_debug qkernel_debug $(VDSO)

qvisor_release:
	$(MAKE) -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) release $(if $(CARGO_FEATURES),CARGO_FEATURES=$(CARGO_FEATURES))

qkernel_release:
	$(MAKE) -C ./qkernel TOOLCHAIN=$(RUST_TOOLCHAIN) release $(if $(CARGO_FEATURES),CARGO_FEATURES=$(CARGO_FEATURES))

qvisor_debug:
	$(MAKE) -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) debug $(if $(CARGO_FEATURES),CARGO_FEATURES=$(CARGO_FEATURES))

qkernel_debug:
	$(MAKE) -C ./qkernel TOOLCHAIN=$(RUST_TOOLCHAIN) debug $(if $(CARGO_FEATURES),CARGO_FEATURES=$(CARGO_FEATURES))

$(VDSO):
	make -C ./vdso

clean:
	rm -rf target build
	make -C ./vdso clean

cleanall: clean
	make -C ./qservice clean
	make -C ./qserverless clean
	make -C ./rdma_cli clean
	make -C ./rdma_srv clean

docker:
	sudo systemctl restart docker

cuda_release:: qvisor_cuda_release qkernel_release cuda_make

cuda_debug:: qvisor_cuda_debug qkernel_debug cuda_make

qvisor_cuda_release:
	make -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) cuda_release

qvisor_cuda_debug:
	make -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) cuda_debug

snp_release:: qvisor_snp_release qkernel_snp_release $(VDSO)

snp_debug:: qvisor_snp_debug qkernel_snp_debug $(VDSO)

qkernel_snp_release:
	make -C ./qkernel TOOLCHAIN=$(RUST_TOOLCHAIN) snp_release

qkernel_snp_debug:
	make -C ./qkernel TOOLCHAIN=$(RUST_TOOLCHAIN) snp_debug

qvisor_snp_release:
	make -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) snp_release

qvisor_snp_debug:
	make -C ./qvisor TOOLCHAIN=$(RUST_TOOLCHAIN) snp_debug

install:
	-sudo cp -f $(QKERNEL_RELEASE) $(QBIN_DIR)/
	-sudo cp -f $(QUARK_RELEASE) $(QBIN_DIR)/quark
	-sudo cp -f $(QUARK_RELEASE) $(QBIN_DIR)/containerd-shim-quark-v1
	sudo cp -f $(VDSO) $(QBIN_DIR)/vdso.so
	sudo mkdir -p $(QCONFIG_DIR)
	sudo cp -f config.json $(QCONFIG_DIR)

install-debug:
	-sudo cp -f $(QKERNEL_DEBUG) $(QBIN_DIR)/
	-sudo cp -f $(QUARK_DEBUG) $(QBIN_DIR)/quark_d
	-sudo cp -f $(QUARK_DEBUG) $(QBIN_DIR)/containerd-shim-quarkd-v1
	sudo cp -f $(VDSO) $(QBIN_DIR)/vdso.so
	sudo mkdir -p $(QCONFIG_DIR)
	sudo cp -f config.json $(QCONFIG_DIR)

cuda_make:
	make -C cudaproxy release
	sudo cp -f $(QTARGET_RELEASE)/libcudaproxy.so $(QBIN_DIR)/libcudaproxy.so
	sudo cp -f $(QTARGET_RELEASE)/libcudaproxy.so $(QROOT_DIR)/test

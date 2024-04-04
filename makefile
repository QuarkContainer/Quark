#
# Paths
#
_QROOT_DIR = $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
QROOT_DIR = $(realpath $(_QROOT_DIR)../Quark)
QKERNEL_BUILD_DIR = $(QROOT_DIR)/build
QTARGET_RELASE = $(QROOT_DIR)/target/release
QTARGET_DEBUG = $(QROOT_DIR)/target/debug
QBIN_DIR ?= /usr/local/bin
QCONFIG_GLOBAL_DIR ?= /etc/quark

#
# Bins
#
QUARK = quark
QUARK_DBG = quark_d
QKERNEL_BIN_REL = qkernel.bin
QKERNEL_BIN_DBG = qkernel_d.bin
QKERNEL_DEBUG = $(QKERNEL_BUILD_DIR)/$(QKERNEL_BIN_DBG)
QKERNEL_RELEASE = $(QKERNEL_BUILD_DIR)/$(QKERNEL_BIN_REL)
QUARK_BIN_DEBUG = $(QTARGET_DEBUG)/$(QUARK)
QUARK_BIN_RELEASE = $(QTARGET_RELASE)/$(QUARK)

#
# Flags
#
MAKEFLAGS += -j4

.PHONY: all release debug clean install qvisor_release qvisor_debug cuda cc_make cc_all

all:: release debug

cc_all:: cc_release cc_debug

release:: qvisor_release qkernel_release 

qvisor_release:
	make -C ./qvisor release
	
qkernel_release:
	make -C ./qkernel release

debug:: qvisor_debug qkernel_debug 

qvisor_debug:
	make -C ./qvisor debug

qkernel_debug:
	make -C ./qkernel debug

clean:
	rm -rf target build

docker:
	sudo systemctl restart docker


cc_release:: qvisor_cc_release qkernel_cc_release

cc_debug:: qvisor_cc_debug qkernel_cc_debug

qkernel_cc_release:
	make -C ./qkernel cc_release

qkernel_cc_debug:
	make -C ./qkernel cc_debug


qvisor_cc_release:
	make -C ./qvisor cc_release

qvisor_cc_debug:
	make -C ./qvisor cc_debug


install:
#
# Release if present
#
ifneq ("$(wildcard $(QKERNEL_RELEASE))","")
ifneq ("$(wildcard $(QUARK_BIN_RELEASE))","")
	sudo cp -f $(QKERNEL_RELEASE) $(QBIN_DIR)/$(QKERNEL_BIN_REL)
	sudo cp -f $(QUARK_BIN_RELEASE) $(QBIN_DIR)/$(QUARK)
	sudo cp -f $(QUARK_BIN_RELEASE) $(QBIN_DIR)/containerd-shim-quark-v1
endif
else
	@echo "Quark release is not build, will not be installed."
endif
#
# Debug if present
#
ifneq ("$(wildcard $(QKERNEL_DEBUG))","")
ifneq ("$(wildcard $(QUARK_BIN_DEBUG))","")
	sudo cp -f $(QKERNEL_DEBUG) $(QBIN_DIR)/$(QKERNEL_BIN_DBG)
	sudo cp -f $(QUARK_BIN_DEBUG) $(QBIN_DIR)/$(QUARK_DBG)
	sudo cp -f $(QUARK_BIN_DEBUG) $(QBIN_DIR)/containerd-shim-quarkd-v1
endif
else
	@echo "Quark debug is not build, will not be installed."
endif
	sudo cp -f $(QROOT_DIR)/vdso/vdso.so $(QBIN_DIR)/vdso.so
#
# Install config if not present
#
# ifeq ("$(wildcard $(QCONFIG_GLOBAL_DIR)/config.json)","")
# 	sudo mkdir -p $(QCONFIG_GLOBAL_DIR)
# 	sudo cp -f $(QROOT_DIR)/config.json $(QCONFIG_GLOBAL_DIR)
# endif

# Always Install config for debug purpose
	sudo cp -f $(QROOT_DIR)/config.json $(QCONFIG_GLOBAL_DIR)

cuda:
	make -C $(QROOT_DIR)/cudaproxy release
	sudo cp -f $(QTARGET_RELASE)/libcudaproxy.so $(QBIN_DIR)/libcudaproxy.so
	sudo cp -f $(QTARGET_RELASE)/libcudaproxy.so $(QROOT_DIR)/test
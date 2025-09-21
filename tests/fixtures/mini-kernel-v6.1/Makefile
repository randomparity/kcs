# SPDX-License-Identifier: GPL-2.0
VERSION = 6
PATCHLEVEL = 1
SUBLEVEL = 0
EXTRAVERSION = -mini

NAME = Mini Test Kernel

# Architecture
ARCH ?= x86_64
ifeq ($(ARCH),x86_64)
    SRCARCH := x86
else ifeq ($(ARCH),ppc64le)
    SRCARCH := ppc64le
else ifeq ($(ARCH),s390x)
    SRCARCH := s390x
endif

# Basic kernel Makefile for testing
obj-y := kernel/ fs/ drivers/ net/ rust/
obj-y += arch/$(ARCH)/

# Test configurations
defconfig:
	@echo "CONFIG_64BIT=y" > .config
	@echo "CONFIG_X86_64=y" >> .config
	@echo "CONFIG_VFS=y" >> .config
	@echo "CONFIG_PROC_FS=y" >> .config

clean:
	rm -f .config

.PHONY: defconfig clean

# SPDX-License-Identifier: GPL-2.0
VERSION = 6
PATCHLEVEL = 1
SUBLEVEL = 0
EXTRAVERSION = -mini

NAME = Mini Test Kernel

# Architecture
ARCH ?= x86_64
SRCARCH := x86

# Basic kernel Makefile for testing
obj-y := kernel/ fs/ drivers/

# Test configurations
defconfig:
	@echo "CONFIG_64BIT=y" > .config
	@echo "CONFIG_X86_64=y" >> .config
	@echo "CONFIG_VFS=y" >> .config
	@echo "CONFIG_PROC_FS=y" >> .config

clean:
	rm -f .config

.PHONY: defconfig clean

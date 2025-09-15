// SPDX-License-Identifier: GPL-2.0
/*
 * System call implementations for mini kernel
 */

#include <linux/kernel.h>
#include <linux/syscalls.h>

/**
 * sys_open - open file system call
 */
SYSCALL_DEFINE2(open, const char __user *, filename, int, flags)
{
    /* Simplified implementation */
    return -EINVAL;
}

/**
 * sys_close - close file system call
 */
SYSCALL_DEFINE1(close, unsigned int, fd)
{
    /* Simplified implementation */
    return 0;
}

/**
 * sys_openat - open file at directory
 */
SYSCALL_DEFINE4(openat, int, dfd, const char __user *, filename,
                int, flags, umode_t, mode)
{
    /* Simplified implementation */
    return -EINVAL;
}

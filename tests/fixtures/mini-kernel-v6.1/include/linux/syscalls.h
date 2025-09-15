/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_SYSCALLS_H
#define _LINUX_SYSCALLS_H

/* Simplified syscall definitions for testing */
#define SYSCALL_DEFINE3(name, arg1_type, arg1_name, \
                       arg2_type, arg2_name, \
                       arg3_type, arg3_name) \
    long sys_##name(arg1_type arg1_name, arg2_type arg2_name, arg3_type arg3_name)

/* System call numbers */
#define __NR_read    0
#define __NR_write   1
#define __NR_open    2
#define __NR_close   3
#define __NR_openat  257

#endif /* _LINUX_SYSCALLS_H */

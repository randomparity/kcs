/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_FS_H
#define _LINUX_FS_H

#include <linux/types.h>

/**
 * struct file - represents an open file
 * @f_pos: current file position
 * @f_op: file operations
 */
struct file {
    loff_t f_pos;
    const struct file_operations *f_op;
};

/**
 * struct file_operations - file operation callbacks
 * @read: read operation
 * @write: write operation
 */
struct file_operations {
    ssize_t (*read)(struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write)(struct file *, const char __user *, size_t, loff_t *);
};

/* VFS function declarations */
extern ssize_t vfs_read(struct file *file, char __user *buf,
                       size_t count, loff_t *pos);
extern ssize_t vfs_write(struct file *file, const char __user *buf,
                        size_t count, loff_t *pos);

/* File handle functions */
extern struct file *fget(unsigned int fd);
extern void fput(struct file *file);

#endif /* _LINUX_FS_H */

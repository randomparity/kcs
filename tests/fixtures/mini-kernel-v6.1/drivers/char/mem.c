// SPDX-License-Identifier: GPL-2.0
/*
 * Memory devices (/dev/null, /dev/zero, etc.)
 */

#include <linux/fs.h>
#include <linux/kernel.h>

/**
 * null_read - read from /dev/null
 */
static ssize_t null_read(struct file *file, char __user *buf,
                        size_t count, loff_t *pos)
{
    return 0;
}

/**
 * null_write - write to /dev/null
 */
static ssize_t null_write(struct file *file, const char __user *buf,
                         size_t count, loff_t *pos)
{
    return count;
}

/**
 * zero_read - read from /dev/zero
 */
static ssize_t zero_read(struct file *file, char __user *buf,
                        size_t count, loff_t *pos)
{
    /* Would normally clear buffer */
    return count;
}

/* File operations for /dev/null */
static const struct file_operations null_fops = {
    .read  = null_read,
    .write = null_write,
};

/* File operations for /dev/zero */
static const struct file_operations zero_fops = {
    .read  = zero_read,
    .write = null_write,
};

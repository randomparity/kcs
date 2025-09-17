// SPDX-License-Identifier: GPL-2.0
/*
 * Test driver with ioctl commands for testing ioctl extraction
 */

#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/kernel.h>
#include <linux/module.h>

/* Define ioctl magic number */
#define TEST_IOC_MAGIC 'T'

/* Define ioctl commands */
#define TEST_IOCRESET     _IO(TEST_IOC_MAGIC, 0)
#define TEST_IOCGETVAL    _IOR(TEST_IOC_MAGIC, 1, int)
#define TEST_IOCSETVAL    _IOW(TEST_IOC_MAGIC, 2, int)
#define TEST_IOCGSTATUS   _IOR(TEST_IOC_MAGIC, 3, struct test_status)
#define TEST_IOCSSTATUS   _IOW(TEST_IOC_MAGIC, 4, struct test_status)
#define TEST_IOCXCHANGE   _IOWR(TEST_IOC_MAGIC, 5, struct test_data)

/* Another magic number for testing */
#define ALT_IOC_MAGIC     0xAB
#define ALT_IOCCOMMAND    _IO(ALT_IOC_MAGIC, 0x10)

/* Test structures */
struct test_status {
    int state;
    unsigned long flags;
};

struct test_data {
    int input;
    int output;
};

/**
 * test_ioctl - handle ioctl commands for test device
 * @file: file pointer
 * @cmd: ioctl command
 * @arg: ioctl argument
 */
static long test_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case TEST_IOCRESET:
        /* Reset device */
        return 0;

    case TEST_IOCGETVAL:
        /* Get value from device */
        return 0;

    case TEST_IOCSETVAL:
        /* Set value on device */
        return 0;

    case TEST_IOCGSTATUS:
        /* Get device status */
        return 0;

    case TEST_IOCSSTATUS:
        /* Set device status */
        return 0;

    case TEST_IOCXCHANGE:
        /* Exchange data with device */
        return 0;

    case ALT_IOCCOMMAND:
        /* Alternative command */
        return 0;

    default:
        return -EINVAL;
    }
}

/**
 * test_compat_ioctl - handle 32-bit ioctl on 64-bit kernel
 */
static long test_compat_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    /* Forward to regular ioctl for simple cases */
    return test_ioctl(file, cmd, arg);
}

/* File operations with ioctl handlers */
static const struct file_operations test_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = test_ioctl,
    .compat_ioctl   = test_compat_ioctl,
};

/* Another device with different ioctl handler */
static long another_device_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    /* Handle commands specific to this device */
    return 0;
}

static const struct file_operations another_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = another_device_ioctl,
};

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Test driver for ioctl extraction");

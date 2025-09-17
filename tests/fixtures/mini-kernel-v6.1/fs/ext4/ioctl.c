// SPDX-License-Identifier: GPL-2.0
/*
 * EXT4 ioctl operations for testing
 */

#include <linux/fs.h>
#include <linux/ioctl.h>

/* EXT4 specific ioctl commands */
#define EXT4_IOC_GETFLAGS       _IOR('f', 1, long)
#define EXT4_IOC_SETFLAGS       _IOW('f', 2, long)
#define EXT4_IOC_GETVERSION     _IOR('f', 3, long)
#define EXT4_IOC_SETVERSION     _IOW('f', 4, long)
#define EXT4_IOC_GROUP_EXTEND   _IOW('f', 7, unsigned long)
#define EXT4_IOC_MOVE_EXT       _IOWR('f', 15, struct move_extent)

struct move_extent {
    __u32 reserved;
    __u32 donor_fd;
    __u64 orig_start;
    __u64 donor_start;
    __u64 len;
    __u64 moved_len;
};

long ext4_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case EXT4_IOC_GETFLAGS:
        /* Get file flags */
        return 0;

    case EXT4_IOC_SETFLAGS:
        /* Set file flags */
        return 0;

    case EXT4_IOC_GETVERSION:
        /* Get inode version */
        return 0;

    case EXT4_IOC_SETVERSION:
        /* Set inode version */
        return 0;

    case EXT4_IOC_GROUP_EXTEND:
        /* Extend filesystem group */
        return 0;

    case EXT4_IOC_MOVE_EXT:
        /* Move extents between files */
        return 0;

    default:
        return -ENOTTY;
    }
}

long ext4_compat_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    /* Forward to regular ioctl */
    return ext4_ioctl(file, cmd, arg);
}

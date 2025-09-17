// SPDX-License-Identifier: GPL-2.0
/*
 * Mini ext4 filesystem implementation for testing subsystem analysis
 */

#include <linux/fs.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sysfs.h>
#include <linux/proc_fs.h>

/* Module parameters for testing */
static int ext4_debug = 0;
module_param(ext4_debug, int, 0644);
MODULE_PARM_DESC(ext4_debug, "Enable ext4 debug messages");

static int ext4_max_batch_time = 15000;
module_param(ext4_max_batch_time, int, 0644);
MODULE_PARM_DESC(ext4_max_batch_time, "Maximum time to batch journal data");

/* Boot parameter */
static int __init ext4_setup_nodelalloc(char *str)
{
    /* Setup code for nodelalloc option */
    return 1;
}
__setup("ext4_nodelalloc", ext4_setup_nodelalloc);

/* Exported symbols */
void ext4_mark_inode_dirty(struct inode *inode, int flags)
{
    /* Mark inode as dirty */
}
EXPORT_SYMBOL(ext4_mark_inode_dirty);

void ext4_journal_start(struct inode *inode, int type)
{
    /* Start journal transaction */
}
EXPORT_SYMBOL_GPL(ext4_journal_start);

/* Sysfs attributes */
static ssize_t ext4_attr_show_errors(struct kobject *kobj,
                                     struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "0\n");
}

static ssize_t ext4_attr_store_errors(struct kobject *kobj,
                                      struct kobj_attribute *attr,
                                      const char *buf, size_t count)
{
    return count;
}

static struct kobj_attribute ext4_attr_errors = __ATTR(errors_count, 0644,
                                                       ext4_attr_show_errors,
                                                       ext4_attr_store_errors);

/* Proc file operations */
static int ext4_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "ext4 filesystem statistics\n");
    return 0;
}

static int ext4_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, ext4_proc_show, NULL);
}

static const struct proc_ops ext4_proc_ops = {
    .proc_open      = ext4_proc_open,
    .proc_read      = seq_read,
    .proc_lseek     = seq_lseek,
    .proc_release   = single_release,
};

/* File operations for ext4 */
static ssize_t ext4_file_read(struct file *file, char __user *buf,
                              size_t count, loff_t *pos)
{
    return 0;
}

static ssize_t ext4_file_write(struct file *file, const char __user *buf,
                               size_t count, loff_t *pos)
{
    return count;
}

static int ext4_file_open(struct inode *inode, struct file *file)
{
    return 0;
}

const struct file_operations ext4_file_operations = {
    .open    = ext4_file_open,
    .read    = ext4_file_read,
    .write   = ext4_file_write,
};

/* Directory operations */
static int ext4_readdir(struct file *file, struct dir_context *ctx)
{
    return 0;
}

const struct file_operations ext4_dir_operations = {
    .iterate_shared = ext4_readdir,
};

/* Inode operations */
static int ext4_create(struct inode *dir, struct dentry *dentry,
                      umode_t mode, bool excl)
{
    return 0;
}

static struct dentry *ext4_lookup(struct inode *dir, struct dentry *dentry,
                                  unsigned int flags)
{
    return NULL;
}

const struct inode_operations ext4_dir_inode_operations = {
    .create     = ext4_create,
    .lookup     = ext4_lookup,
};

/* Super operations */
static struct inode *ext4_alloc_inode(struct super_block *sb)
{
    return NULL;
}

static void ext4_destroy_inode(struct inode *inode)
{
}

static const struct super_operations ext4_sops = {
    .alloc_inode    = ext4_alloc_inode,
    .destroy_inode  = ext4_destroy_inode,
};

/* Mount and initialization */
static struct dentry *ext4_mount(struct file_system_type *fs_type, int flags,
                                 const char *dev_name, void *data)
{
    return NULL;
}

static struct file_system_type ext4_fs_type = {
    .owner      = THIS_MODULE,
    .name       = "ext4",
    .mount      = ext4_mount,
};

static int __init ext4_init_fs(void)
{
    int err;

    /* Register filesystem */
    err = register_filesystem(&ext4_fs_type);
    if (err)
        return err;

    /* Create proc entry */
    proc_create("fs/ext4/stats", 0, NULL, &ext4_proc_ops);

    printk(KERN_INFO "EXT4-fs: registered.\n");
    return 0;
}

static void __exit ext4_exit_fs(void)
{
    unregister_filesystem(&ext4_fs_type);
    remove_proc_entry("fs/ext4/stats", NULL);
}

module_init(ext4_init_fs);
module_exit(ext4_exit_fs);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Fourth Extended Filesystem");
MODULE_AUTHOR("Test Author");

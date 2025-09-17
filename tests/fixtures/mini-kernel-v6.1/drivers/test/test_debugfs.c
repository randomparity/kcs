/*
 * Test debugfs entry points for KCS extraction
 */

#include <linux/debugfs.h>
#include <linux/seq_file.h>

static struct dentry *test_debugfs_dir;

/* Simple debugfs file with custom operations */
static int test_debugfs_show(struct seq_file *m, void *v)
{
    seq_printf(m, "Test debugfs content\n");
    return 0;
}

static int test_debugfs_open(struct inode *inode, struct file *file)
{
    return single_open(file, test_debugfs_show, NULL);
}

static const struct file_operations test_debugfs_fops = {
    .owner = THIS_MODULE,
    .open = test_debugfs_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

/* Debugfs attribute for simple integer */
static int test_value = 42;

static int __init test_debugfs_init(void)
{
    /* Create main debugfs directory */
    test_debugfs_dir = debugfs_create_dir("test_driver", NULL);

    /* Create various debugfs entries */
    debugfs_create_file("status", 0444, test_debugfs_dir, NULL, &test_debugfs_fops);

    /* Simple types */
    debugfs_create_u32("test_u32", 0644, test_debugfs_dir, &test_value);
    debugfs_create_bool("test_bool", 0644, test_debugfs_dir, &test_value);
    debugfs_create_x32("test_hex", 0444, test_debugfs_dir, &test_value);

    /* Blob data */
    debugfs_create_blob("test_blob", 0444, test_debugfs_dir, NULL);

    /* Using debugfs_create_file_unsafe */
    debugfs_create_file_unsafe("unsafe_file", 0444, test_debugfs_dir, NULL, &test_debugfs_fops);

    return 0;
}

static void __exit test_debugfs_exit(void)
{
    debugfs_remove_recursive(test_debugfs_dir);
}

module_init(test_debugfs_init);
module_exit(test_debugfs_exit);

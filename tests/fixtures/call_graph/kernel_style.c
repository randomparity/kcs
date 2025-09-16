// Kernel-style C code with typical patterns
// Tests realistic kernel function call patterns

#include <linux/kernel.h>
#include <linux/fs.h>

// Function declarations
static int ext4_alloc_inode(struct super_block *sb);
static void ext4_destroy_inode(struct inode *inode);
static int ext4_write_inode(struct inode *inode, struct writeback_control *wbc);

// File operations structure with function pointers
static const struct file_operations ext4_file_ops = {
    .read = generic_file_read,
    .write = ext4_file_write,
};

// Function pointer call through ops structure
static int ext4_file_open(struct inode *inode, struct file *filp) {
    int result = generic_file_open(inode, filp);  // Direct call
    if (result)
        return result;

    // Call through function pointer
    return filp->f_op->read(filp, NULL, 0, NULL);  // Indirect call
}

// Typical kernel function with multiple calls
static int ext4_create(struct inode *dir, struct dentry *dentry,
                      umode_t mode, bool excl) {
    struct inode *inode;
    int err;

    inode = ext4_alloc_inode(dir->i_sb);  // Direct call
    if (!inode)
        return -ENOMEM;

    err = ext4_write_inode(inode, NULL);  // Direct call
    if (err) {
        ext4_destroy_inode(inode);  // Direct call
        return err;
    }

    return 0;
}

// Macro usage (common in kernel)
#define ext4_error(sb, fmt, ...) \
    __ext4_error(__func__, __LINE__, sb, fmt, ##__VA_ARGS__)

static void test_error_handling(struct super_block *sb) {
    ext4_error(sb, "test error message");  // Macro call
}

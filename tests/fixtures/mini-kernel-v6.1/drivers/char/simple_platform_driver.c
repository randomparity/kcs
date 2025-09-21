/*
 * This is a generated file for testing purposes.
 * A simple platform driver example.
 */
#include <linux/module.h>
#include <linux/platform_device.h>

static int simple_platform_probe(struct platform_device *pdev)
{
    printk(KERN_INFO "Simple platform driver probed.\n");
    return 0;
}

static int simple_platform_remove(struct platform_device *pdev)
{
    printk(KERN_INFO "Simple platform driver removed.\n");
    return 0;
}

static struct platform_driver simple_platform_driver = {
    .driver = {
        .name = "simple_platform_driver",
    },
    .probe = simple_platform_probe,
    .remove = simple_platform_remove,
};

module_platform_driver(simple_platform_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jules");
MODULE_DESCRIPTION("A simple platform driver for testing.");

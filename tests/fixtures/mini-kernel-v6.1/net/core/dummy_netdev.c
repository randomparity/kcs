/*
 * This is a generated file for testing purposes.
 * A simple dummy network device.
 */
#include <linux/module.h>
#include <linux/netdevice.h>

static struct net_device *dummy_netdev;

static int dummy_netdev_open(struct net_device *dev)
{
    printk(KERN_INFO "dummy_netdev: device opened\n");
    netif_start_queue(dev);
    return 0;
}

static int dummy_netdev_stop(struct net_device *dev)
{
    printk(KERN_INFO "dummy_netdev: device closed\n");
    netif_stop_queue(dev);
    return 0;
}

static const struct net_device_ops dummy_netdev_ops = {
    .ndo_open = dummy_netdev_open,
    .ndo_stop = dummy_netdev_stop,
};

static void dummy_netdev_setup(struct net_device *dev)
{
    ether_setup(dev);
    dev->netdev_ops = &dummy_netdev_ops;
}

static int __init dummy_netdev_init(void)
{
    dummy_netdev = alloc_netdev(0, "dummyeth%d", NET_NAME_UNKNOWN, dummy_netdev_setup);
    if (!dummy_netdev)
        return -ENOMEM;
    if (register_netdev(dummy_netdev)) {
        free_netdev(dummy_netdev);
        return -ENODEV;
    }
    return 0;
}

static void __exit dummy_netdev_exit(void)
{
    unregister_netdev(dummy_netdev);
    free_netdev(dummy_netdev);
}

module_init(dummy_netdev_init);
module_exit(dummy_netdev_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jules");
MODULE_DESCRIPTION("A simple dummy network device for testing.");

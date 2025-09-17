/*
 * Test netlink socket handlers for KCS extraction
 */

#include <linux/netlink.h>
#include <net/sock.h>
#include <net/netlink.h>

/* Netlink message handler */
static int test_netlink_rcv_msg(struct sk_buff *skb, struct nlmsghdr *nlh,
                                struct netlink_ext_ack *extack)
{
    /* Handle specific netlink message */
    return 0;
}

/* Netlink input handler */
static void test_netlink_rcv(struct sk_buff *skb)
{
    netlink_rcv_skb(skb, test_netlink_rcv_msg);
}

/* Configuration for netlink socket */
static struct netlink_kernel_cfg test_netlink_cfg = {
    .input = test_netlink_rcv,
    .groups = 1,
};

/* Another netlink handler with different pattern */
static void test_netlink_input(struct sk_buff *skb)
{
    /* Process netlink message */
}

static int __init test_netlink_init(void)
{
    struct sock *test_nl_sock;

    /* Create netlink kernel socket - standard pattern */
    test_nl_sock = netlink_kernel_create(&init_net, NETLINK_TEST_PROTO,
                                         &test_netlink_cfg);
    if (!test_nl_sock)
        return -ENOMEM;

    /* Alternative pattern with inline config */
    netlink_kernel_create(&init_net, NETLINK_GENERIC, &(struct netlink_kernel_cfg){
        .input = test_netlink_input,
        .groups = 4,
    });

    /* Legacy pattern (older kernels) */
    netlink_kernel_create(&init_net, NETLINK_ROUTE, 0, test_netlink_rcv,
                         NULL, THIS_MODULE);

    return 0;
}

static void __exit test_netlink_exit(void)
{
    /* Cleanup would go here */
}

module_init(test_netlink_init);
module_exit(test_netlink_exit);

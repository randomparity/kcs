/*
 * This is a generated file for testing purposes.
 * It is based on the structure of a real prom_init.c file for PowerPC.
 */

#include <linux/kernel.h>
#include <linux/init.h>
#include <asm/prom.h>

void __init prom_init(void)
{
    printk(KERN_INFO "prom_init: Initializing Open Firmware...\n");
    /* In a real kernel, this would interact with the firmware */
}

void __init prom_init_secondary(void)
{
    /* Nothing to do here for this test file */
}

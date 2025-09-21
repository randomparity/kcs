/*
 * This is a generated file for testing purposes.
 * It is based on the structure of a real head_64.S file for PowerPC.
 */
#include <asm/page.h>

.section ".text.boot"

_HEAD
.globl _HEAD
_start:
.globl _start
    /* Minimal boot sequence for PPC64 */
    li      r3, 0
    li      r4, 0
    li      r5, 0
    mtlr    r0
    blr

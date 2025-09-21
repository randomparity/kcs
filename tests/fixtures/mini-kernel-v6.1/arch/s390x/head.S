/*
 * This is a generated file for testing purposes.
 * It is based on the structure of a real head.S file for s390.
 */
#include <asm/page.h>

.section ".text.boot"

_HEAD
.globl _HEAD
_start:
.globl _start
    /* Minimal boot sequence for s390 */
    lgr %r0, %r0
    br  %r14

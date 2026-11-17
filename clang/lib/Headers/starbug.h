/*===---- starbug.h - Starbug VLIW DSP kernel facts -------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __STARBUG_H
#define __STARBUG_H

/* Place an object in .bank0 or .bank1. The Starbug linker script aligns
   both sections to 8 bytes and adds SKEW (0 or 4, from -Wl,--defsym,SKEW=)
   to .bank1. Address bit 2 then selects the bank. Symbol alignment stays
   4 bytes: an 8-byte alignment is rounded up inside .bank1 and cancels a
   skew of 4.

   Hot loops should also say what the compiler cannot prove:

     #pragma clang loop unroll_count(N)
     #pragma clang loop pipeline_initiation_interval(N)

   unroll_count becomes llvm.loop.unroll.count. pipeline_initiation_interval
   becomes llvm.loop.pipeline.initiationinterval, which the machine
   pipeliner reads. Pointer arguments of a kernel should be __restrict__,
   and buffers passed to __builtin_assume_aligned, so loads and stores from
   distinct banks are not treated as aliases.
*/
#define starbug_bank(N) __attribute__((section(".bank" #N), aligned(4)))

#endif /* __STARBUG_H */

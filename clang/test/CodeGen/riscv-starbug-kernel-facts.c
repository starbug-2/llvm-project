// RUN: %clang_cc1 -triple riscv32-unknown-elf -emit-llvm -disable-llvm-passes \
// RUN:   -I%S/../../lib/Headers -o - %s | FileCheck %s

#include <starbug.h>

starbug_bank(0) int bank0_buf[4];
starbug_bank(1) int bank1_buf[4];

// CHECK: @bank0_buf = {{.*}}section ".bank0"
// CHECK: @bank1_buf = {{.*}}section ".bank1"

void dot(const int *restrict a, const int *restrict b, int *restrict out,
         int n) {
  a = __builtin_assume_aligned(a, 4);
  b = __builtin_assume_aligned(b, 4);
  out = __builtin_assume_aligned(out, 4);
#pragma clang loop unroll_count(8)
#pragma clang loop pipeline_initiation_interval(1)
  for (int i = 0; i < n; ++i)
    out[i] = a[i] + b[i];
}

// CHECK: define {{.*}} void @dot(ptr noalias {{.*}}, ptr noalias {{.*}}, ptr noalias
// CHECK: llvm.loop.unroll.count
// CHECK: i32 8
// CHECK: llvm.loop.pipeline.initiationinterval
// CHECK: i32 1

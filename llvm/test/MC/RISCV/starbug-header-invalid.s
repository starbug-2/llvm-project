# RUN: not llvm-mc -triple=riscv32 -mattr=+starbug-vliw < %s 2>&1 | FileCheck %s

# CHECK: :[[@LINE+1]]:12: error: predicate register must be x9-x15
sb2.header x8, un, un, un, 7

# CHECK: :[[@LINE+1]]:13: error: unconditional predicate cannot test zero
sb2.header !un, un, un, un, 7

# CHECK: :[[@LINE+1]]:12: error: predicate register must be x9-x15
sb2.header x16, un, un, un, 7

# CHECK: :[[@LINE+1]]:28: error: immediate must be an integer in the range [0, 7]
sb2.header un, un, un, un, 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
sb2.header un, un, un

# RUN: not llvm-mc -triple=riscv32 -mattr=+starbug-vliw < %s 2>&1 | FileCheck %s

# CHECK: :[[@LINE+1]]:12: error: predicate register must be 'un' or x9-x15
sb2.header x8, 0, 128

# CHECK: :[[@LINE+1]]:12: error: predicate register must be 'un' or x9-x15
sb2.header !x9, 0, 128

# CHECK: :[[@LINE+1]]:12: error: predicate register must be 'un' or x9-x15
sb2.header x16, 0, 128

# CHECK: :[[@LINE+1]]:16: error: immediate must be an integer in the range [0, 255]
sb2.header un, 256, 128

# CHECK: :[[@LINE+1]]:19: error: lane mask must be an integer in the range [1, 255]
sb2.header un, 0, 0

# CHECK: :[[@LINE+1]]:19: error: lane mask must be an integer in the range [1, 255]
sb2.header un, 0, 256

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
sb2.header un, 0

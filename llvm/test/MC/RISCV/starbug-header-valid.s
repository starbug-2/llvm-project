# RUN: llvm-mc %s -triple=riscv32 -mattr=+starbug-vliw -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+starbug-vliw < %s \
# RUN:     | llvm-objdump --mattr=+starbug-vliw -d - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

# Unconditional execute packet. Every predicate is creg=0, z=0.
# parallel = 7 sets p0, p1, and p2, so the next four operations issue together.
# p3 and reserved[11:7] are zero. Opcode is custom-0 (0001011).
# CHECK-ASM-AND-OBJ: sb2.header un, un, un, un, 7
# CHECK-ASM: encoding: [0x0b,0xe0,0x00,0x00]
# CHECK-NO-EXT: :[[#@LINE+1]]:1: error: instruction requires the following: 'starbug-vliw' (Starbug execute-packet header)
sb2.header un, un, un, un, 7
# CHECK-ASM-AND-OBJ: add a0, a1, a2
add a0, a1, a2
# CHECK-ASM-AND-OBJ: add a3, a4, a5
add a3, a4, a5
# CHECK-ASM-AND-OBJ: add a6, a7, s2
add a6, a7, s2
# CHECK-ASM-AND-OBJ: add s3, s4, s5
add s3, s4, s5

# Same header fields, written with architectural names. x9 is s1, x10 is a0,
# x15 is a5. '!' selects the zero test (z = 1). parallel = 4 is p0 only.
# CHECK-ASM-AND-OBJ: sb2.header s1, !a0, un, a5, 4
# CHECK-ASM: encoding: [0x0b,0x80,0x0e,0x25]
sb2.header x9, !x10, un, x15, 4

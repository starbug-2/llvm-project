# RUN: llvm-mc %s -triple=riscv32 -mattr=+starbug-vliw -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+starbug-vliw < %s \
# RUN:     | llvm-objdump --mattr=+starbug-vliw -d - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

# Unconditional four-lane packet. Mask bit 7 is lane 0, so 240 is lanes 0-3.
# Sense is ignored when the register field is 0. Opcode is custom-0 (0001011).
# CHECK-ASM-AND-OBJ: sb2.header un, 0, 240
# CHECK-ASM: encoding: [0x0b,0xf0,0x00,0x00]
# CHECK-NO-EXT: :[[#@LINE+1]]:1: error: instruction requires the following: 'starbug-vliw' (Starbug execute-packet header)
sb2.header un, 0, 240
# CHECK-ASM-AND-OBJ: add a0, a1, a2
add a0, a1, a2
# CHECK-ASM-AND-OBJ: add a3, a4, a5
add a3, a4, a5
# CHECK-ASM-AND-OBJ: add a6, a7, s2
add a6, a7, s2
# CHECK-ASM-AND-OBJ: add s3, s4, s5
add s3, s4, s5

# x9 is s1. Sense 128 tests lane 0 for zero and lane 1 for nonzero.
# Mask 192 occupies lanes 0 and 1.
# CHECK-ASM-AND-OBJ: sb2.header s1, 128, 192
# CHECK-ASM: encoding: [0x0b,0xc0,0x00,0x30]
sb2.header x9, 128, 192

# All eight lanes, unconditional.
# CHECK-ASM-AND-OBJ: sb2.header un, 0, 255
# CHECK-ASM: encoding: [0x0b,0xff,0x00,0x00]
sb2.header un, 0, 255

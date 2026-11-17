; Pre-RA VLIW schedule: hoist the next iteration's loads, keep a pointer
; bump next to the load it feeds, and still packetize a block larger than
; the region cap.
;
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m -stop-after=machine-scheduler -o - %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m,+c -o - %s | FileCheck %s --check-prefix=ASM

@b0 = global [32 x i32] zeroinitializer, section ".bank0", align 4
@b1 = global [32 x i32] zeroinitializer, section ".bank1", align 4

; Source order is load/load/mul/add/store, twice. The second iteration's
; loads have no dependence on the first mul, so they move above it.
define void @mac_trace(i32 %k) nounwind {
; MIR-LABEL: name: mac_trace
; MIR: LW
; MIR: LW
; MIR-NOT: MUL
; MIR: LW
; MIR: LW
; MIR: MUL
  %a0 = load i32, ptr @b0, align 4
  %c0 = load i32, ptr @b1, align 4
  %m0 = mul i32 %a0, %c0
  %s0 = add i32 %m0, %k
  %d0 = getelementptr i32, ptr @b0, i32 1
  store i32 %s0, ptr %d0, align 4
  %p1 = getelementptr i32, ptr @b0, i32 2
  %q1 = getelementptr i32, ptr @b1, i32 2
  %a1 = load i32, ptr %p1, align 4
  %c1 = load i32, ptr %q1, align 4
  %m1 = mul i32 %a1, %c1
  %s1 = add i32 %m1, %k
  %d1 = getelementptr i32, ptr @b0, i32 3
  store i32 %s1, ptr %d1, align 4
  ret void
}

; The bump of the pointer is the instruction after the load that uses it.
define i32 @pointer_bump(ptr %p) nounwind {
; MIR-LABEL: name: pointer_bump
; MIR: LW %[[BASE:[0-9]+]], 0
; MIR-NEXT: {{.*}}ADDI %[[BASE]], 4
entry:
  br label %loop
loop:
  %ptr = phi ptr [ %p, %entry ], [ %next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %loop ]
  %i = phi i32 [ 0, %entry ], [ %ni, %loop ]
  %v = load i32, ptr %ptr, align 4
  %sum = add i32 %acc, %v
  %next = getelementptr i32, ptr %ptr, i32 1
  %ni = add i32 %i, 1
  %c = icmp ult i32 %ni, 8
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %sum
}

; More instructions than -starbug-vliw-sched-region-limit (64).
; The cap splits the block. Independent multiplies still fill eight-lane
; packets on either side of the split. A sum of multiplies would fold.
define i32 @many_ops(i32 %a) nounwind {
; ASM-LABEL: many_ops:
; ASM: sb2.header un, 0, 255
; ASM: sb2.header un, 0, 255
; ASM: ret
  %u0 = mul i32 %a, 37
  %u1 = mul i32 %a, 41
  %u2 = mul i32 %a, 43
  %u3 = mul i32 %a, 47
  %u4 = mul i32 %a, 53
  %u5 = mul i32 %a, 59
  %u6 = mul i32 %a, 61
  %u7 = mul i32 %a, 67
  %u8 = mul i32 %a, 71
  %u9 = mul i32 %a, 73
  %u10 = mul i32 %a, 79
  %u11 = mul i32 %a, 83
  %u12 = mul i32 %a, 89
  %u13 = mul i32 %a, 97
  %u14 = mul i32 %a, 101
  %u15 = mul i32 %a, 103
  %u16 = mul i32 %a, 107
  %u17 = mul i32 %a, 109
  %u18 = mul i32 %a, 113
  %u19 = mul i32 %a, 127
  %u20 = mul i32 %a, 131
  %u21 = mul i32 %a, 137
  %u22 = mul i32 %a, 139
  %u23 = mul i32 %a, 149
  %u24 = mul i32 %a, 151
  %u25 = mul i32 %a, 157
  %u26 = mul i32 %a, 163
  %u27 = mul i32 %a, 167
  %u28 = mul i32 %a, 173
  %u29 = mul i32 %a, 179
  %u30 = mul i32 %a, 181
  %u31 = mul i32 %a, 191
  %u32 = mul i32 %a, 193
  %u33 = mul i32 %a, 197
  %u34 = mul i32 %a, 199
  %u35 = mul i32 %a, 211
  %u36 = mul i32 %a, 223
  %u37 = mul i32 %a, 227
  %u38 = mul i32 %a, 229
  %u39 = mul i32 %a, 233
  %s1 = xor i32 %u0, %u1
  %s2 = xor i32 %s1, %u2
  %s3 = xor i32 %s2, %u3
  %s4 = xor i32 %s3, %u4
  %s5 = xor i32 %s4, %u5
  %s6 = xor i32 %s5, %u6
  %s7 = xor i32 %s6, %u7
  %s8 = xor i32 %s7, %u8
  %s9 = xor i32 %s8, %u9
  %s10 = xor i32 %s9, %u10
  %s11 = xor i32 %s10, %u11
  %s12 = xor i32 %s11, %u12
  %s13 = xor i32 %s12, %u13
  %s14 = xor i32 %s13, %u14
  %s15 = xor i32 %s14, %u15
  %s16 = xor i32 %s15, %u16
  %s17 = xor i32 %s16, %u17
  %s18 = xor i32 %s17, %u18
  %s19 = xor i32 %s18, %u19
  %s20 = xor i32 %s19, %u20
  %s21 = xor i32 %s20, %u21
  %s22 = xor i32 %s21, %u22
  %s23 = xor i32 %s22, %u23
  %s24 = xor i32 %s23, %u24
  %s25 = xor i32 %s24, %u25
  %s26 = xor i32 %s25, %u26
  %s27 = xor i32 %s26, %u27
  %s28 = xor i32 %s27, %u28
  %s29 = xor i32 %s28, %u29
  %s30 = xor i32 %s29, %u30
  %s31 = xor i32 %s30, %u31
  %s32 = xor i32 %s31, %u32
  %s33 = xor i32 %s32, %u33
  %s34 = xor i32 %s33, %u34
  %s35 = xor i32 %s34, %u35
  %s36 = xor i32 %s35, %u36
  %s37 = xor i32 %s36, %u37
  %s38 = xor i32 %s37, %u38
  %s39 = xor i32 %s38, %u39
  ret i32 %s39
}

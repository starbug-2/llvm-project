; Eight-lane packets, the two-bank memory rule, and a shared predicate.
;
; A dense eight-lane mask is 255. Offsets 0 and 8 of a .bank0 symbol are the
; same bank, so those loads are not one packet. Offsets 0 and 4 are different
; banks and share a header. A predicated header names one register.
;
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m,+c -starbug-bank1-skew=0 < %s | FileCheck %s
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m,+c -starbug-bank1-skew=4 < %s | FileCheck %s --check-prefix=SKEW4

@bank0_a = global i32 0, section ".bank0", align 4
@bank1_a = global i32 0, section ".bank1", align 4

define i32 @eight_adds(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                       i32 %g, i32 %h) nounwind {
; CHECK-LABEL: eight_adds:
; CHECK: sb2.header un, 0, 255
; CHECK: ret
  %t0 = add i32 %a, %b
  %t1 = add i32 %c, %d
  %t2 = add i32 %e, %f
  %t3 = add i32 %g, %h
  %t4 = add i32 %a, %c
  %t5 = add i32 %b, %d
  %t6 = add i32 %e, %g
  %t7 = add i32 %f, %h
  %s0 = add i32 %t0, %t1
  %s1 = add i32 %t2, %t3
  %s2 = add i32 %t4, %t5
  %s3 = add i32 %t6, %t7
  %u0 = add i32 %s0, %s1
  %u1 = add i32 %s2, %s3
  %r = add i32 %u0, %u1
  ret i32 %r
}

define i32 @same_bank_loads(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: same_bank_loads:
; CHECK: addi a2, a2, %lo(bank0_a)
; CHECK-NEXT: sb2.header un, 0, 192
; CHECK-NEXT: lw
; CHECK-NEXT: add
; CHECK-NEXT: lw
  %p8 = getelementptr i8, ptr @bank0_a, i32 8
  %a = load i32, ptr @bank0_a
  %b = load i32, ptr %p8
  %p = add i32 %x, %y
  %s = add i32 %a, %b
  %kept = ptrtoint ptr @bank0_a to i32
  %r = add i32 %s, %p
  %r2 = add i32 %r, %kept
  ret i32 %r2
}

define i32 @different_bank_loads(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: different_bank_loads:
; CHECK: sb2.header un, 0, 224
; CHECK-NEXT: lw
; CHECK-NEXT: lw
  %p4 = getelementptr i8, ptr @bank0_a, i32 4
  %a = load i32, ptr @bank0_a
  %b = load i32, ptr %p4
  %p = add i32 %x, %y
  %s = add i32 %a, %b
  %kept = ptrtoint ptr @bank0_a to i32
  %r = add i32 %s, %p
  %r2 = add i32 %r, %kept
  ret i32 %r2
}

; Both stores are predicated on one register. Each header names that
; register, so the packet has no unconditional operation.
define void @predicated_packet(i32 %c, i32 %a, i32 %b, ptr %p, ptr %q) nounwind {
; CHECK-LABEL: predicated_packet:
; CHECK: sb2.header a0, 0, 128
; CHECK-NEXT: sw
; CHECK: sb2.header a0, 128, 128
; CHECK-NEXT: sw
; CHECK-NOT: sb2.header un,
; CHECK: ret
  %nz = icmp ne i32 %c, 0
  br i1 %nz, label %then, label %else
then:
  store i32 %a, ptr %p
  store i32 %b, ptr %q
  br label %join
else:
  store i32 %b, ptr %p
  store i32 %a, ptr %q
  br label %join
join:
  ret void
}

; Both sections start 8-byte aligned. Skew 0 leaves the bases on one bank, so
; the two loads are not one packet. Skew 4 puts .bank1 on the other bank.
define i32 @section_bases(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: section_bases:
; CHECK: lw
; CHECK: sb2.header
; CHECK: lw
; SKEW4-LABEL: section_bases:
; SKEW4: sb2.header
; SKEW4: lw
; SKEW4-NOT: sb2.header
; SKEW4: lw
  %a = load i32, ptr @bank0_a
  %b = load i32, ptr @bank1_a
  %p = add i32 %x, %y
  %s = add i32 %a, %b
  %k0 = ptrtoint ptr @bank0_a to i32
  %k1 = ptrtoint ptr @bank1_a to i32
  %r = add i32 %s, %p
  %r2 = add i32 %r, %k0
  %r3 = add i32 %r2, %k1
  ret i32 %r3
}

; A[k] and B[k] step by one word. At skew 0 both pointers start on the same
; bank and stay together. At skew 4 they start on opposite banks.
@stride_a = global [8 x i32] zeroinitializer, section ".bank0", align 4
@stride_b = global [8 x i32] zeroinitializer, section ".bank1", align 4

define i32 @stride_pair() nounwind {
; CHECK-LABEL: stride_pair:
; CHECK: addi a1, a2, 4
; CHECK-NEXT: sb2.header un, 0, 240
; CHECK-NEXT: beq
; SKEW4-LABEL: stride_pair:
; SKEW4: sb2.header un, 0, 240
; SKEW4-NEXT: lw
; SKEW4-NEXT: addi
; SKEW4-NEXT: addi
; SKEW4-NEXT: lw
entry:
  br label %loop
loop:
  %k = phi i32 [ 0, %entry ], [ %kn, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sum2, %loop ]
  %pa = getelementptr [8 x i32], ptr @stride_a, i32 0, i32 %k
  %pb = getelementptr [8 x i32], ptr @stride_b, i32 0, i32 %k
  %a = load i32, ptr %pa
  %b = load i32, ptr %pb
  %sum = add i32 %acc, %a
  %sum2 = add i32 %sum, %b
  %kn = add i32 %k, 1
  %c = icmp ult i32 %kn, 8
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %sum2
}

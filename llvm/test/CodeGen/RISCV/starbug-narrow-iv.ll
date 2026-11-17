; SCEV narrowing of a 64-bit latch on RV32.
;
; A start and a bound that fit, with the exiting increment still inside i32,
; become a 32-bit phi. A start of 2^32-1 makes the first increment overflow
; i32, so the latch stays 64-bit. A symbolic bound is versioned: the wide
; copy remains, and the narrow copy is reached only when both operands stay
; below the overflow limit.
;
; RUN: opt -mtriple=riscv32 -mcpu=starbug-vliw -passes=starbug-narrow-iv -S < %s | FileCheck %s

declare void @use32(i32)

define i32 @fits(ptr %p) {
; CHECK-LABEL: @fits(
; CHECK: phi i32
; CHECK-NOT: phi i64
; CHECK-NOT: wide
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %t = trunc i64 %iv to i32
  %g = getelementptr i32, ptr %p, i32 %t
  %v = load i32, ptr %g
  %inc = add i64 %iv, 1
  %c = icmp ult i64 %inc, 100
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %v
}

define i32 @step_two(ptr %p) {
; CHECK-LABEL: @step_two(
; CHECK: add nuw i32 {{.*}}, 2
; CHECK-NOT: phi i64
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %t = trunc i64 %iv to i32
  call void @use32(i32 %t)
  %inc = add i64 %iv, 2
  %c = icmp ult i64 %inc, 10
  br i1 %c, label %loop, label %exit
exit:
  ret i32 0
}

define i64 @start_overflows(ptr %p) {
; CHECK-LABEL: @start_overflows(
; CHECK: phi i64
; CHECK-NOT: phi i32
; CHECK-NOT: wide
entry:
  br label %loop
loop:
  %iv = phi i64 [ 4294967295, %entry ], [ %inc, %loop ]
  %t = trunc i64 %iv to i32
  call void @use32(i32 %t)
  %inc = add i64 %iv, 1
  %c = icmp ult i64 %inc, 10
  br i1 %c, label %loop, label %exit
exit:
  ret i64 %inc
}

define i64 @bound_overflows(ptr %p) {
; CHECK-LABEL: @bound_overflows(
; CHECK: phi i64
; CHECK-NOT: phi i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %t = trunc i64 %iv to i32
  call void @use32(i32 %t)
  %inc = add i64 %iv, 1
  %c = icmp ult i64 %inc, 4294967296
  br i1 %c, label %loop, label %exit
exit:
  ret i64 %inc
}

define i32 @version_symbolic(ptr %p, i64 %n) {
; CHECK-LABEL: @version_symbolic(
; CHECK: icmp ule i64 %n
; CHECK: wide
; CHECK: phi i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %t = trunc i64 %iv to i32
  %g = getelementptr i32, ptr %p, i32 %t
  %v = load i32, ptr %g
  %inc = add i64 %iv, 1
  %c = icmp ult i64 %inc, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %v
}

define i64 @keeps_wide_use(i64 %n) {
; CHECK-LABEL: @keeps_wide_use(
; CHECK: phi i64
; CHECK-NOT: phi i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %iv, 1
  %c = icmp ult i64 %inc, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i64 %iv
}

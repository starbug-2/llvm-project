; The unroll factor follows the machine. A single-block loop the pipeliner
; can overlap stays rolled: extra copies would only grow the kernel. A loop
; with a call is not pipelined, so it is unrolled until the body can fill
; the lanes, and a requested factor of 32 is then cut back by the spill guard.
;
; RUN: opt -mtriple=riscv32 -mcpu=starbug-vliw -passes=loop-unroll -S < %s | FileCheck %s --check-prefixes=CHECK,BASE
; RUN: opt -mtriple=riscv32 -mcpu=starbug-vliw -passes=loop-unroll -starbug-vliw-unroll-factor=32 -S < %s | FileCheck %s --check-prefixes=CHECK,EXPL

define void @pipelined(ptr %a, i32 %n) {
; CHECK-LABEL: @pipelined(
; CHECK-COUNT-8: mul i32
; CHECK-NOT: mul i32
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %g = getelementptr i32, ptr %a, i32 %i
  %v = load i32, ptr %g
  %m0 = mul i32 %v, 3
  %m1 = mul i32 %v, 4
  %m2 = mul i32 %v, 5
  %m3 = mul i32 %v, 6
  %m4 = mul i32 %v, 7
  %m5 = mul i32 %v, 8
  %m6 = mul i32 %v, 9
  %m7 = mul i32 %v, 10
  %inc = add i32 %i, 1
  %c = icmp ult i32 %inc, %n
  br i1 %c, label %body, label %exit
exit:
  ret void
}

declare void @use(i32)

define void @thin_call(i32 %n, i32 %x) {
; BASE-LABEL: @thin_call(
; BASE-COUNT-3: call void @use
; BASE-NOT: call void @use
; EXPL-LABEL: @thin_call(
; EXPL-COUNT-5: call void @use
; EXPL-NOT: call void @use
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %a = add i32 %i, %x
  call void @use(i32 %a)
  %inc = add i32 %i, 1
  %c = icmp ult i32 %inc, %n
  br i1 %c, label %body, label %exit
exit:
  ret void
}

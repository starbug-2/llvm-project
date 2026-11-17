; Accumulator splitting. A wrapping integer reduction becomes four partial
; sums. A floating-point reduction is split only when the fadd allows
; reassociation, or when -starbug-reassoc-reductions is set. The tail of a
; runtime trip count stays in the remainder loop.
;
; RUN: opt -mtriple=riscv32 -mcpu=starbug-vliw -passes=starbug-split-accum -S < %s | FileCheck %s --check-prefixes=CHECK,ON
; RUN: opt -mtriple=riscv32 -mcpu=starbug-vliw -passes=starbug-split-accum -starbug-reassoc-reductions -S < %s | FileCheck %s --check-prefixes=CHECK,REASSOC

define i32 @dot(ptr %a, ptr %b, i32 %n) {
; CHECK-LABEL: @dot(
; CHECK: %sum.3 = phi i32
; CHECK: %sum.2 = phi i32
; CHECK: %sum.1 = phi i32
; CHECK-NOT: %sum.4 = phi i32
; CHECK: rem.sum
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ga = getelementptr i32, ptr %a, i32 %i
  %gb = getelementptr i32, ptr %b, i32 %i
  %va = load i32, ptr %ga
  %vb = load i32, ptr %gb
  %prod = mul i32 %va, %vb
  %sum.next = add i32 %sum, %prod
  %i.next = add i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %sum.next
}

define i32 @dot_nsw(ptr %a, ptr %b, i32 %n) {
; ON-LABEL: @dot_nsw(
; ON-NOT: %sum.1 = phi i32
; REASSOC-LABEL: @dot_nsw(
; REASSOC: %sum.1 = phi i32
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ga = getelementptr i32, ptr %a, i32 %i
  %va = load i32, ptr %ga
  %sum.next = add nsw i32 %sum, %va
  %i.next = add i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %sum.next
}

define float @dotf(ptr %a, ptr %b, i32 %n) {
; ON-LABEL: @dotf(
; ON-NOT: %sum.1 = phi float
; REASSOC-LABEL: @dotf(
; REASSOC: %sum.1 = phi float
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %ga = getelementptr float, ptr %a, i32 %i
  %gb = getelementptr float, ptr %b, i32 %i
  %va = load float, ptr %ga
  %vb = load float, ptr %gb
  %prod = fmul float %va, %vb
  %sum.next = fadd float %sum, %prod
  %i.next = add i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret float %sum.next
}

define float @dotf_fast(ptr %a, ptr %b, i32 %n) {
; CHECK-LABEL: @dotf_fast(
; CHECK: %sum.3 = phi float
; CHECK: %sum.2 = phi float
; CHECK: %sum.1 = phi float
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %ga = getelementptr float, ptr %a, i32 %i
  %gb = getelementptr float, ptr %b, i32 %i
  %va = load float, ptr %ga
  %vb = load float, ptr %gb
  %prod = fmul reassoc float %va, %vb
  %sum.next = fadd reassoc float %sum, %prod
  %i.next = add i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret float %sum.next
}

define void @has_store(ptr %a, ptr %out, i32 %n) {
; CHECK-LABEL: @has_store(
; CHECK-NOT: %sum.1 = phi i32
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ga = getelementptr i32, ptr %a, i32 %i
  %va = load i32, ptr %ga
  %sum.next = add i32 %sum, %va
  store i32 %sum.next, ptr %out
  %i.next = add i32 %i, 1
  %c = icmp ult i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; Branchless argmax index. This is a FindLast recurrence, not an associative
; sum. The pass must leave it alone instead of asking for a fold opcode.
define i32 @find_last(ptr %a, i32 %n) {
; CHECK-LABEL: @find_last(
; CHECK-NOT: %idx.1 = phi i32
; CHECK: ret i32 %idx.next
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %p = getelementptr i32, ptr %a, i32 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  %idx.next = select i1 %c, i32 %i, i32 %idx
  %i.next = add i32 %i, 1
  %cont = icmp ult i32 %i.next, %n
  br i1 %cont, label %loop, label %exit
exit:
  ret i32 %idx.next
}

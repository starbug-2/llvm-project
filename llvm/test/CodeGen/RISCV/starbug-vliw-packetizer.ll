; Packetizer safety tests for the Starbug execute-packet header.
;
; sb2.header marks the following operations as one packet. Predicates in these
; tests are unconditional. The parallel field is 7 for four operations, 6 for
; three, and 4 for two. Hardware does not check independence, so a header is
; emitted only for a group the packetizer has already proved safe.
;
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m,+c < %s | FileCheck %s

; A strictly serial dependence chain has no independent pair, so it must not
; get a header.
define i32 @serial_chain(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: serial_chain:
; CHECK-NOT: sb2.header
; CHECK: ret
  %t1 = add i32 %a, %b
  %t2 = mul i32 %t1, %a
  %t3 = add i32 %t2, %t1
  %t4 = mul i32 %t3, %t2
  %t5 = add i32 %t4, %t3
  ret i32 %t5
}

; Four mutually independent multiplies. The pre-RA scheduler has four issue
; slots and the packetizer should mark them as one unconditional packet.
define i32 @independent_ops(i32 %a, i32 %b, i32 %c, i32 %d,
                            i32 %e, i32 %f, i32 %g, i32 %h) nounwind {
; CHECK-LABEL: independent_ops:
; CHECK: sb2.header un, un, un, un, 7
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
  %p1 = mul i32 %a, %b
  %p2 = mul i32 %c, %d
  %p3 = mul i32 %e, %f
  %p4 = mul i32 %g, %h
  %s1 = add i32 %p1, %p2
  %s2 = add i32 %p3, %p4
  %r  = add i32 %s1, %s2
  ret i32 %r
}

; The machine has four slots. The printer's parallel field is only three bits,
; and a packet of five operations is not a value it emits. 7 is four
; operations, 6 is three, 4 is two.
define void @bundle_width_cap(ptr %p, i32 %a, i32 %b, i32 %c, i32 %d,
                              i32 %e, i32 %f) nounwind {
; CHECK-LABEL: bundle_width_cap:
; CHECK: sb2.header
; CHECK-NOT: sb2.header un, un, un, un, 5
  %x1 = add i32 %a, %b
  %x2 = add i32 %c, %d
  %x3 = add i32 %e, %f
  %x4 = xor i32 %a, %c
  %x5 = xor i32 %b, %d
  %x6 = xor i32 %e, %a
  %y1 = add i32 %x1, %x2
  %y2 = add i32 %x3, %x4
  %y3 = add i32 %x5, %x6
  %z  = add i32 %y1, %y2
  %w  = add i32 %z, %y3
  store i32 %w, ptr %p
  ret void
}

; A branch that tests x0 becomes a predicate. The then-body runs when the
; source is nonzero, so the header names the register and does not test zero.
define void @predicated_then(i32 %c, i32 %a, i32 %b, ptr %p) nounwind {
; CHECK-LABEL: predicated_then:
; CHECK: sb2.header {{!?[a-z][0-9]+}}, un, un, un, 0
; CHECK-NOT: beq
  %nz = icmp ne i32 %c, 0
  br i1 %nz, label %then, label %join
then:
  store i32 %a, ptr %p
  br label %join
join:
  ret void
}

; Both sides of a diamond become predicates on one register, opposite sense,
; so the two stores may write the same address.
define void @predicated_diamond(i32 %c, i32 %a, i32 %b, ptr %p) nounwind {
; CHECK-LABEL: predicated_diamond:
; CHECK: sb2.header {{[a-z][0-9]+}}, un, un, un, 0
; CHECK: sb2.header !{{[a-z][0-9]+}}, un, un, un, 0
  %nz = icmp ne i32 %c, 0
  br i1 %nz, label %then, label %else
then:
  store i32 %a, ptr %p
  br label %join
else:
  store i32 %b, ptr %p
  br label %join
join:
  ret void
}

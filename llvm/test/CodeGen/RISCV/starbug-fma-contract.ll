; Stock RISC-V instruction selection matches a contracted fmul+fadd to
; fmadd.*. The contract fast-math flag is what Clang puts on the operations
; under -ffp-contract=fast. Without it, the multiply and the add stay
; separate and each takes a packet lane. -fp-contract=fast is the backend
; half of that driver default: it contracts even when the IR flags are absent.
;
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+f < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,STD
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+f -fp-contract=fast < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,FAST

define float @contract_fadd(float %a, float %b, float %c) {
; CHECK-LABEL: contract_fadd:
; CHECK:       fmadd.s fa0, fa0, fa1, fa2
; CHECK-NEXT:  ret
  %m = fmul contract float %a, %b
  %s = fadd contract float %m, %c
  ret float %s
}

; c - a*b is fnmsub: -(a*b - c) == c - a*b, with no intermediate rounding.
define float @contract_fsub(float %a, float %b, float %c) {
; CHECK-LABEL: contract_fsub:
; CHECK:       fnmsub.s fa0, fa0, fa1, fa2
; CHECK-NEXT:  ret
  %m = fmul contract float %a, %b
  %s = fsub contract float %c, %m
  ret float %s
}

define float @no_contract(float %a, float %b, float %c) {
; STD-LABEL: no_contract:
; STD:       fmul.s fa5, fa0, fa1
; STD-NEXT:  fadd.s fa0, fa5, fa2
; STD-NEXT:  ret
; FAST-LABEL: no_contract:
; FAST:       fmadd.s fa0, fa0, fa1, fa2
; FAST-NEXT:  ret
  %m = fmul float %a, %b
  %s = fadd float %m, %c
  ret float %s
}

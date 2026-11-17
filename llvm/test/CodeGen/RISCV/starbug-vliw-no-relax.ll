; Starbug packets count the bytes that follow the header. Linker relaxation
; deletes bytes after the object is written, so starbug-vliw clears +relax
; even when the IR still asks for it, and -mattr=+relax is an error.
;
; RUN: not llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+relax -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -filetype=obj -o %t.o %s
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=RELOCS

; ERR: linker relaxation cannot be enabled for starbug-vliw
; RELOCS-NOT: R_RISCV_RELAX
; RELOCS-NOT: R_RISCV_ALIGN

declare i32 @callee(i32)

define i32 @caller(i32 %a) nounwind #0 {
  %b = add i32 %a, 1
  %c = call i32 @callee(i32 %b)
  ret i32 %c
}

attributes #0 = { "target-cpu"="starbug-vliw" "target-features"="+m,+a,+c,+relax,+zicsr,+zifencei,+starbug-vliw" }

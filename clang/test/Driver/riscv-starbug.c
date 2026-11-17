// Starbug is a CPU, not an ISA string. -mcpu= must still select the
// rv32imac features, force relaxation off, and name +starbug-vliw.
// RUN: %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CPU %s
// CPU: "-target-cpu" "starbug-vliw"
// CPU: "-target-feature" "+i" "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+c"
// CPU: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// CPU: "-target-feature" "+starbug-vliw" "-target-feature" "-relax"
// CPU: "-target-abi" "ilp32"

// DSP builds contract fmul+fadd across statements. Stock ISel then emits
// fmadd. An explicit mode replaces the default.
// RUN: %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPFAST %s
// FPFAST: "-ffp-contract=fast"

// RUN: %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw -ffp-contract=off -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPOFF %s
// FPOFF: "-ffp-contract=off"
// FPOFF-NOT: "-ffp-contract=fast"

// RUN: %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw -ffp-model=precise -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPPRECISE %s
// FPPRECISE: "-ffp-contract=on"

// RUN: not %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw -mrelax -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=RELAX %s
// RELAX: error: unsupported option '-mrelax' for target 'starbug-vliw'

// --sysroot points at the xPack riscv-none-elf directory. GCC and
// riscv-none-elf-ld live in the parent. The link uses the semihosting
// runtime and the Starbug linker script, and does not pass --no-relax
// (newlib may still relax; Starbug objects have no relax relocs).
// RUN: %clang --target=riscv32-unknown-elf -mcpu=starbug-vliw \
// RUN:   --sysroot=%S/Inputs/starbug-xpack/riscv-none-elf -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LINK %s
// LINK: riscv-none-elf-ld
// LINK-NOT: --no-relax
// LINK: crt0.o
// LINK: crtbegin.o
// LINK: "-T" "{{.*}}starbug-vliw.ld"
// LINK: "-lgcc"
// LINK: "-lc"
// LINK: "-lsemihost"

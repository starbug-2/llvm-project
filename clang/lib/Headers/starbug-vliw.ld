/* Starbug bare-metal layout. RAM matches the rv32 VLIW simulator.
   .bank0 and .bank1 start 8-byte aligned, so bit 2 of each base is clear.
   SKEW=4 (passed as -Wl,--defsym,SKEW=4) moves .bank1 onto the other bank.
   SKEW defaults to 0, which leaves both bases on the same side.
   Linker relaxation stays enabled for newlib. Starbug objects are compiled
   with the relax feature off, so a packet has no R_RISCV_RELAX reloc. */
ENTRY(_start)

/* --defsym SKEW=4 wins. PROVIDE supplies 0 when the link does not. */
PROVIDE(SKEW = 0);

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}

SECTIONS
{
  . = ORIGIN(RAM);

  .text : {
    *(.text.init)
    *(.text)
    *(.text.*)
  } > RAM

  .rodata : {
    *(.rodata)
    *(.rodata.*)
  } > RAM

  .data : {
    *(.data)
    *(.data.*)
    __global_pointer$ = . + 0x800;
  } > RAM

  .preinit_array : {
    . = ALIGN(4);
    __preinit_array_start = .;
    KEEP(*(.preinit_array))
    __preinit_array_end = .;
  } > RAM

  .init_array : {
    . = ALIGN(4);
    __init_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP(*(.init_array))
    __init_array_end = .;
  } > RAM

  .fini_array : {
    . = ALIGN(4);
    __fini_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP(*(.fini_array))
    __fini_array_end = .;
  } > RAM

  .bank0 : ALIGN(8) {
    *(.bank0)
    *(.bank0.*)
  } > RAM

  .bank1 : ALIGN(8) {
    . = . + SKEW;
    *(.bank1)
    *(.bank1.*)
  } > RAM

  .bss : {
    __bss_start = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    _end = .;
  } > RAM

  ASSERT(SKEW == 0 || SKEW == 4, "SKEW must be 0 or 4")
}

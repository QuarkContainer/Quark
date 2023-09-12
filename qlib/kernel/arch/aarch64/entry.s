.globl _start, hlt
.extern rust_main
_start:
  b rust_main

.globl _start, hlt
.extern rust_main
_start:
  b rust_main
hlt:
  mov x0, 0x10000000
  mov x1, #0
  str w1, [x0]
  

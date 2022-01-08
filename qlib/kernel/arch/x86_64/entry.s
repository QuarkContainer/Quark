.globl _start, hlt
.extern rust_main
.intel_syntax noprefix
_start:
  call rust_main
hlt:
  hlt
  jmp hlt

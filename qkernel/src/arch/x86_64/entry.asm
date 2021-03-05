global _start, hlt
extern rust_main

section .text
bits 64
_start:
  ;hlt
  ;mov rdx, [rsp]
  ;lea rcx, [rsp + 8]
  ;extern rust_main
  call rust_main
hlt:
  hlt
  jmp hlt


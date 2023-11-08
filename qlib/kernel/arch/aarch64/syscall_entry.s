.globl syscall_entry, CopyPageUnsafe
.globl context_swap, __vsyscall_page

syscall_entry:
    ret
__vsyscall_page:
CopyPageUnsafe:
    ret

context_swap:
    mov     x8, x0
    mov     x9, sp
    isb
    stp     x19, x20, [x8], #16
    stp     x21, x22, [x8], #16
    stp     x23, x24, [x8], #16
    stp     x25, x26, [x8], #16
    stp     x27, x28, [x8], #16
    stp     x29, x9, [x8], #16
    str     lr, [x8], #16
    dsb ish
    str     x3, [x8]
    mov     x8, x1
    isb
    ldp     x19, x20, [x8], #16
    ldp     x21, x22, [x8], #16
    ldp     x23, x24, [x8], #16
    ldp     x25, x26, [x8], #16
    ldp     x27, x28, [x8], #16
    ldp     x29, x9, [x8], #16
    ldr     lr, [x8], #8
    ldr     x0, [x8], #8
    dsb ish
    isb
    mov     sp, x9
    str     x4, [x8]
    ret

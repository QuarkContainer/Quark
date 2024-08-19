.globl CopyPageUnsafe
.globl context_swap, __vsyscall_page

// For aarch64 this file does not serve as the actuall syscall entry
// The actuall syscall is caused by SVC exception and the entry is vector_table
// with offset 0x200, see exception.s and exception.rs
// TODO rename this file
__vsyscall_page:

/// UNSAFE CODE! the caller must make sure the to and from address are 4k aligned.
CopyPageUnsafe:
	ldp	x2, x3, [x1]
	ldp	x4, x5, [x1, #16]
	ldp	x6, x7, [x1, #32]
	ldp	x8, x9, [x1, #48]
	ldp	x10, x11, [x1, #64]
	ldp	x12, x13, [x1, #80]
	ldp	x14, x15, [x1, #96]
	ldp	x16, x17, [x1, #112]

	add	x0, x0, #256
	add	x1, x1, #128
1:
    // PAGE_SIZE
	tst	x0, #(0x1000 - 1)

	stnp	x2, x3, [x0, #-256]
	ldp	x2, x3, [x1]
	stnp	x4, x5, [x0, #16 - 256]
	ldp	x4, x5, [x1, #16]
	stnp	x6, x7, [x0, #32 - 256]
	ldp	x6, x7, [x1, #32]
	stnp	x8, x9, [x0, #48 - 256]
	ldp	x8, x9, [x1, #48]
	stnp	x10, x11, [x0, #64 - 256]
	ldp	x10, x11, [x1, #64]
	stnp	x12, x13, [x0, #80 - 256]
	ldp	x12, x13, [x1, #80]
	stnp	x14, x15, [x0, #96 - 256]
	ldp	x14, x15, [x1, #96]
	stnp	x16, x17, [x0, #112 - 256]
	ldp	x16, x17, [x1, #112]

	add	x0, x0, #128
	add	x1, x1, #128

	b.ne	1b

	stnp	x2, x3, [x0, #-256]
	stnp	x4, x5, [x0, #16 - 256]
	stnp	x6, x7, [x0, #32 - 256]
	stnp	x8, x9, [x0, #48 - 256]
	stnp	x10, x11, [x0, #64 - 256]
	stnp	x12, x13, [x0, #80 - 256]
	stnp	x14, x15, [x0, #96 - 256]
	stnp	x16, x17, [x0, #112 - 256]

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
    mov     x19, #1
    str     x19, [x8]
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
    mov     x9, #0
    str     x9, [x8]
    ret

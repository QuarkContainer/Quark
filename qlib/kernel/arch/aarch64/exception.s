// VBAR should be set to &vector_table during kernel setup
.globl vector_table
// handlers defined in exception.rs
.extern exception_handler_unhandled

.extern exception_handler_el0_sync
.extern exception_handler_el0_irq
.extern exception_handler_el0_fiq
.extern exception_handler_el0_serror

.extern exception_handler_el1h_sync
.extern exception_handler_el1h_irq
.extern exception_handler_el1h_fiq
.extern exception_handler_el1h_serror

// trapframe i.e. PtRegs:
//
//        low                                                          high
// -----------------------------------------------------------------------------
// REGS   | x0 x1 x2 ... x29 x30| sp_elx | elr_el1 | spsr_el1| x0      |zero
// -------+---------------------+--------+---------+---------+---------+--------
// PtRegs | ----regs[31]--------| sp     | pc      | pstate  | orig_x0 |__pad
// -----------------------------------------------------------------------------

.macro save_ptregs elx
    sub sp, sp, #288
    stp x0, x1, [sp, #16 * 0]
    stp x2, x3, [sp, #16 * 1]
    stp x4, x5, [sp, #16 * 2]
    stp x6, x7, [sp, #16 * 3]
    stp x8, x9, [sp, #16 * 4]
    stp x10, x11, [sp, #16 * 5]
    stp x12, x13, [sp, #16 * 6]
    stp x14, x15, [sp, #16 * 7]
    stp x16, x17, [sp, #16 * 8]
    stp x18, x19, [sp, #16 * 9]
    stp x20, x21, [sp, #16 * 10]
    stp x22, x23, [sp, #16 * 11]
    stp x24, x25, [sp, #16 * 12]
    stp x26, x27, [sp, #16 * 13]
    stp x28, x29, [sp, #16 * 14]
.if \elx == 0
    mrs x9, sp_el0;
.else
    add x9, sp, #288
.endif
    mrs x10, elr_el1
    mrs x11, spsr_el1
    stp x30, x9, [sp, #16 * 15]
    stp x10, x11,[sp, #16 * 16]
    stp x0, xzr, [sp, #16 * 17]
.endm

// TODO may need to disable/enable interrupts
.macro restore_ptregs elx
    ldp x10, x11, [sp, #16 * 16]
    ldp x30, x9,  [sp, #16 * 15]
    msr elr_el1, x10
    msr spsr_el1, x11
.if elx == 0
    msr sp_el0, x9
.endif

    ldp x0, x1, [sp, #16 * 0]
    ldp x2, x3, [sp, #16 * 1]
    ldp x4, x5, [sp, #16 * 2]
    ldp x6, x7, [sp, #16 * 3]
    ldp x8, x9, [sp, #16 * 4]
    ldp x10, x11, [sp, #16 * 5]
    ldp x12, x13, [sp, #16 * 6]
    ldp x14, x15, [sp, #16 * 7]
    ldp x16, x17, [sp, #16 * 8]
    ldp x18, x19, [sp, #16 * 9]
    ldp x20, x21, [sp, #16 * 10]
    ldp x22, x23, [sp, #16 * 11]
    ldp x24, x25, [sp, #16 * 12]
    ldp x26, x27, [sp, #16 * 13]
    ldp x28, x29, [sp, #16 * 14]

    add sp, sp, #288    // reset sp_el1
.endm


.macro save_regs elx
    stp x30, xzr, [sp, #-16]!
    stp x28, x29, [sp, #-16]!
    stp x26, x27, [sp, #-16]!
    stp x24, x25, [sp, #-16]!
    stp x22, x23, [sp, #-16]!
    stp x20, x21, [sp, #-16]!
    stp x18, x19, [sp, #-16]!
    stp x16, x17, [sp, #-16]!
    stp x14, x15, [sp, #-16]!
    stp x12, x13, [sp, #-16]!
    stp x10, x11, [sp, #-16]!
    stp x7, x9, [sp, #-16]!
    stp x6, x7, [sp, #-16]!
    stp x4, x5, [sp, #-16]!
    stp x2, x3, [sp, #-16]!
    stp x0, x1, [sp, #-16]!
    mrs x9, tpidr_el0
    mrs x10, esr_el1
    mrs x11, elr_el1
    mrs x12, spsr_el1
.if \elx == 0
    mrs x13, sp_el0
.else
    // save the "old" value of sp_el1
    // i.e. the value before pushing Xn
    add x13, sp, #16 * 16
.endif
    stp x9, x10, [sp, #-16]!
    stp x11, x12, [sp, #-16]!
    stp xzr, x13, [sp, #-16]!
.endm

.macro restore_regs elx
    ldp xzr, x13, [sp], #16
    ldp x11, x12, [sp], #16
    ldp x9, x10, [sp], #16
.if elx == 0
    msr sp_el0, x13
    // no need to restore sp_el1 from the trap frame.
    // popping out the frame does effectively the same
.endif
    msr elr_el1, x11
    msr spsr_el1, x12
    msr esr_el1, x10
    msr tpidr_el0, x9
    ldp x0, x1, [sp], #16
    ldp x2, x1, [sp], #16
    ldp x4, x1, [sp], #16
    ldp x6, x1, [sp], #16
    ldp x8, x1, [sp], #16
    ldp x10, x1, [sp], #16
    ldp x12, x1, [sp], #16
    ldp x14, x1, [sp], #16
    ldp x16, x1, [sp], #16
    ldp x18, x1, [sp], #16
    ldp x20, x1, [sp], #16
    ldp x22, x1, [sp], #16
    ldp x24, x1, [sp], #16
    ldp x26, x1, [sp], #16
    ldp x28, x1, [sp], #16
    ldp x30, xzr, [sp], #16
.endm

// mitigaton of specter bhi see
// TODO insert mitigation to the handler flow if the exception is taken
// from EL0
// https://documentation-service.arm.com/static/623c60d13b9f553dde8fd8e6?token=
// Another mitigation is do_ast upon returning to user.
.macro spectre_bhb_loop cnt
    mov x0, #\cnt
1:
    b pc + 4
    subs x18, x18, #1
    bne 1b
    dsb nsh
    isb
.endm

enter_el1h_sync:
    save_regs 1
    mov x0, sp
    bl exception_handler_el1h_sync
    restore_regs 1
    eret

enter_el1h_irq:
    save_regs 1
    mov x0, sp
    bl exception_handler_el1h_irq
    restore_regs 1
    eret

enter_el1h_fiq:
    save_regs 1
    mov x0, sp
    bl exception_handler_el1h_fiq
    restore_regs 1
    eret

enter_el1h_serror:
    save_regs 1
    mov x0, sp
    bl exception_handler_el1h_serror
    restore_regs 1
    eret

enter_el0_sync:
    save_regs 0
    mov x0, sp
    bl exception_handler_el0_sync
    restore_regs 0
    eret

enter_el0_irq:
    save_regs 0
    mov x0, sp
    bl exception_handler_el0_irq
    restore_regs 0
    eret

enter_el0_fiq:
    save_regs 0
    mov x0, sp
    bl exception_handler_el0_fiq
    restore_regs 0
    eret

enter_el0_serror:
    save_regs 0
    mov x0, sp
    bl exception_handler_el0_serror
    restore_regs 0
    eret


// this should be more sophisticated e.g. causing
// exception with brk. But for now we simply cause a panic
// without saving/restoring the registers
.macro v_empty, elx, type
.align 7
    mov x0, #0
    mov x1, #\type
    bl exception_handler_unhandled
    eret
.endm

.macro v_entry elx handler
.align 7
    b \handler
.endm



.align 11
.globl vector_table
.type vector_table STT_FUNC
vector_table:
// for v_empty entries the handler parameter is instead the vector offset.
//          ELx    Handler
v_empty     1      0
v_empty     1      1
v_empty     1      2
v_empty     1      3

// interrupts are currently masked kernel
v_entry     1      enter_el1h_sync
v_empty     1      5
v_empty     1      6
v_empty     1      7

v_entry     0      enter_el0_sync
v_entry     0      enter_el0_irq
v_entry     0      enter_el0_fiq
v_entry     0      enter_el0_serror

// 32bit state is not used
v_empty     0     12
v_empty     0     13
v_empty     0     14
v_empty     0     15

// END exception vector table

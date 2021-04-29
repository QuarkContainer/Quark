.globl syscall_entry, kernel_stack,
.globl context_swap, context_swap_to, signal_call, __vsyscall_page, rdtsc,

.globl div_zero_handler
.globl debug_handler
.globl nm_handler
.globl breakpoint_handler
.globl overflow_handler
.globl bound_range_handler
.globl invalid_op_handler
.globl device_not_available_handler
.globl double_fault_handler
.globl invalid_tss_handler
.globl segment_not_present_handler
.globl stack_segment_handler
.globl gp_handler
.globl page_fault_handler
.globl x87_fp_handler
.globl alignment_check_handler
.globl machine_check_handler
.globl simd_fp_handler
.globl virtualization_handler
.globl security_handler

.extern syscall_handler, CopyData,

.extern DivByZeroHandler
.extern DebugHandler
.extern NonmaskableInterrupt
.extern BreakpointHandler
.extern BoundRangeHandler
.extern OverflowHandler
.extern InvalidOpcodeHandler
.extern DeviceNotAvailableHandler
.extern DoubleFaultHandler
.extern InvalidTSSHandler
.extern SegmentNotPresentHandler
.extern StackSegmentHandler
.extern GPHandler
.extern PageFaultHandler
.extern X87FPHandler
.extern AlignmentCheckHandler
.extern MachineCheckHandler
.extern SIMDFPHandler
.extern VirtualizationHandler
.extern SecurityHandler

.intel_syntax noprefix

kernel_stack: .quad 0
user_stack: .quad 0

syscall_entry:
      swapgs

      //user stack
      mov gs:8, rsp

      //kernel stack
      mov rsp, gs:0

      //reserve the space for exception stack frame
      sub rsp, 1 * 8
      push gs:8
      sub rsp, 3 * 8
      push rax

      push rdi
      push rsi
      push rdx
      push rcx
      push rax
      push r8
      push r9
      push r10
      push r11

      //callee-preserved
      push rbx
      push rbp
      push r12
      push r13
      push r14
      push r15

      mov rcx, r10
      call syscall_handler


.balign 4096, 0xcc
__vsyscall_page:
    //sys_gettimeofday
    mov rax, 96
    syscall
    ret

    .balign 1024, 0xcc
    //sys_time
    mov rax, 201
    syscall
    ret

    .balign 1024, 0xcc
    //sys_getcpu
    mov rax, 309
    syscall
    ret

    .balign 4096, 0xcc
    .size __vsyscall_page, 4096

rdtsc:
    lfence
    rdtsc
    shlq rdx, 32
    addq rax, rdx
    ret

context_swap:
    mov [rdi+0x00], rsp
    mov [rdi+0x08], r15
    mov [rdi+0x10], r14
    mov [rdi+0x18], r13
    mov [rdi+0x20], r12
    mov [rdi+0x28], rbx
    mov [rdi+0x30], rbp

    mov [rdi+0x40], rdx

    mov rsp, [rsi+0x00]
    mov r15, [rsi+0x08]
    mov r14, [rsi+0x10]
    mov r13, [rsi+0x18]
    mov r12, [rsi+0x20]
    mov rbx, [rsi+0x28]
    mov rbp, [rsi+0x30]
    mov rdi, [rsi+0x38]
    mov [rsi+0x40], rcx
    ret

context_swap_to:
    mov rsp, [rsi+0x00]
    mov r15, [rsi+0x08]
    mov r14, [rsi+0x10]
    mov r13, [rsi+0x18]
    mov r12, [rsi+0x20]
    mov rbx, [rsi+0x28]
    mov rbp, [rsi+0x30]
    mov rdi, [rsi+0x38]
    mov [rsi+0x40], rcx
    ret

.macro HandlerWithoutErrorCode target
    //push dummy error code
    sub rsp, 8

    push rdi
    push rsi
    push rdx
    push rcx
    push rax
    push r8
    push r9
    push r10
    push r11

    // switch to task kernel stack
    mov rdi, rsp
    // cs of call, if it from user, last 3 bit is 0b11
    mov rsi, [rsp + 11*8]
    //caused in user mode?
    and rsi, 0b11
    jz 1f
    //load kernel rsp
    swapgs
    mov rsp, gs:0
    swapgs
    jmp 2f
    1:
    //load exception rsp, which is kernel rsp
    mov rsi, [rsp + 13 *8]
    2:
    sub rsi, 15 * 8
    mov rdx, 15
    mov rsi, rsp
    call CopyData

    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    mov rdi, rsp
    // calculate exception stack frame pointer
    add rdi, 16*8
    // align the stack pointer
    //sub rsp, 8
    call \target
    // undo stack pointer alignment
    //add rsp, 8

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx

    pop r11
    pop r10
    pop r9
    pop r8
    pop rax
    pop rcx
    pop rdx
    pop rsi
    pop rdi

    // pop error code
    add rsp, 8
    iretq
.endm

//todo: better solution with higher performance? fix this.
.macro HandlerWithErrorCode1 target
    push rdi

    mov rdi, [rsp + 3*8]    /* regs->cs */

    //caused in user mode?
    and rdi, 0b11
    jz 1f
    //load kernel rsp
    mov rdi, rsp
    swapgs
    mov rsp, gs:0
    swapgs
1:
    //load exception rsp, which is kernel rsp
    mov rsp, [rsp + 5 * 8]
2:
    pushq	[rdi + 6*8]		/* regs->ss */
    pushq	[rdi + 5*8]		/* regs->rsp */
    pushq	[rdi + 4*8]		/* regs->eflags */
    pushq	[rdi + 3*8]		/* regs->cs */
    pushq	[rdi + 2*8]		/* regs->ip */
    pushq	[rdi + 1*8]		/* regs->orig_ax */
    pushq	[rdi + 0*8]     /* regs->rdi */


    push rsi
    push rdx
    push rcx
    push rax
    push r8
    push r9
    push r10
    push r11

    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    mov rsi, [rsp + 15*8]
    mov rdi, rsp
    // calculate exception stack frame pointer
    add rdi, 16*8
    // align the stack pointer
    //sub rsp, 8
    call \target
    // undo stack pointer alignment
    //add rsp, 8

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx

    pop r11
    pop r10
    pop r9
    pop r8
    pop rax
    pop rcx
    pop rdx
    pop rsi
    pop rdi

    // pop error code
    add rsp, 8
    iretq
.endm

.macro HandlerWithErrorCode target
	push rdi
    push rsi
    push rdx
    push rcx
    push rax
    push r8
    push r9
    push r10
    push r11

    // switch to task kernel stack
    mov rdi, rsp
    // cs of call, if it from user, last 3 bit is 0b11
    mov rsi, [rsp + 11*8]
    //caused in user mode?
    and rsi, 0b11
    jz 1f
    //load kernel rsp
    swapgs
    mov rsp, gs:0
    swapgs
    jmp 2f
    1:
    //load exception rsp, which is kernel rsp
    mov rsp, [rsp + 13 *8]
    2:
    sub rsp, 15 * 8
    mov rdx, 15
    mov rsi, rsp
    call CopyData

    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    mov rsi, [rsp + 15*8]
    mov rdi, rsp
    // calculate exception stack frame pointer
    add rdi, 16*8
    // align the stack pointer
    //sub rsp, 8
    call \target
    // undo stack pointer alignment
    //add rsp, 8

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx

    pop r11
    pop r10
    pop r9
    pop r8
    pop rax
    pop rcx
    pop rdx
    pop rsi
    pop rdi

    // pop error code
    add rsp, 8
    iretq
.endm

div_zero_handler:
    HandlerWithoutErrorCode DivByZeroHandler

debug_handler:
    HandlerWithoutErrorCode DebugHandler

nm_handler:
    HandlerWithoutErrorCode NonmaskableInterrupt

breakpoint_handler:
    HandlerWithoutErrorCode BreakpointHandler

bound_range_handler:
    HandlerWithoutErrorCode BoundRangeHandler

overflow_handler:
    HandlerWithoutErrorCode OverflowHandler

invalid_op_handler:
    HandlerWithoutErrorCode InvalidOpcodeHandler

device_not_available_handler:
    HandlerWithoutErrorCode DeviceNotAvailableHandler

double_fault_handler:
    HandlerWithErrorCode DoubleFaultHandler

invalid_tss_handler:
    HandlerWithErrorCode InvalidTSSHandler

segment_not_present_handler:
    HandlerWithoutErrorCode SegmentNotPresentHandler

stack_segment_handler:
    HandlerWithErrorCode StackSegmentHandler

gp_handler:
    HandlerWithErrorCode GPHandler

page_fault_handler:
    HandlerWithErrorCode PageFaultHandler

x87_fp_handler:
    HandlerWithoutErrorCode X87FPHandler

alignment_check_handler:
    HandlerWithoutErrorCode AlignmentCheckHandler

machine_check_handler:
    HandlerWithoutErrorCode MachineCheckHandler

simd_fp_handler:
    HandlerWithoutErrorCode SIMDFPHandler

virtualization_handler:
    HandlerWithoutErrorCode VirtualizationHandler

security_handler:
    HandlerWithoutErrorCode SecurityHandler


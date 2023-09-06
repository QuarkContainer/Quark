.globl syscall_entry, CopyPageUnsafe
.globl context_swap, __vsyscall_page

syscall_entry:
    ret
__vsyscall_page:
CopyPageUnsafe:
    ret

context_swap:
    ret

#[derive(Debug, Clone)]
pub struct Frame {
    rbp: u64,
    rsp: u64,
    rip: u64,
}

impl Frame {
    pub fn new(rbp: u64, rsp: u64, rip: u64) -> Frame {
        Frame {
            rbp: rbp,
            rsp: rsp,
            rip: rip,
        }
    }

    pub fn ip(&self) -> *mut u8 {
        (self.rip - 1) as *mut u8
    }

    pub fn symbol_address(&self) -> *mut u8 {
        0 as *mut u8
    }
}

#[inline(always)]
pub fn trace_from(mut curframe: Frame, cb: &mut dyn FnMut(&Frame) -> bool) {
    loop {
        let ctxt = curframe.clone();

        let keep_going = cb(&ctxt);

        if keep_going {
            unsafe {
                curframe.rip = *((curframe.rbp + 8) as *mut u64);
                curframe.rsp = curframe.rbp;
                curframe.rbp = *(curframe.rbp as *mut u64);

                if curframe.rip == 0 || curframe.rbp <= 0xfff {
                    break;
                }
            }
        } else {
            break;
        }
    }
}

#[inline(always)]
pub fn trace1(rip: u64, rsp: u64, rbp: u64, cb: &mut dyn FnMut(&Frame) -> bool) {
    let curframe = Frame::new(rbp, rsp, rip);
    trace_from(curframe, cb);
}

#[inline(always)]
pub fn trace(cb: &mut dyn FnMut(&Frame) -> bool) {
    use x86::current::registers;
    let curframe = Frame::new(registers::rbp(), registers::rsp(), registers::rip());
    trace_from(curframe.clone(), cb);
}

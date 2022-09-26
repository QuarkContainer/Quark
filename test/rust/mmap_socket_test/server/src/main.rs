#![allow(non_snake_case)]

use libc::*;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SockAddrInet {
    pub Family: u16,
    pub Port: u16,
    pub Addr: [u8; 4],
    pub Zero: [u8; 8], // pad to sizeof(struct sockaddr).
}


fn main() {
    println!("Hello, world!");
    CreateSocket();
}

pub fn CreateSocket() {
    let fd = unsafe {
        socket(libc::AF_INET, libc::SOCK_STREAM, 0)
    };

    assert!(fd > 0);

    let addr = SockAddrInet {
        Family: libc::AF_INET as _,
        Port: 1234,
        Addr: [127, 0, 0, 1],
        Zero: [0, 0, 0, 0, 0, 0, 0, 0],
    };

    let ret = unsafe {
        libc::bind(fd, &addr as * const _ as u64 as _, 16)
    };

    println!("bind result {}", ret);

    let ret = unsafe {
        libc::listen(fd, 8)
    };

    println!("listen result {}", ret);

    //let str = "hello world!";
    //let bytes = str.as_bytes();
    
    
    let mut buf = Vec::with_capacity(1024);
    unsafe {
        buf.set_len(1024);
    }
    
    loop {
        let newsocket = unsafe {
            libc::accept(fd, 0 as _, 0 as _)
        };

        println!("accept result {}", newsocket);
        loop {
            let rcnt = unsafe {
                read(newsocket, &mut buf[0] as * mut c_void, buf.len() as _)
            };

            if rcnt == 0 {
                break;
            }
    
            let wcnt = unsafe {
                write(newsocket, &buf[0] as * const _ as u64 as * const c_void, rcnt as _)
            };
            //println!("read result {}/{}", rcnt, wcnt);
            
        }
        
        let ret = unsafe {
            close(newsocket)
        };
        println!("close result {}", ret);
    }

}

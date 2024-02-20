// client.rs
#![feature(unix_socket_ancillary_data)]

use std::os::unix::io::FromRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::io::prelude::*;
use std::os::unix::net::{SocketAncillary, AncillaryData};
use std::io::IoSliceMut;
use std::io::IoSlice;

pub static SOCKET_PATH: &'static str = "/test/loopback-socket";
// pub static SOCKET_PATH: &'static str = "/home/brad/rust/Quark/test/loopback-socket";

fn main() {
    let socket = Path::new(SOCKET_PATH);

    let mut stream = match UnixStream::connect(&socket) {
        Ok(sock) => sock,
        Err(e) => {
            println!("Couldn't connect: {e:?}");
            return
        }
    };

    match stream.write_all(b"client message") {
        Err(_) => panic!("couldn't send message"),
        Ok(_) => {}
    }

    let mut buff : [u8; 100]= [0; 100];
    let count = stream.read(&mut buff[0..10]).unwrap();
    let str = unsafe {
        std::str::from_utf8_unchecked(&buff[0..count])
    };

    println!("server said: {}", str); 

    let bufs = &mut [
        IoSliceMut::new(&mut buff),
    ][..];

    //let mut fds = [0; 8];
    let mut ancillary_buffer = [0; 128];
    let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    let size = stream.recv_vectored_with_ancillary(bufs, &mut ancillary).unwrap();
    println!("received {size}");
    let mut fds = Vec::new();
    for ancillary_result in ancillary.messages() {
        if let AncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
            for fd in scm_rights {
                println!("receive file descriptor: {fd}");
                fds.push(fd);
            }
        }
    }

    let mut f = unsafe {
        std::fs::File::from_raw_fd(fds[1])
    };
    f.write_all(b"Hello, world!\n").unwrap();

    let fds = [0, 1, 2];
    let mut ancillary_buffer = [0; 128];
    let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ancillary.add_fds(&fds[..]);

    let data = "send with fds";
    let bytes = data.as_bytes();
    let bufs = &[
        IoSlice::new(bytes),
    ];

    stream.send_vectored_with_ancillary(bufs, &mut ancillary)
        .expect("send_vectored_with_ancillary function failed");

}
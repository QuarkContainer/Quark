// server.rs
#![feature(unix_socket_ancillary_data)]

use std::fs;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::io::prelude::*;
use std::os::unix::net::{SocketAncillary, AncillaryData};
use std::io::IoSlice;
use std::io::IoSliceMut;
use std::os::unix::io::FromRawFd;

pub static SOCKET_PATH: &'static str = "/home/brad/rust/Quark/test/loopback-socket";

fn main() {
    let socket = Path::new(SOCKET_PATH);

    // Delete old socket if necessary
    if socket.exists() {
        fs::remove_file(&socket).unwrap();
    }

    // Bind to socket
    let stream = match UnixListener::bind(&socket) {
        Err(_) => panic!("failed to bind socket"),
        Ok(stream) => stream,
    };

    println!("Server started, waiting for clients");

    // Iterate over clients, blocks if no client available
    for client in stream.incoming() {
        let mut client = client.unwrap();
        let mut buff : [u8; 100]= [0; 100];
        let count = client.read(&mut buff).unwrap();
        let str = unsafe {
            std::str::from_utf8_unchecked(&buff[0..count])
        };
        println!("Client said: {}", str);
        
        match client.write_all(b"serve message\n") {
            Err(_) => panic!("couldn't send message"),
            Ok(_) => {}
        }

        let fds = [0, 1, 2];
        let mut ancillary_buffer = [0; 128];
        let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
        ancillary.add_fds(&fds[..]);
    
        let data = "send with fds";
        let bytes = data.as_bytes();
        let bufs = &[
            IoSlice::new(bytes),
        ];
    
        client.send_vectored_with_ancillary(bufs, &mut ancillary)
            .expect("send_vectored_with_ancillary function failed");
        
        let bufs = &mut [
            IoSliceMut::new(&mut buff),
        ][..];
        let mut ancillary_buffer = [0; 128];
        let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
        let size = client.recv_vectored_with_ancillary(bufs, &mut ancillary).unwrap();
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
        f.write_all(b"Hello, world! from server\n").unwrap();
    
    }
}
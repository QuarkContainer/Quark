// client.rs

use std::os::unix::net::UnixStream;
use std::path::Path;
use std::io::prelude::*;

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
    let count = stream.read(&mut buff).unwrap();
    let str = unsafe {
        std::str::from_utf8_unchecked(&buff[0..count])
    };
    println!("server said: {}", str); 
}
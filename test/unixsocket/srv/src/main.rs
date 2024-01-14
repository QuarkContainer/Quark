// server.rs

use std::fs;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::io::prelude::*;

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
        
        match client.write_all(b"serve message") {
            Err(_) => panic!("couldn't send message"),
            Ok(_) => {}
        }
        drop(client);
        
    }
}
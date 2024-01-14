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
        let mut response = String::new();
        client.unwrap().read_to_string(&mut response).unwrap();
        println!("Client said: {}", response);
    }
}
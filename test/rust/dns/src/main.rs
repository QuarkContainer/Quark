use dns_lookup::*;

fn main() {
    let hostname = "www.google.com";
    let ips: Vec<std::net::IpAddr> = lookup_host(hostname).unwrap();
    
    println!("Hello, world! {:?}", &ips);

    let hostname = "dns.ns1.svc.cluster.local";
    let ips: Vec<std::net::IpAddr> = lookup_host(hostname).unwrap();
    
    println!("Hello, world! {:?}", &ips);

    let hostname = "localhost1";
    let ips: Vec<std::net::IpAddr> = lookup_host(hostname).unwrap();
    
    println!("Hello, world! {:?}", &ips);
}

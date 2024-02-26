use reqwest::Error;
use std::env;

async fn get_request() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    let url = format!("http://{}:2345", args[1]);
    let response = reqwest::get(url).await?;
    println!("Status: {}", response.status());

    let body = response.text().await?;
    println!("Body:\n{}", body);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    match get_request().await {
        Err(e) => println!("get error {:?}", e),
        Ok(()) => (),
    }
    Ok(())
}

// use std::env;
// use std::collections::HashMap;

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let args: Vec<String> = env::args().collect();
//     let url = format!("http://{}:2345", args[1]);
//     let resp = reqwest::blocking::get(url)?
//         .json::<HashMap<String, String>>()?;
//     println!("{resp:#?}");
//     Ok(())
// }
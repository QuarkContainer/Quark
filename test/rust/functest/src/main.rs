use axum::{routing::get,routing::post, Router};
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let app = Router::new()
    .route("/liveness", get(|| async { "liveness" }))
    .route("/readiness", get(|| async { "readiness" }))
    .route("/funccall", post(|| async { "funccall" }));

    let addr = SocketAddr::from(([0, 0, 0, 0], 80));
    println!("listening on {}", addr);
    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
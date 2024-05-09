use axum::{routing::get,routing::post, Json, Router};
use std::net::SocketAddr;
use hyper::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() {
    let app = Router::new()
    .route("/liveness", get(|| async { "liveness" }))
    .route("/readiness", get(|| async { "readiness" }))
    .route("/funccall", post(post_func_call));

    let addr = SocketAddr::from(([0, 0, 0, 0], 80));
    println!("listening on {}", addr);
    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn post_func_call(
    req: String
) -> impl IntoResponse {
    return (StatusCode::OK, req)
}


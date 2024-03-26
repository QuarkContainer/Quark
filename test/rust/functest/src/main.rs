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
    Json(req): Json<PromptReq>
) -> impl IntoResponse {
    return (StatusCode::OK, Json(req))
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct PromptReq {
    pub namespace: String,
    pub func: String,
    pub prompt: String,
}
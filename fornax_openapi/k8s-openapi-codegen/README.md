This crate generates Rust bindings for the resources and operations in the Kubernetes OpenAPI spec.

NOTE: In this package k8s-openapi-codegen is modified to read [Fornax api server Swagger defintion](https://github.com/CentaurusInfra/fornax-serverless/blob/main/config/swagger.json).

# Generating the bindings

Run this binary:

```sh
cargo run
```

Bindings will now be generated in the `../` directory.

The binary accepts command-line arguments for advanced usage scenarios. Run `cargo run -- --help` for more details.

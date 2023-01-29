This crate is a Rust API client for Fornax Serverless Kubernetes extenstion server resources, it's a SHAMELESS copy from [k8s-openapi](https://arnavion.github.io/k8s-openapi/).
for more details how to use k8s_openapi to access kubernetes api and resoruces, please read [k8s-openapi](https://crates.io/crates/k8s-openapi)

### generate code

In this package k8s-openapi-codegen is modified to read [Fornax api server Swagger defintion](https://github.com/CentaurusInfra/fornax-serverless/blob/main/config/swagger.json).
Read [here](./k8s-openapi-codegen/README.md) to know how to generate fornax_openapi code under src/v1_0.

### build client

in this folder

```bash
cargo build
```

### test

```bash
cd ./fornax-openapi-tests
cargo test
```

For examples how to use generated clients and resource codes, please check [fornax-openapi-tests](./fornax-openapi-tests/src/)


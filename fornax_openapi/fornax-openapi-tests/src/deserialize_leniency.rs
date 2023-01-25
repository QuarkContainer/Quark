use fornax_openapi::serde_json;

#[test]
fn bytestring_null() {
    for (input, expected) in [
        (r#""azhzLW9wZW5hcGk=""#, &b"k8s-openapi"[..]),
        ("null", &b""[..]),
    ] {
        let actual: fornax_openapi::ByteString =
            serde_json::from_str(input).expect("couldn't deserialize ByteString");
        assert_eq!(actual.0, expected);
    }
}

#[test]
fn application() {
    for input in [
        r#"{"apiVersion":"core.fornax-serverless.centaurusinfra.io/v1","kind":"Application","metadata":{},"spec":{"selector":{},"template":{"spec":{}}}}"#,
    ] {
        let application: fornax_openapi::fornax_serverless::pkg::apis::core::v1::Application =
            serde_json::from_str(input).expect("couldn't deserialize DaemonSet");
        let containers = application
            .spec
            .expect("couldn't get ApplicationSpec")
            .containers;
        println!("application containers = {:?}", containers);
    }
}

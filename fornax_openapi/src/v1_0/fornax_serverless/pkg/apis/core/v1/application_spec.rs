// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSpec

/// ApplicationSpec defines the desired state of Application
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplicationSpec {
    /// Data contains the configuration data. Each key must consist of alphanumeric characters, '-', '_' or '.'. Values with non-UTF-8 base64 string of byte sequences
    pub config_data: Option<std::collections::BTreeMap<String, String>>,

    /// runtime image and resource requirement of a application container
    pub containers: Option<Vec<crate::api::core::v1::Container>>,

    /// application scaling policy
    pub scaling_policy: Option<crate::fornax_serverless::pkg::apis::core::v1::ScalingPolicy>,

    /// container will use grpc session service on node agent to start application session
    pub using_node_session_service: Option<bool>,
}

impl crate::DeepMerge for ApplicationSpec {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.config_data, other.config_data);
        crate::DeepMerge::merge_from(&mut self.containers, other.containers);
        crate::DeepMerge::merge_from(&mut self.scaling_policy, other.scaling_policy);
        crate::DeepMerge::merge_from(&mut self.using_node_session_service, other.using_node_session_service);
    }
}

impl<'de> crate::serde::Deserialize<'de> for ApplicationSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_config_data,
            Key_containers,
            Key_scaling_policy,
            Key_using_node_session_service,
            Other,
        }

        impl<'de> crate::serde::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
                struct Visitor;

                impl<'de> crate::serde::de::Visitor<'de> for Visitor {
                    type Value = Field;

                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str("field identifier")
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: crate::serde::de::Error {
                        Ok(match v {
                            "configData" => Field::Key_config_data,
                            "containers" => Field::Key_containers,
                            "scalingPolicy" => Field::Key_scaling_policy,
                            "usingNodeSessionService" => Field::Key_using_node_session_service,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = ApplicationSpec;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("ApplicationSpec")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_config_data: Option<std::collections::BTreeMap<String, String>> = None;
                let mut value_containers: Option<Vec<crate::api::core::v1::Container>> = None;
                let mut value_scaling_policy: Option<crate::fornax_serverless::pkg::apis::core::v1::ScalingPolicy> = None;
                let mut value_using_node_session_service: Option<bool> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_config_data => value_config_data = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_containers => value_containers = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_scaling_policy => value_scaling_policy = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_using_node_session_service => value_using_node_session_service = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(ApplicationSpec {
                    config_data: value_config_data,
                    containers: value_containers,
                    scaling_policy: value_scaling_policy,
                    using_node_session_service: value_using_node_session_service,
                })
            }
        }

        deserializer.deserialize_struct(
            "ApplicationSpec",
            &[
                "configData",
                "containers",
                "scalingPolicy",
                "usingNodeSessionService",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for ApplicationSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "ApplicationSpec",
            self.config_data.as_ref().map_or(0, |_| 1) +
            self.containers.as_ref().map_or(0, |_| 1) +
            self.scaling_policy.as_ref().map_or(0, |_| 1) +
            self.using_node_session_service.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.config_data {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "configData", value)?;
        }
        if let Some(value) = &self.containers {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "containers", value)?;
        }
        if let Some(value) = &self.scaling_policy {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "scalingPolicy", value)?;
        }
        if let Some(value) = &self.using_node_session_service {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "usingNodeSessionService", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for ApplicationSpec {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSpec".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                description: Some("ApplicationSpec defines the desired state of Application".to_owned()),
                ..Default::default()
            })),
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "configData".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Data contains the configuration data. Each key must consist of alphanumeric characters, '-', '_' or '.'. Values with non-UTF-8 base64 string of byte sequences".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
                            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                                additional_properties: Some(Box::new(
                                    crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                                        instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::String))),
                                        ..Default::default()
                                    })
                                )),
                                ..Default::default()
                            })),
                            ..Default::default()
                        }),
                    ),
                    (
                        "containers".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("runtime image and resource requirement of a application container".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Array))),
                            array: Some(Box::new(crate::schemars::schema::ArrayValidation {
                                items: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(__gen.subschema_for::<crate::api::core::v1::Container>()))),
                                ..Default::default()
                            })),
                            ..Default::default()
                        }),
                    ),
                    (
                        "scalingPolicy".to_owned(),
                        {
                            let mut schema_obj = __gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::ScalingPolicy>().into_object();
                            schema_obj.metadata = Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("application scaling policy".to_owned()),
                                ..Default::default()
                            }));
                            crate::schemars::schema::Schema::Object(schema_obj)
                        },
                    ),
                    (
                        "usingNodeSessionService".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("container will use grpc session service on node agent to start application session".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Boolean))),
                            ..Default::default()
                        }),
                    ),
                ].into(),
                ..Default::default()
            })),
            ..Default::default()
        })
    }
}

// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSessionSpec

/// ApplicationSessionSpec defines the desired state of ApplicationSession
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplicationSessionSpec {
    /// ApplicationName, client provided application
    pub application_name: Option<String>,

    /// how long to wait for before close session, default 60
    pub close_grace_period_seconds: Option<i32>,

    /// if a application instance evacuated all session, kill it, default true
    pub kill_instance_when_session_closed: Option<bool>,

    /// how long to wait for session status from Starting to Available
    pub open_timeout_seconds: Option<i32>,

    /// Session data is a base64 string pass through into application instances when session started
    pub session_data: Option<String>,
}

impl crate::DeepMerge for ApplicationSessionSpec {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.application_name, other.application_name);
        crate::DeepMerge::merge_from(&mut self.close_grace_period_seconds, other.close_grace_period_seconds);
        crate::DeepMerge::merge_from(&mut self.kill_instance_when_session_closed, other.kill_instance_when_session_closed);
        crate::DeepMerge::merge_from(&mut self.open_timeout_seconds, other.open_timeout_seconds);
        crate::DeepMerge::merge_from(&mut self.session_data, other.session_data);
    }
}

impl<'de> crate::serde::Deserialize<'de> for ApplicationSessionSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_application_name,
            Key_close_grace_period_seconds,
            Key_kill_instance_when_session_closed,
            Key_open_timeout_seconds,
            Key_session_data,
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
                            "applicationName" => Field::Key_application_name,
                            "closeGracePeriodSeconds" => Field::Key_close_grace_period_seconds,
                            "killInstanceWhenSessionClosed" => Field::Key_kill_instance_when_session_closed,
                            "openTimeoutSeconds" => Field::Key_open_timeout_seconds,
                            "sessionData" => Field::Key_session_data,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = ApplicationSessionSpec;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("ApplicationSessionSpec")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_application_name: Option<String> = None;
                let mut value_close_grace_period_seconds: Option<i32> = None;
                let mut value_kill_instance_when_session_closed: Option<bool> = None;
                let mut value_open_timeout_seconds: Option<i32> = None;
                let mut value_session_data: Option<String> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_application_name => value_application_name = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_close_grace_period_seconds => value_close_grace_period_seconds = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_kill_instance_when_session_closed => value_kill_instance_when_session_closed = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_open_timeout_seconds => value_open_timeout_seconds = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_session_data => value_session_data = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(ApplicationSessionSpec {
                    application_name: value_application_name,
                    close_grace_period_seconds: value_close_grace_period_seconds,
                    kill_instance_when_session_closed: value_kill_instance_when_session_closed,
                    open_timeout_seconds: value_open_timeout_seconds,
                    session_data: value_session_data,
                })
            }
        }

        deserializer.deserialize_struct(
            "ApplicationSessionSpec",
            &[
                "applicationName",
                "closeGracePeriodSeconds",
                "killInstanceWhenSessionClosed",
                "openTimeoutSeconds",
                "sessionData",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for ApplicationSessionSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "ApplicationSessionSpec",
            self.application_name.as_ref().map_or(0, |_| 1) +
            self.close_grace_period_seconds.as_ref().map_or(0, |_| 1) +
            self.kill_instance_when_session_closed.as_ref().map_or(0, |_| 1) +
            self.open_timeout_seconds.as_ref().map_or(0, |_| 1) +
            self.session_data.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.application_name {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "applicationName", value)?;
        }
        if let Some(value) = &self.close_grace_period_seconds {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "closeGracePeriodSeconds", value)?;
        }
        if let Some(value) = &self.kill_instance_when_session_closed {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "killInstanceWhenSessionClosed", value)?;
        }
        if let Some(value) = &self.open_timeout_seconds {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "openTimeoutSeconds", value)?;
        }
        if let Some(value) = &self.session_data {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "sessionData", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for ApplicationSessionSpec {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSessionSpec".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                description: Some("ApplicationSessionSpec defines the desired state of ApplicationSession".to_owned()),
                ..Default::default()
            })),
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "applicationName".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("ApplicationName, client provided application".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::String))),
                            ..Default::default()
                        }),
                    ),
                    (
                        "closeGracePeriodSeconds".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("how long to wait for before close session, default 60".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "killInstanceWhenSessionClosed".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("if a application instance evacuated all session, kill it, default true".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Boolean))),
                            ..Default::default()
                        }),
                    ),
                    (
                        "openTimeoutSeconds".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("how long to wait for session status from Starting to Available".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "sessionData".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Session data is a base64 string pass through into application instances when session started".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::String))),
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

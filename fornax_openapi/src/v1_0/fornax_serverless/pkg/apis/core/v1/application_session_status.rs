// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSessionStatus

/// ApplicationSessionStatus defines the observed state of ApplicationSession
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplicationSessionStatus {
    /// Endpoint this session is using
    pub access_end_points: Option<Vec<crate::fornax_serverless::pkg::apis::core::v1::AccessEndPoint>>,

    pub available_time: Option<crate::apimachinery::pkg::apis::meta::v1::Time>,

    pub available_time_micro: Option<i64>,

    pub client_sessions: Option<Vec<crate::api::core::v1::LocalObjectReference>>,

    pub close_time: Option<crate::apimachinery::pkg::apis::meta::v1::Time>,

    /// Session status, is Starting, Available or Closed.
    ///
    /// Possible enum values:
    ///  - `""` session is not allocated yet
    ///  - `"Available"` session is started on instance, not used yet
    ///  - `"Closed"` session is closed on instance
    ///  - `"Closing"` session is closing on instance, wait for session client exit
    ///  - `"InUse"` session is started on instance, session is being used
    ///  - `"Pending"` session is not allocated yet
    ///  - `"Starting"` session is send to instance, waiting for instance report session state
    ///  - `"Timeout"` session is dead, no heartbeat, should close and start a new one
    pub session_status: Option<String>,
}

impl crate::DeepMerge for ApplicationSessionStatus {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.access_end_points, other.access_end_points);
        crate::DeepMerge::merge_from(&mut self.available_time, other.available_time);
        crate::DeepMerge::merge_from(&mut self.available_time_micro, other.available_time_micro);
        crate::DeepMerge::merge_from(&mut self.client_sessions, other.client_sessions);
        crate::DeepMerge::merge_from(&mut self.close_time, other.close_time);
        crate::DeepMerge::merge_from(&mut self.session_status, other.session_status);
    }
}

impl<'de> crate::serde::Deserialize<'de> for ApplicationSessionStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_access_end_points,
            Key_available_time,
            Key_available_time_micro,
            Key_client_sessions,
            Key_close_time,
            Key_session_status,
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
                            "accessEndPoints" => Field::Key_access_end_points,
                            "availableTime" => Field::Key_available_time,
                            "availableTimeMicro" => Field::Key_available_time_micro,
                            "clientSessions" => Field::Key_client_sessions,
                            "closeTime" => Field::Key_close_time,
                            "sessionStatus" => Field::Key_session_status,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = ApplicationSessionStatus;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("ApplicationSessionStatus")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_access_end_points: Option<Vec<crate::fornax_serverless::pkg::apis::core::v1::AccessEndPoint>> = None;
                let mut value_available_time: Option<crate::apimachinery::pkg::apis::meta::v1::Time> = None;
                let mut value_available_time_micro: Option<i64> = None;
                let mut value_client_sessions: Option<Vec<crate::api::core::v1::LocalObjectReference>> = None;
                let mut value_close_time: Option<crate::apimachinery::pkg::apis::meta::v1::Time> = None;
                let mut value_session_status: Option<String> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_access_end_points => value_access_end_points = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_available_time => value_available_time = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_available_time_micro => value_available_time_micro = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_client_sessions => value_client_sessions = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_close_time => value_close_time = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_session_status => value_session_status = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(ApplicationSessionStatus {
                    access_end_points: value_access_end_points,
                    available_time: value_available_time,
                    available_time_micro: value_available_time_micro,
                    client_sessions: value_client_sessions,
                    close_time: value_close_time,
                    session_status: value_session_status,
                })
            }
        }

        deserializer.deserialize_struct(
            "ApplicationSessionStatus",
            &[
                "accessEndPoints",
                "availableTime",
                "availableTimeMicro",
                "clientSessions",
                "closeTime",
                "sessionStatus",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for ApplicationSessionStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "ApplicationSessionStatus",
            self.access_end_points.as_ref().map_or(0, |_| 1) +
            self.available_time.as_ref().map_or(0, |_| 1) +
            self.available_time_micro.as_ref().map_or(0, |_| 1) +
            self.client_sessions.as_ref().map_or(0, |_| 1) +
            self.close_time.as_ref().map_or(0, |_| 1) +
            self.session_status.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.access_end_points {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "accessEndPoints", value)?;
        }
        if let Some(value) = &self.available_time {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "availableTime", value)?;
        }
        if let Some(value) = &self.available_time_micro {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "availableTimeMicro", value)?;
        }
        if let Some(value) = &self.client_sessions {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "clientSessions", value)?;
        }
        if let Some(value) = &self.close_time {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "closeTime", value)?;
        }
        if let Some(value) = &self.session_status {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "sessionStatus", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for ApplicationSessionStatus {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationSessionStatus".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                description: Some("ApplicationSessionStatus defines the observed state of ApplicationSession".to_owned()),
                ..Default::default()
            })),
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "accessEndPoints".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Endpoint this session is using".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Array))),
                            array: Some(Box::new(crate::schemars::schema::ArrayValidation {
                                items: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(__gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::AccessEndPoint>()))),
                                ..Default::default()
                            })),
                            ..Default::default()
                        }),
                    ),
                    (
                        "availableTime".to_owned(),
                        __gen.subschema_for::<crate::apimachinery::pkg::apis::meta::v1::Time>(),
                    ),
                    (
                        "availableTimeMicro".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "clientSessions".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Array))),
                            array: Some(Box::new(crate::schemars::schema::ArrayValidation {
                                items: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(__gen.subschema_for::<crate::api::core::v1::LocalObjectReference>()))),
                                ..Default::default()
                            })),
                            ..Default::default()
                        }),
                    ),
                    (
                        "closeTime".to_owned(),
                        __gen.subschema_for::<crate::apimachinery::pkg::apis::meta::v1::Time>(),
                    ),
                    (
                        "sessionStatus".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Session status, is Starting, Available or Closed.\n\nPossible enum values:\n - `\"\"` session is not allocated yet\n - `\"Available\"` session is started on instance, not used yet\n - `\"Closed\"` session is closed on instance\n - `\"Closing\"` session is closing on instance, wait for session client exit\n - `\"InUse\"` session is started on instance, session is being used\n - `\"Pending\"` session is not allocated yet\n - `\"Starting\"` session is send to instance, waiting for instance report session state\n - `\"Timeout\"` session is dead, no heartbeat, should close and start a new one".to_owned()),
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

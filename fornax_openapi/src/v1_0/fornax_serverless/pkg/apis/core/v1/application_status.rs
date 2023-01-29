// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationStatus

/// ApplicationStatus defines the observed state of Application
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplicationStatus {
    /// Total number of instances which have been started by node
    pub allocated_instances: Option<i32>,

    /// Total number of instances pending delete and cleanup
    pub deleting_instances: Option<i32>,

    /// Total number of non-terminated pods targeted
    pub desired_instances: Option<i32>,

    /// Represents the latest available observations of a deployment's current state.
    pub history: Option<Vec<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory>>,

    /// Total number of pods which do not have session on it
    pub idle_instances: Option<i32>,

    /// The latest deploy history of this app.
    pub latest_history: Option<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory>,

    /// Total number of instances pending schedule and implement
    pub pending_instances: Option<i32>,

    /// Total number of available instances, including pod not scheduled yet
    pub total_instances: Option<i32>,
}

impl crate::DeepMerge for ApplicationStatus {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.allocated_instances, other.allocated_instances);
        crate::DeepMerge::merge_from(&mut self.deleting_instances, other.deleting_instances);
        crate::DeepMerge::merge_from(&mut self.desired_instances, other.desired_instances);
        crate::DeepMerge::merge_from(&mut self.history, other.history);
        crate::DeepMerge::merge_from(&mut self.idle_instances, other.idle_instances);
        crate::DeepMerge::merge_from(&mut self.latest_history, other.latest_history);
        crate::DeepMerge::merge_from(&mut self.pending_instances, other.pending_instances);
        crate::DeepMerge::merge_from(&mut self.total_instances, other.total_instances);
    }
}

impl<'de> crate::serde::Deserialize<'de> for ApplicationStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_allocated_instances,
            Key_deleting_instances,
            Key_desired_instances,
            Key_history,
            Key_idle_instances,
            Key_latest_history,
            Key_pending_instances,
            Key_total_instances,
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
                            "allocatedInstances" => Field::Key_allocated_instances,
                            "deletingInstances" => Field::Key_deleting_instances,
                            "desiredInstances" => Field::Key_desired_instances,
                            "history" => Field::Key_history,
                            "idleInstances" => Field::Key_idle_instances,
                            "latestHistory" => Field::Key_latest_history,
                            "pendingInstances" => Field::Key_pending_instances,
                            "totalInstances" => Field::Key_total_instances,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = ApplicationStatus;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("ApplicationStatus")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_allocated_instances: Option<i32> = None;
                let mut value_deleting_instances: Option<i32> = None;
                let mut value_desired_instances: Option<i32> = None;
                let mut value_history: Option<Vec<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory>> = None;
                let mut value_idle_instances: Option<i32> = None;
                let mut value_latest_history: Option<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory> = None;
                let mut value_pending_instances: Option<i32> = None;
                let mut value_total_instances: Option<i32> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_allocated_instances => value_allocated_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_deleting_instances => value_deleting_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_desired_instances => value_desired_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_history => value_history = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_idle_instances => value_idle_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_latest_history => value_latest_history = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_pending_instances => value_pending_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_total_instances => value_total_instances = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(ApplicationStatus {
                    allocated_instances: value_allocated_instances,
                    deleting_instances: value_deleting_instances,
                    desired_instances: value_desired_instances,
                    history: value_history,
                    idle_instances: value_idle_instances,
                    latest_history: value_latest_history,
                    pending_instances: value_pending_instances,
                    total_instances: value_total_instances,
                })
            }
        }

        deserializer.deserialize_struct(
            "ApplicationStatus",
            &[
                "allocatedInstances",
                "deletingInstances",
                "desiredInstances",
                "history",
                "idleInstances",
                "latestHistory",
                "pendingInstances",
                "totalInstances",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for ApplicationStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "ApplicationStatus",
            self.allocated_instances.as_ref().map_or(0, |_| 1) +
            self.deleting_instances.as_ref().map_or(0, |_| 1) +
            self.desired_instances.as_ref().map_or(0, |_| 1) +
            self.history.as_ref().map_or(0, |_| 1) +
            self.idle_instances.as_ref().map_or(0, |_| 1) +
            self.latest_history.as_ref().map_or(0, |_| 1) +
            self.pending_instances.as_ref().map_or(0, |_| 1) +
            self.total_instances.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.allocated_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "allocatedInstances", value)?;
        }
        if let Some(value) = &self.deleting_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "deletingInstances", value)?;
        }
        if let Some(value) = &self.desired_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "desiredInstances", value)?;
        }
        if let Some(value) = &self.history {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "history", value)?;
        }
        if let Some(value) = &self.idle_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "idleInstances", value)?;
        }
        if let Some(value) = &self.latest_history {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "latestHistory", value)?;
        }
        if let Some(value) = &self.pending_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "pendingInstances", value)?;
        }
        if let Some(value) = &self.total_instances {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "totalInstances", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for ApplicationStatus {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ApplicationStatus".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                description: Some("ApplicationStatus defines the observed state of Application".to_owned()),
                ..Default::default()
            })),
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "allocatedInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of instances which have been started by node".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "deletingInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of instances pending delete and cleanup".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "desiredInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of non-terminated pods targeted".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "history".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Represents the latest available observations of a deployment's current state.".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Array))),
                            array: Some(Box::new(crate::schemars::schema::ArrayValidation {
                                items: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(__gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory>()))),
                                ..Default::default()
                            })),
                            ..Default::default()
                        }),
                    ),
                    (
                        "idleInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of pods which do not have session on it".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "latestHistory".to_owned(),
                        {
                            let mut schema_obj = __gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::DeploymentHistory>().into_object();
                            schema_obj.metadata = Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("The latest deploy history of this app.".to_owned()),
                                ..Default::default()
                            }));
                            crate::schemars::schema::Schema::Object(schema_obj)
                        },
                    ),
                    (
                        "pendingInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of instances pending schedule and implement".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "totalInstances".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("Total number of available instances, including pod not scheduled yet".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int32".to_owned()),
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

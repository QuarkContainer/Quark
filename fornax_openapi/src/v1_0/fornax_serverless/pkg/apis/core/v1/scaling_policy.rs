// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ScalingPolicy

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScalingPolicy {
    pub burst: Option<i64>,

    pub idle_session_num_threshold: Option<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionNumThreshold>,

    pub idle_session_percent_threshold: Option<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionPercentThreshold>,

    pub maximum_instance: Option<i64>,

    pub minimum_instance: Option<i64>,

    /// what session scaling policy to use, absolute num or percent
    pub scaling_policy_type: Option<String>,
}

impl crate::DeepMerge for ScalingPolicy {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.burst, other.burst);
        crate::DeepMerge::merge_from(&mut self.idle_session_num_threshold, other.idle_session_num_threshold);
        crate::DeepMerge::merge_from(&mut self.idle_session_percent_threshold, other.idle_session_percent_threshold);
        crate::DeepMerge::merge_from(&mut self.maximum_instance, other.maximum_instance);
        crate::DeepMerge::merge_from(&mut self.minimum_instance, other.minimum_instance);
        crate::DeepMerge::merge_from(&mut self.scaling_policy_type, other.scaling_policy_type);
    }
}

impl<'de> crate::serde::Deserialize<'de> for ScalingPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_burst,
            Key_idle_session_num_threshold,
            Key_idle_session_percent_threshold,
            Key_maximum_instance,
            Key_minimum_instance,
            Key_scaling_policy_type,
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
                            "burst" => Field::Key_burst,
                            "idleSessionNumThreshold" => Field::Key_idle_session_num_threshold,
                            "idleSessionPercentThreshold" => Field::Key_idle_session_percent_threshold,
                            "maximumInstance" => Field::Key_maximum_instance,
                            "minimumInstance" => Field::Key_minimum_instance,
                            "scalingPolicyType" => Field::Key_scaling_policy_type,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = ScalingPolicy;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("ScalingPolicy")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_burst: Option<i64> = None;
                let mut value_idle_session_num_threshold: Option<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionNumThreshold> = None;
                let mut value_idle_session_percent_threshold: Option<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionPercentThreshold> = None;
                let mut value_maximum_instance: Option<i64> = None;
                let mut value_minimum_instance: Option<i64> = None;
                let mut value_scaling_policy_type: Option<String> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_burst => value_burst = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_idle_session_num_threshold => value_idle_session_num_threshold = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_idle_session_percent_threshold => value_idle_session_percent_threshold = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_maximum_instance => value_maximum_instance = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_minimum_instance => value_minimum_instance = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_scaling_policy_type => value_scaling_policy_type = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(ScalingPolicy {
                    burst: value_burst,
                    idle_session_num_threshold: value_idle_session_num_threshold,
                    idle_session_percent_threshold: value_idle_session_percent_threshold,
                    maximum_instance: value_maximum_instance,
                    minimum_instance: value_minimum_instance,
                    scaling_policy_type: value_scaling_policy_type,
                })
            }
        }

        deserializer.deserialize_struct(
            "ScalingPolicy",
            &[
                "burst",
                "idleSessionNumThreshold",
                "idleSessionPercentThreshold",
                "maximumInstance",
                "minimumInstance",
                "scalingPolicyType",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for ScalingPolicy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "ScalingPolicy",
            self.burst.as_ref().map_or(0, |_| 1) +
            self.idle_session_num_threshold.as_ref().map_or(0, |_| 1) +
            self.idle_session_percent_threshold.as_ref().map_or(0, |_| 1) +
            self.maximum_instance.as_ref().map_or(0, |_| 1) +
            self.minimum_instance.as_ref().map_or(0, |_| 1) +
            self.scaling_policy_type.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.burst {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "burst", value)?;
        }
        if let Some(value) = &self.idle_session_num_threshold {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "idleSessionNumThreshold", value)?;
        }
        if let Some(value) = &self.idle_session_percent_threshold {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "idleSessionPercentThreshold", value)?;
        }
        if let Some(value) = &self.maximum_instance {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "maximumInstance", value)?;
        }
        if let Some(value) = &self.minimum_instance {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "minimumInstance", value)?;
        }
        if let Some(value) = &self.scaling_policy_type {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "scalingPolicyType", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for ScalingPolicy {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.ScalingPolicy".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "burst".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "idleSessionNumThreshold".to_owned(),
                        __gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionNumThreshold>(),
                    ),
                    (
                        "idleSessionPercentThreshold".to_owned(),
                        __gen.subschema_for::<crate::fornax_serverless::pkg::apis::core::v1::IdelSessionPercentThreshold>(),
                    ),
                    (
                        "maximumInstance".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "minimumInstance".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "scalingPolicyType".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("what session scaling policy to use, absolute num or percent".to_owned()),
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

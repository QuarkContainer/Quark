// Generated from definition io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.IdelSessionNumThreshold

/// high watermark should \> low watermark, if both are 0, then no auto scaling for idle buffer, application instance are created on demand when there is no instance to hold a comming session
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdelSessionNumThreshold {
    /// scaling down when idle session more than this number
    pub high: Option<i64>,

    /// scaling up when idle session less than this number
    pub low: Option<i64>,
}

impl crate::DeepMerge for IdelSessionNumThreshold {
    fn merge_from(&mut self, other: Self) {
        crate::DeepMerge::merge_from(&mut self.high, other.high);
        crate::DeepMerge::merge_from(&mut self.low, other.low);
    }
}

impl<'de> crate::serde::Deserialize<'de> for IdelSessionNumThreshold {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: crate::serde::Deserializer<'de> {
        #[allow(non_camel_case_types)]
        enum Field {
            Key_high,
            Key_low,
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
                            "high" => Field::Key_high,
                            "low" => Field::Key_low,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(Visitor)
            }
        }

        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = IdelSessionNumThreshold;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("IdelSessionNumThreshold")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: crate::serde::de::MapAccess<'de> {
                let mut value_high: Option<i64> = None;
                let mut value_low: Option<i64> = None;

                while let Some(key) = crate::serde::de::MapAccess::next_key::<Field>(&mut map)? {
                    match key {
                        Field::Key_high => value_high = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Key_low => value_low = crate::serde::de::MapAccess::next_value(&mut map)?,
                        Field::Other => { let _: crate::serde::de::IgnoredAny = crate::serde::de::MapAccess::next_value(&mut map)?; },
                    }
                }

                Ok(IdelSessionNumThreshold {
                    high: value_high,
                    low: value_low,
                })
            }
        }

        deserializer.deserialize_struct(
            "IdelSessionNumThreshold",
            &[
                "high",
                "low",
            ],
            Visitor,
        )
    }
}

impl crate::serde::Serialize for IdelSessionNumThreshold {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: crate::serde::Serializer {
        let mut state = serializer.serialize_struct(
            "IdelSessionNumThreshold",
            self.high.as_ref().map_or(0, |_| 1) +
            self.low.as_ref().map_or(0, |_| 1),
        )?;
        if let Some(value) = &self.high {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "high", value)?;
        }
        if let Some(value) = &self.low {
            crate::serde::ser::SerializeStruct::serialize_field(&mut state, "low", value)?;
        }
        crate::serde::ser::SerializeStruct::end(state)
    }
}

#[cfg(feature = "schemars")]
impl crate::schemars::JsonSchema for IdelSessionNumThreshold {
    fn schema_name() -> String {
        "io.centaurusinfra.fornax-serverless.pkg.apis.core.v1.IdelSessionNumThreshold".to_owned()
    }

    fn json_schema(__gen: &mut crate::schemars::gen::SchemaGenerator) -> crate::schemars::schema::Schema {
        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                description: Some("high watermark should > low watermark, if both are 0, then no auto scaling for idle buffer, application instance are created on demand when there is no instance to hold a comming session".to_owned()),
                ..Default::default()
            })),
            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Object))),
            object: Some(Box::new(crate::schemars::schema::ObjectValidation {
                properties: [
                    (
                        "high".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("scaling down when idle session more than this number".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
                            ..Default::default()
                        }),
                    ),
                    (
                        "low".to_owned(),
                        crate::schemars::schema::Schema::Object(crate::schemars::schema::SchemaObject {
                            metadata: Some(Box::new(crate::schemars::schema::Metadata {
                                description: Some("scaling up when idle session less than this number".to_owned()),
                                ..Default::default()
                            })),
                            instance_type: Some(crate::schemars::schema::SingleOrVec::Single(Box::new(crate::schemars::schema::InstanceType::Integer))),
                            format: Some("int64".to_owned()),
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

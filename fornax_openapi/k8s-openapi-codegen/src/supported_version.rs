pub(crate) const ALL: &[SupportedVersion] = &[
    SupportedVersion::V1_0,
];

#[derive(Clone, Copy, Debug)]
pub(crate) enum SupportedVersion {
    V1_0,
}

impl SupportedVersion {
    pub(crate) fn name(self) -> &'static str {
        match self {
            SupportedVersion::V1_0 => "1.0",
        }
    }

    pub(crate) fn mod_root(self) -> &'static str {
        match self {
            SupportedVersion::V1_0 => "v1_0",
        }
    }

    pub(crate) fn spec_url(self) -> &'static str {
        match self {
			SupportedVersion::V1_0 => "https://raw.githubusercontent.com/CentaurusInfra/fornax-serverless/main/config/swagger.json",
		}
    }

    pub(crate) fn fixup(self, spec: &mut crate::swagger20::Spec) -> Result<(), crate::Error> {
        #[allow(clippy::match_same_arms, clippy::type_complexity)]
		let upstream_bugs_fixups: &[fn(&mut crate::swagger20::Spec) -> Result<(), crate::Error>] = match self {

			SupportedVersion::V1_0 => &[
				// crate::fixups::upstream_bugs::connect_options_gvk,
				// crate::fixups::upstream_bugs::pod_exec_command_parameter_type,
				// crate::fixups::upstream_bugs::required_properties::validating_admission_policy_binding_list,
				// crate::fixups::upstream_bugs::required_properties::validating_admission_policy_list,
				// crate::fixups::upstream_bugs::status_extra_gvk,
			],
		};

        let special_fixups = &[
            // crate::fixups::special::json_ty::json_schema_props_or_array,
            // crate::fixups::special::json_ty::json_schema_props_or_bool,
            // crate::fixups::special::json_ty::json_schema_props_or_string_array,
            crate::fixups::special::create_delete_optional,
            crate::fixups::special::create_optionals,
            crate::fixups::special::patch,
            crate::fixups::special::remove_delete_collection_operations_query_parameters,
            crate::fixups::special::remove_delete_operations_query_parameters,
            crate::fixups::special::remove_read_operations_query_parameters,
            crate::fixups::special::separate_watch_from_list_operations,
            crate::fixups::special::watch_event,
            crate::fixups::special::list, // Must run after separate_watch_from_list_operations
            crate::fixups::special::response_types,
            crate::fixups::special::resource_metadata_not_optional,
        ];

        for fixup in upstream_bugs_fixups.iter().chain(special_fixups) {
            fixup(spec)?;
        }

        Ok(())
    }
}

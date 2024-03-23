// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::*;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    pub static ref QUALIFIED_NAME_FMT: String =
        "(".to_owned() + QNAME_CHAR_FMT + QNAME_EXT_CHAR_FMT + "*)?" + QNAME_CHAR_FMT;
    pub static ref QUALIFIED_NAME_REGEXP: Regex =
        Regex::new(&("^".to_owned() + &QUALIFIED_NAME_FMT + "$")).unwrap();
}

pub const QNAME_CHAR_FMT: &str = "[A-Za-z0-9]";
pub const QNAME_EXT_CHAR_FMT: &str = "[-A-Za-z0-9_.]";
pub const QUALIFIED_NAME_ERR_MSG : &str = "must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character";
pub const QUALIFIED_NAME_MAX_LENGTH: usize = 63;
pub const LABEL_VALUE_ERR_MSG : &str = "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character";

// IsQualifiedName tests whether the value passed is what Kubernetes calls a
// "qualified name".  This is a format used in various places throughout the
// system.  If the value is not valid, a list of error strings is returned.
// Otherwise an empty list (or nil) is returned.
pub fn IsQualifiedName(value: &str) -> Result<()> {
    let parts: Vec<&str> = value.split("/").collect();

    let (_prefix, name) = match parts.len() {
        1 => ("", parts[0]),
        2 => {
            if parts[0].len() == 0 {
                return Err(Error::CommonError("Prefix must be non empty".to_owned()));
            }

            IsDNS1123Subdomain(parts[0])?;
            (parts[0], parts[1])
        }
        _ => {
            return Err(Error::CommonError(format!(
                "'{}' is not a qualified name",
                value
            )));
        }
    };

    if name.len() == 0 {
        return Err(Error::CommonError("Name must be non empty".to_owned()));
    }

    if name.len() > QUALIFIED_NAME_MAX_LENGTH {
        return Err(Error::CommonError(
            "Name len must be less than 64".to_owned(),
        ));
    }

    if !QUALIFIED_NAME_REGEXP.is_match(name) {
        return Err(Error::CommonError("Name is not legal".to_owned()));
    }

    return Ok(());
}

pub fn IsFullyQualifiedDomainName(name: &str) -> Result<()> {
    let mut name = name;
    if name.len() == 0 {
        return Err(Error::CommonError("Name must be non empty".to_owned()));
    }

    if name.ends_with(".") {
        name = &name[0..name.len() - 1];
    }

    IsDNS1123Subdomain(name)?;

    let parts: Vec<&str> = name.split(".").collect();
    if parts.len() < 2 {
        return Err(Error::CommonError(
            "should be a domain with at least two segments separated by dots".to_owned(),
        ));
    }

    for label in parts {
        IsDNS1123Label(label)?;
    }

    return Ok(());
}

pub const HTTP_PATH_FMT: &str = r"[A-Za-z0-9/\-._~%!$&'()*+,;=:]+";

lazy_static! {
    pub static ref HTTP_PATH_REGEXP: Regex =
        Regex::new(&("^".to_owned() + &HTTP_PATH_FMT + "$")).unwrap();
}

// IsDomainPrefixedPath checks if the given string is a domain-prefixed path
// (e.g. acme.io/foo). All characters before the first "/" must be a valid
// subdomain as defined by RFC 1123. All characters trailing the first "/" must
// be valid HTTP Path characters as defined by RFC 3986.
pub fn IsDomainPrefixedPath(dpPath: &str) -> Result<()> {
    if dpPath.len() == 0 {
        return Err(Error::CommonError("Name must be non empty".to_owned()));
    }

    let segments: Vec<&str> = dpPath.splitn(2, "/").collect();

    if segments.len() != 2 || segments[0].len() == 0 || segments[1].len() == 0 {
        return Err(Error::CommonError(
            "must be a domain-prefixed path (such as \"acme.io/foo\"".to_owned(),
        ));
    }

    let host = segments[0];
    IsDNS1123Subdomain(host)?;

    let path = segments[1];
    if !HTTP_PATH_REGEXP.is_match(path) {
        return Err(Error::CommonError("Invalid path: ".to_owned() + path));
    }

    return Ok(());
}

// IsValidLabelValue tests whether the value passed is a valid label value.  If
// the value is not valid, a list of error strings is returned.  Otherwise an
// empty list (or nil) is returned.
pub fn IsValidLabelValue(val: &str) -> Result<()> {
    if val.len() > QUALIFIED_NAME_MAX_LENGTH {
        return Err(Error::CommonError(format!(
            "must be no more than {} characters, actual {}",
            QUALIFIED_NAME_MAX_LENGTH,
            val.len()
        )));
    }

    if !QUALIFIED_NAME_REGEXP.is_match(val) {
        return Err(Error::CommonError(format!(
            "{} actual:{}",
            LABEL_VALUE_ERR_MSG, val
        )));
    }
    return Ok(());
}

pub const DNS1123_LABEL_FMT: &str = "[a-z0-9]([-a-z0-9]*[a-z0-9])?";
pub const DNS1123_LABEL_ERR_MSG : &str = "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character";

// DNS1123LabelMaxLength is a label's max length in DNS (RFC 1123)
pub const DNS1123_LABEL_MAX_LENGTH: usize = 63;

lazy_static! {
    pub static ref DNS1123_LABEL_REGEXP: Regex =
        Regex::new(&("^".to_owned() + DNS1123_LABEL_FMT + "$")).unwrap();
}

// IsDNS1123Label tests for a string that conforms to the definition of a label in
// DNS (RFC 1123).
pub fn IsDNS1123Label(value: &str) -> Result<()> {
    if value.len() > QUALIFIED_NAME_MAX_LENGTH {
        return Err(Error::CommonError(
            "DNS1123Label len must be less than 64".to_owned(),
        ));
    }

    if !DNS1123_LABEL_REGEXP.is_match(value) {
        // It was a valid subdomain and not a valid label.  Since we
        // already checked length, it must be dots.
        if DNS1123_SUBDOMAIN_REGEXP.is_match(value) {
            return Err(Error::CommonError(
                "DNS1123Label must not contain dots".to_owned(),
            ));
        } else {
            return Err(Error::CommonError(DNS1123_LABEL_ERR_MSG.to_string()));
        }
    }
    return Ok(());
}

pub const DNS1123_SUBDOMAIN_ERROR_MSG : &str = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character";
// DNS1123SubdomainMaxLength is a subdomain's max length in DNS (RFC 1123)
pub const DNS1123_SUBDOMAIN_MAX_LENGTH: usize = 253;

lazy_static! {
    pub static ref DNS1123_SUBDOMAIN_FMT: String =
        DNS1123_LABEL_FMT.to_string() + "(\\." + DNS1123_LABEL_FMT + ")*";
    pub static ref DNS1123_SUBDOMAIN_REGEXP: Regex =
        Regex::new(&("^".to_owned() + &DNS1123_SUBDOMAIN_FMT + "$")).unwrap();
}

// IsDNS1123Subdomain tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123).
pub fn IsDNS1123Subdomain(value: &str) -> Result<()> {
    if value.len() > DNS1123_SUBDOMAIN_MAX_LENGTH {
        return Err(Error::CommonError(
            "DNS1123Subdomain len must be less than 254".to_owned(),
        ));
    }

    if !DNS1123_SUBDOMAIN_REGEXP.is_match(value) {
        return Err(Error::CommonError(DNS1123_SUBDOMAIN_ERROR_MSG.to_string()));
    }

    return Ok(());
}

pub const DNS1035_LABEL_FMT: &str = "[a-z]([-a-z0-9]*[a-z0-9])?";
pub const DNS1035_LABEL_ERR_MSG : &str = "a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character";

// DNS1035LabelMaxLength is a label's max length in DNS (RFC 1035)
pub const DNS1035_LABEL_MAX_LENGTH: usize = 63;

lazy_static! {
    pub static ref DNS1035_LABEL_REGEXP: Regex =
        Regex::new(&("^".to_owned() + DNS1035_LABEL_FMT + "$")).unwrap();
}

// IsDNS1035Label tests for a string that conforms to the definition of a label in
// DNS (RFC 1035).

pub fn IsDNS1035Label(value: &str) -> Result<()> {
    if value.len() > DNS1035_LABEL_MAX_LENGTH {
        return Err(Error::CommonError(
            "DNS1035 Label len must be less than 64".to_owned(),
        ));
    }

    if !DNS1035_LABEL_REGEXP.is_match(value) {
        return Err(Error::CommonError(DNS1035_LABEL_ERR_MSG.to_string()));
    }

    return Ok(());
}

// wildcard definition - RFC 1034 section 4.3.3.
// examples:
// - valid: *.bar.com, *.foo.bar.com
// - invalid: *.*.bar.com, *.foo.*.com, *bar.com, f*.bar.com, *

lazy_static! {
    pub static ref WILDCARD_DNS1123_SUBDOMAIN_FMT: String =
        "\\*\\.".to_string() + DNS1123_LABEL_FMT;
}

pub const WILDCARD_DNS1123_SUBDOMAIN_ERR_MSG : &str = "a wildcard DNS-1123 subdomain must start with '*.', followed by a valid DNS subdomain, which must consist of lower case alphanumeric characters, '-' or '.' and end with an alphanumeric character";

// IsWildcardDNS1123Subdomain tests for a string that conforms to the definition of a
// wildcard subdomain in DNS (RFC 1034 section 4.3.3).
pub fn IsWildcardDNS1123Subdomain(value: &str) -> Result<()> {
    let wildcardDNS1123SubdomainRegexp =
        Regex::new(&("^".to_owned() + &WILDCARD_DNS1123_SUBDOMAIN_FMT + "$")).unwrap();
    if value.len() > DNS1123_SUBDOMAIN_MAX_LENGTH {
        return Err(Error::CommonError(
            "wildcard DNS1123 Subdomain len must be less than 254".to_owned(),
        ));
    }

    if !wildcardDNS1123SubdomainRegexp.is_match(value) {
        return Err(Error::CommonError(
            WILDCARD_DNS1123_SUBDOMAIN_ERR_MSG.to_string(),
        ));
    }
    return Ok(());
}

pub const C_IDENTIFIER_FMT: &str = "[A-Za-z_][A-Za-z0-9_]*";
pub const IDENTIFIER_ERR_MSG : &str = "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_'";

lazy_static! {
    pub static ref C_IDENTIFIER_REGEXP: Regex =
        Regex::new(&("^".to_owned() + C_IDENTIFIER_FMT + "$")).unwrap();
}

// IsCIdentifier tests for a string that conforms the definition of an identifier
// in C. This checks the format, but not the length.
pub fn IsCIdentifier(value: &str) -> Result<()> {
    if !C_IDENTIFIER_REGEXP.is_match(value) {
        return Err(Error::CommonError(IDENTIFIER_ERR_MSG.to_string()));
    }

    return Ok(());
}

// IsValidPortNum tests that the argument is a valid, non-zero port number.
pub fn IsValidPortNum(port: i32) -> Result<()> {
    if 1 <= port && port <= 65535 {
        return Ok(());
    }

    return Err(InclusiveRangeError(1, 65535));
}

// IsInRange tests that the argument is in an inclusive range.
pub fn IsInRange(value: i32, min: i32, max: i32) -> Result<()> {
    if value >= min && value <= max {
        return Ok(());
    }

    return Err(InclusiveRangeError(min, max));
}

// Now in libcontainer UID/GID limits is 0 ~ 1<<31 - 1
// TODO: once we have a type for UID/GID we should make these that type.
pub const MIN_USER_ID: i32 = 0;
pub const MAX_USER_ID: i32 = i32::MAX;
pub const MIN_GROUP_ID: i32 = 0;
pub const MAX_GROUP_ID: i32 = i32::MAX;

// IsValidGroupID tests that the argument is a valid Unix GID.
pub fn IsValidGroupID(gid: i64) -> Result<()> {
    if MIN_GROUP_ID as i64 <= gid && gid <= MAX_GROUP_ID as i64 {
        return Ok(());
    }

    return Err(InclusiveRangeError(MIN_GROUP_ID, MAX_GROUP_ID));
}

// IsValidUserID tests that the argument is a valid Unix UID.
pub fn IsValidUserID(uid: i64) -> Result<()> {
    if MIN_USER_ID as i64 <= uid && uid <= MAX_USER_ID as i64 {
        return Ok(());
    }

    return Err(InclusiveRangeError(MIN_USER_ID, MAX_USER_ID));
}

lazy_static! {
    pub static ref PORT_NAME_CHARSET_REGEX: Regex = Regex::new("^[-a-z0-9]+$").unwrap();
    pub static ref PORT_NAME_ONE_LETTER_REGEXP: Regex = Regex::new("[a-z]").unwrap();
}

// IsValidPortName check that the argument is valid syntax. It must be
// non-empty and no more than 15 characters long. It may contain only [-a-z0-9]
// and must contain at least one letter [a-z]. It must not start or end with a
// hyphen, nor contain adjacent hyphens.
//
// Note: We only allow lower-case characters, even though RFC 6335 is case
// insensitive.
pub fn IsValidPortName(port: &str) -> Result<()> {
    if port.len() > 15 {
        return Err(Error::CommonError(
            "port len must be less than 16".to_owned(),
        ));
    }

    if !PORT_NAME_CHARSET_REGEX.is_match(port) {
        return Err(Error::CommonError(
            "must contain only alpha-numeric characters (a-z, 0-9), and hyphens (-)".to_owned(),
        ));
    }

    if !PORT_NAME_ONE_LETTER_REGEXP.is_match(port) {
        return Err(Error::CommonError(
            "must contain at least one letter (a-z)".to_owned(),
        ));
    }

    if port.contains("--") {
        return Err(Error::CommonError(
            "must not contain consecutive hyphens".to_owned(),
        ));
    }

    let chars: Vec<_> = port.chars().collect();
    if chars.len() > 0 && (chars[0] == '-' || chars[chars.len() - 1] == '-') {
        return Err(Error::CommonError(
            "must not begin or end with a hyphen".to_owned(),
        ));
    }

    return Ok(());
}

// IsValidIP tests that the argument is a valid IP address.
pub fn IsValidIP(value: &str) -> Result<()> {
    use std::net::IpAddr;
    match value.parse::<IpAddr>() {
        Err(_) => return Err(Error::CommonError("must be a valid IP address".to_owned())),
        Ok(_addr) => return Ok(()),
    }
}

// IsValidIPv4Address tests that the argument is a valid IPv4 address.
pub fn IsValidIPv4Address(value: &str) -> Result<()> {
    use std::net::IpAddr;
    match value.parse::<IpAddr>() {
        Err(_) => {
            return Err(Error::CommonError(
                "must be a valid IPv4 address".to_owned(),
            ))
        }
        Ok(addr) => {
            if !addr.is_ipv4() {
                return Err(Error::CommonError(
                    "must be a valid IPv4 address".to_owned(),
                ));
            }
        }
    }
    return Ok(());
}

// IsValidIPv6Address tests that the argument is a valid IPv6 address.
pub fn IsValidIPv6Address(value: &str) -> Result<()> {
    use std::net::IpAddr;
    match value.parse::<IpAddr>() {
        Err(_) => {
            return Err(Error::CommonError(
                "must be a valid IPv6 address".to_owned(),
            ))
        }
        Ok(addr) => {
            if !addr.is_ipv6() {
                return Err(Error::CommonError(
                    "must be a valid IPv6 address".to_owned(),
                ));
            }
        }
    }
    return Ok(());
}

pub const PERCENT_FMT: &str = "[0-9]+%";
pub const PERCENT_ERR_MSG: &str =
    "a valid percent string must be a numeric string followed by an ending '%'";

lazy_static! {
    pub static ref PERCENT_REGEXP: Regex =
        Regex::new(&("^".to_owned() + PERCENT_FMT + "$")).unwrap();
}

// IsValidPercent checks that string is in the form of a percentage
pub fn IsValidPercent(value: &str) -> Result<()> {
    if !PERCENT_REGEXP.is_match(value) {
        return Err(Error::CommonError(PERCENT_ERR_MSG.to_owned()));
    }

    return Ok(());
}

pub const HTTP_HEADER_NAME_FMT: &str = "[-A-Za-z0-9]+";
pub const HTTP_HEADER_NAME_ERR_MSG: &str =
    "a valid HTTP header must consist of alphanumeric characters or '-'";

lazy_static! {
    pub static ref HTTP_HEADER_NAME_REGEXP: Regex =
        Regex::new(&("^".to_owned() + HTTP_HEADER_NAME_FMT + "$")).unwrap();
}

// IsHTTPHeaderName checks that a string conforms to the Go HTTP library's
// definition of a valid header field name (a stricter subset than RFC7230).
pub fn IsHTTPHeaderName(value: &str) -> Result<()> {
    if !HTTP_HEADER_NAME_REGEXP.is_match(value) {
        return Err(Error::CommonError(HTTP_HEADER_NAME_ERR_MSG.to_owned()));
    }

    return Ok(());
}

pub const ENV_VAR_NAME_FMT: &str = "[-._a-zA-Z][-._a-zA-Z0-9]*";
pub const ENV_VAR_NAME_FMT_ERR_MSG : &str = "a valid environment variable name must consist of alphabetic characters, digits, '_', '-', or '.', and must not start with a digit";

lazy_static! {
    pub static ref ENV_VAR_NAME_REGEXP: Regex =
        Regex::new(&("^".to_owned() + ENV_VAR_NAME_FMT + "$")).unwrap();
}

// IsEnvVarName tests if a string is a valid environment variable name.
pub fn IsEnvVarName(value: &str) -> Result<()> {
    if !ENV_VAR_NAME_REGEXP.is_match(value) {
        return Err(Error::CommonError(ENV_VAR_NAME_FMT_ERR_MSG.to_owned()));
    }

    return Ok(());
}

pub const CONFIG_MAP_KEY_FMT: &str = "[-._a-zA-Z][-._a-zA-Z0-9]*";
pub const CONFIG_MAP_KEY_ERR_MSG : &str = "a valid environment variable name must consist of alphabetic characters, digits, '_', '-', or '.', and must not start with a digit";

lazy_static! {
    pub static ref CONFIG_MAP_KEY_REGEXP: Regex =
        Regex::new(&("^".to_owned() + CONFIG_MAP_KEY_FMT + "$")).unwrap();
}

// IsConfigMapKey tests for a string that is a valid key for a ConfigMap or Secret
pub fn IsConfigMapKey(value: &str) -> Result<()> {
    if value.len() > 253 {
        return Err(Error::CommonError(
            "port len must be less than 254".to_owned(),
        ));
    }

    if !CONFIG_MAP_KEY_REGEXP.is_match(value) {
        return Err(Error::CommonError(CONFIG_MAP_KEY_ERR_MSG.to_owned()));
    }

    return Ok(());
}

pub fn InclusiveRangeError(lo: i32, hi: i32) -> Error {
    return Error::CommonError(format!("must be between {} and {}, inclusive", lo, hi));
}

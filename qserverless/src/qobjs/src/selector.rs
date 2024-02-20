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

use std::collections::{BTreeMap, BTreeSet, HashSet};
use lazy_static::lazy_static;
use serde_json::Value;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::cmp::Ord;
use std::sync::Arc;
use core::ops::Deref;
use serde::{Deserialize, Serialize};

use crate::qmeta::*;
use crate::common::*;
use crate::types::DeepCopy;

use super::validation::*;

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionOp {
    None,           // ""
    DoesNotExist,   // "!"
	Equals,         // "="
	DoubleEquals,   // "=="
	In,             // "in"
	NotEquals,      // "!="
	NotIn,          // "notin"
	Exists,         // "exists"
	GreaterThan,    // "gt"
	LessThan,       // "lt"
}

impl Default for SelectionOp {
    fn default() -> Self {
        return Self::DoesNotExist
    }
}

impl SelectionOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => return "",
            Self::DoesNotExist => return "!",
            Self::Equals => return "=",
            Self::DoubleEquals => return "==",
            Self::In => return "in",
            Self::NotEquals => return "!=",
            Self::NotIn => return "notin",
            Self::Exists => return "exists",
            Self::GreaterThan => return "gt",
            Self::LessThan => return "lt",
        }
    }
}


#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Selector(pub Vec<Requirement>);

impl Selector {
    pub fn Parse(selector: &str) -> Result<Self> {
        return Parse(selector)
    }

    pub fn Equ(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        for i in 0..self.0.len() {
            if !self.0[i].Equ(&other.0[i]) {
                return false;
            }
        }

        return true;
    }

    pub fn DeepCopy(&self) -> Self{
        let mut copy = Self::default();
        for r in &self.0 {
            copy.0.push(r.Copy());
        }

        return copy;
    }

    pub fn Sort(&mut self) {
        self.0.sort();
    }

    pub fn Add(&mut self, r: Requirement) {
        self.0.push(r);
    }

    // Matches for a internalSelector returns true if all
    // its Requirements match the input Labels. If any
    // Requirement does not match, false is returned.
    pub fn Match(&self, l: &Labels) -> bool {
        for r in &self.0 {
            if !r.Matchs(l) {
                return false;
            }
        }

        return true;
    }

    pub fn ToString(val: &Value) -> String {
        match val {
            Value::Null => "".to_string(),
            Value::Bool(boolean) => format!("{}", boolean),
            Value::Number(number) => format!("{}", number),
            Value::String(string) => format!("{}", string),
            Value::Array(vec) => format!("{:?}", vec),
            Value::Object(map) => format!("{:?}", map),
        }
    }

    pub fn GetAttributes(&self, val: &serde_json::Value) -> Option<BTreeMap<String, String>> {
        let mut map = BTreeMap::new();
        for r in &self.0 {
            let split = r.key.split(".");
            let mut tmp = val;
            for s in split {
                tmp = match tmp.get(s) {
                    None => {
                        return None
                    },
                    Some(v) => {
                        &v
                    }
                };
            }

            let str = Self::ToString(tmp);
            map.insert(r.key.clone(), str);
        }

        return Some(map);
    }

    // String returns a comma-separated string of all
    // the internalSelector Requirements' human-readable strings.
    pub fn String(&self) -> String {
        let mut reqs : Vec<String> = Vec::new();
        for r in &self.0 {
            reqs.push(r.String());
        }

        return reqs.join(",")
    }

    // RequiresExactMatch introspects whether a given selector requires a single specific field
    // to be set, and if so returns the value it requires.
    pub fn RequiresExactMatch(&self, label: &str) -> Option<String> {
        for r in &self.0 {
            if &r.key == label {
                match r.op {
                    SelectionOp::Equals | SelectionOp::DoubleEquals | SelectionOp::In => {
                        if r.strVals.len() == 1 {
                            return Some(r.strVals[0].clone());
                        }
                    }
                    _ => return None,
                }
            } else {
                return None;
            }
        }

        return None;
    }

    pub fn Empty(&self) -> bool {
        return self.0.len() == 0;
    }
}

fn GetRequirement(key: &str, op: SelectionOp, vals: Vec<String>) -> Requirement {
    let req = match Requirement::New(key, op, vals) {
        Err(_e) => {
            //assert!(false, "error is {:?}", e);
            Requirement::default()
        }
        Ok(r) => r,
    };

    return req;
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Requirement {
    pub key: String,
    pub op: SelectionOp,
    pub strVals: Vec<String>
}

impl Ord for Requirement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for Requirement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Requirement {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Requirement {

}

impl Requirement {
    pub fn Equ(&self, other: &Self) -> bool {
        if self.strVals.len() != other.strVals.len() {
            return false;
        }

        let mut set : BTreeSet<String> = BTreeSet::new();

        for str in &self.strVals {
            set.insert(str.to_string());
        }

        for str in &other.strVals {
            if !set.contains(str) {
                return false;
            }
        }

        return self.key == other.key && self.op == other.op;
    }

    pub fn Copy(&self) -> Self {
        let mut out = Self {
            key: self.key.clone(),
            op: self.op,
            strVals: Vec::new(),
        };

        for str in &self.strVals {
            out.strVals.push(str.clone());
        }

        return out;
    }

    // If any of these rules is violated, an error is returned:
    //  1. The operator can only be In, NotIn, Equals, DoubleEquals, Gt, Lt, NotEquals, Exists, or DoesNotExist.
    //  2. If the operator is In or NotIn, the values set must be non-empty.
    //  3. If the operator is Equals, DoubleEquals, or NotEquals, the values set must contain one value.
    //  4. If the operator is Exists or DoesNotExist, the value set must be empty.
    //  5. If the operator is Gt or Lt, the values set must contain only one value, which will be interpreted as an integer.
    //  6. The key is invalid due to its length, or sequence of characters. See validateLabelKey for more details.
    pub fn New(key: &str, op: SelectionOp, vals: Vec<String>) -> Result<Requirement> {
        ValidateLabelKey(key)?;
        match op {
            SelectionOp::None => panic!("selector::None"),
            SelectionOp::In | SelectionOp::NotIn => {
                if vals.len() == 0 {
                    return Err(Error::CommonError("for 'in', 'notin' operators, values set can't be empty".to_owned()))
                }
            }
            SelectionOp::Equals | SelectionOp::DoubleEquals | SelectionOp::NotEquals => {
                if vals.len() != 1 {
                    return Err(Error::CommonError("exact-match compatibility requires one single value".to_owned()))
                }
            }
            SelectionOp::Exists | SelectionOp::DoesNotExist => {
                if vals.len() != 0 {
                    return Err(Error::CommonError("values set must be empty for exists and does not exist".to_owned()))
                }
            }
            SelectionOp::GreaterThan | SelectionOp::LessThan => {
                if vals.len() != 1 {
                    return Err(Error::CommonError("for 'Gt', 'Lt' operators, exactly one value is required".to_owned()))
                }

                for val in &vals {
                    match val.parse::<u64>() {
                        Err(_) => {
                            return Err(Error::CommonError("for 'Gt', 'Lt' operators, the value must be an integer".to_owned()))
                        }
                        _ => {}
                    }
                }

            }
        }

        return Ok(Requirement { key: key.to_owned(), op: op, strVals: vals })
    }

    pub fn HasValue(&self, val: &str) -> bool {
        for str in &self.strVals {
            if str == val {
                return true
            }
        }

        return false;
    }

    // Matches returns true if the Requirement matches the input Labels.
    // There is a match in the following cases:
    //  1. The operator is Exists and Labels has the Requirement's key.
    //  2. The operator is In, Labels has the Requirement's key and Labels'
    //     value for that key is in Requirement's value set.
    //  3. The operator is NotIn, Labels has the Requirement's key and
    //     Labels' value for that key is not in Requirement's value set.
    //  4. The operator is DoesNotExist or NotIn and Labels does not have the
    //     Requirement's key.
    //  5. The operator is GreaterThanOperator or LessThanOperator, and Labels has
    //     the Requirement's key and the corresponding value satisfies mathematical inequality.
    pub fn Matchs(&self, ls: &Labels) -> bool {
        match self.op {
            SelectionOp::None => panic!("selector::None"),
            SelectionOp::In | SelectionOp::Equals | SelectionOp::DoubleEquals => {
                let val = match ls.Get(&self.key) {
                    None => return false,
                    Some(v) => v,
                };
                return self.HasValue(&val);
            }
            SelectionOp::NotIn | SelectionOp::NotEquals => {
                let val = match ls.Get(&self.key) {
                    None => return true,
                    Some(v) => v,
                };

                return !self.HasValue(&val);
            }
            SelectionOp::Exists => {
                return ls.Has(&self.key);
            }
            SelectionOp::DoesNotExist => {
                return !ls.Has(&self.key);
            }
            SelectionOp::GreaterThan | SelectionOp::LessThan => {
                if !ls.Has(&self.key) {
                    return false;
                }

                let val = match ls.Get(&self.key) {
                    None => {
                        return false;
                    }
                    Some(v) => v,
                };


                let lsValue: i64 = match val.parse() {
                    Err(_) => {
                        error!("ParseInt failed for value {} in label {:?}", val, ls);
                        return false;
                    }
                    Ok(v) => v,
                };

                if self.strVals.len() != 1 {
                    error!("Invalid values count {} of requirement {:?}, for 'Gt', 'Lt' operators, exactly one value is required", self.strVals.len(), self);
                    return false;
                }

                for val in &self.strVals {
                    let rValue : i64 = match val.parse() {
                        Err(_) => {
                            error!("ParseInt failed for value {} in requirement {:?}, for 'Gt', 'Lt' operators, the value must be an integer", val, self);
                            return false;
                        }
                        Ok(v) => v
                    };

                    if self.op == SelectionOp::GreaterThan {
                        return lsValue > rValue;
                    }

                    //if self.op == SelectionOp::LessThan {
                        return lsValue < rValue;
                    //}
                }

                // won't reach
                return false;
            }
        }
    }

    // Key returns requirement key
    pub fn Key(&self) -> String {
        return self.key.clone();
    }

    // Operator returns requirement operator
    pub fn Operator(&self) -> SelectionOp {
        return self.op;
    }

    // Values returns requirement values
    pub fn Values(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        for v in &self.strVals {
            set.insert(v.clone());
        }

        return set;
    }

    // Equal checks the equality of requirement.
    pub fn Equal(&self, x: &Self) -> bool {
        if self.key !=  x.key {
            return false;
        }

        if self.op != x.op {
            return false;
        }

        if self.strVals.len() != x.strVals.len() {
            return false;
        }

        for i in 0..self.strVals.len() {
            if self.strVals[i] != self.strVals[i] {
                return false
            }
        }
        
        return true;
    }

    // String returns a human-readable string that represents this
    // Requirement. If called on an invalid Requirement, an error is
    // returned. See NewRequirement for creating a valid Requirement.
    pub fn String(&self) -> String {
        let mut output = "".to_owned();

        if self.op == SelectionOp::DoesNotExist {
            output = output + "!";
        }

        output = output + &self.key;

        match self.op {
            SelectionOp::DoesNotExist | SelectionOp::Exists => { 
                return output;
            }
            SelectionOp::Equals => output = output + "=",
            SelectionOp::DoubleEquals => output = output + "==",
            SelectionOp::In => output = output + " in ",
            SelectionOp::NotEquals => output = output + "!=",
            SelectionOp::NotIn => output = output + " notin ",
            SelectionOp::GreaterThan => output = output + ">",
            SelectionOp::LessThan => output = output + "<",
            _ => {}
        }

        if self.op == SelectionOp::In || self.op == SelectionOp::NotIn {
            output = output + "(";
        }

        let values = self.Values();
        let mut first = true;
        for v in &values {
            if first {
                first = false;
            } else {
                output = output + ",";
            }

            output = output + v;
        }

        if self.op == SelectionOp::In || self.op == SelectionOp::NotIn {
            output = output + ")";
        }

        return output;
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Labels(Arc<BTreeMap<String, String>>);

impl From<BTreeMap<String, String>> for Labels {
    fn from(item: BTreeMap<String, String>) -> Self {
        return Self(Arc::new(item));
    }
}

impl Deref for Labels {
    type Target = Arc<BTreeMap<String, String>>;

    fn deref(&self) -> &Arc<BTreeMap<String, String>> {
        &self.0
    }
}

impl DeepCopy for Labels {
    fn DeepCopy(&self) -> Self {
        return self.Copy();
    }
}

impl Labels {
    pub fn NewFromMap(map: BTreeMap<String, String>) -> Self {
        return Self(Arc::new(map));
    }

    pub fn Copy(&self) -> Self {
        let mut map = BTreeMap::new();
        for (k, v) in self.as_ref() {
            map.insert(k.clone(), v.clone());
        }

        return map.into();
    }

    pub fn ToVec(&self) -> Vec<Kv> {
        let mut ret = Vec::with_capacity(self.0.len());
        for (k, v) in self.0.as_ref() {
            let kv = Kv {
                key: k.clone(),
                val: v.clone(),
            };
            ret.push(kv);
        }

        return ret;
    }

    pub fn NewFromSlice(item: &[(String, String)]) -> Self {
        let mut map = BTreeMap::new();
        for (k, v) in item {
            map.insert(k.clone(), v.clone());
        }
        return map.into();
    }

    // ConvertSelectorToLabelsMap converts selector string to labels map
    // and validates keys and values
    pub fn New(selector: &str) -> Result<Self> {
        let mut map = BTreeMap::new();

        if selector.len() == 0 {
            return Ok(map.into());
        }

        let labels : Vec<&str> = selector.split(",").collect();
        for label in labels {
            let l : Vec<&str> = label.split("=").collect();
            if l.len() != 2 {
                return Err(Error::CommonError(format!("invalid selector: {}", label)));
            }

            let key = l[0].trim();
            ValidateLabelKey(key)?;

            let value = l[1].trim();
            ValidateLabelValue(key, value)?;

            map.insert(key.to_string(), value.to_string());
        }

        return Ok(map.into())
    }

    // String returns all labels listed as a human readable string.
    // Conveniently, exactly the format that ParseSelector takes.
    pub fn String(&self) -> String {
        let mut ret = "".to_owned();
        for (k, v) in self.as_ref() {
            if ret.len() != 0 {
                ret = ret + ",";
            }

            ret = ret + k + "=" + v;
        }

        return ret;
    }

    // Has returns whether the provided label exists in the map.
    pub fn Has(&self, label: &str) -> bool {
        return self.0.contains_key(label);
    }

    // Get returns the value in the map for the provided label.
    pub fn Get(&self, label: &str) -> Option<String> {
        match self.0.get(label) {
            None => return None,
            Some(v) => return Some(v.to_string()),
        }
    }

    // FormatLabels converts label map into plain string
    pub fn Format(&self) -> String {
        let l = self.String();
        if l.len() == 0 {
            return "<none>".to_owned();
        }
        return l;
    }

    // Conflicts takes 2 maps and returns true if there a key match between
    // the maps but the value doesn't match, and returns false in other cases
    pub fn Conflict(&self, labels: &Self) -> bool {
        let (small, big) = if self.0.len() < labels.0.len() {
            (self, labels)
        } else {
            (labels, self)
        };

        for (k, v) in small.as_ref() {
            match big.0.get(k) {
                None => return false,
                Some(val) => {
                    if v != val {
                        return true
                    }
                }
            }
        }

        return false;
    }

    // Merge combines given maps, and does not check for any conflicts
    // between the maps. In case of conflicts, second map (labels2) wins
    pub fn Merge(&self, labels: &Self) -> Self {
        let mut merged = BTreeMap::new();

        for (k, v) in self.as_ref() {
            merged.insert(k.to_string(), v.to_string());
        }

        for (k, v) in labels.as_ref() {
            merged.insert(k.to_string(), v.to_string());
        }

        return merged.into();
    }

    // Equals returns true if the given maps are equal
    pub fn Equals(&self, labels: &Self) -> bool {
        if self.0.len() != labels.0.len() {
            return false;
        }

        for (k, v) in self.as_ref() {
            match labels.0.get(k) {
                None => return false,
                Some(val) => {
                    if v != val {
                        return false
                    }
                }
            }
        }

        return true;
    }

    pub fn Matches(&self, labels: &Labels) -> bool {
        for (k, v) in self.as_ref() {
            if !labels.Has(k) || Some(v.to_string()) != labels.Get(k) {
                return false;
            }
        }

        return true;
    }

    pub fn Empty(&self) -> bool {
        return self.0.len() == 0;
    }

    pub fn ToSelector(&self) -> Selector {
        let mut res = Selector::default();
        for (k, v) in self.as_ref() {
            res.Add(Requirement { key: k.to_string(), op: SelectionOp::Equals, strVals: vec![v.to_string()] })
        }

        return res;
    }

    pub fn RequiresExactMatch(&self, lable: &str) -> Option<String> {
        return self.Get(lable)
    }

    pub fn toFullSelector(&self) -> Selector {
        return SelectorFromSet(self);
    }
}


// Token represents constant definition for lexer token
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token {
   	// ErrorToken represents scan error
	ErrorToken = 0 as isize,
	// EndOfStringToken represents end of string
	EndOfStringToken,
	// ClosedParToken represents close parenthesis
	ClosedParToken,
	// CommaToken represents the comma
	CommaToken,
	// DoesNotExistToken represents logic not
	DoesNotExistToken,
	// DoubleEqualsToken represents double equals
	DoubleEqualsToken,
	// EqualsToken represents equal
	EqualsToken,
	// GreaterThanToken represents greater than
	GreaterThanToken,
	// IdentifierToken represents identifier, e.g. keys and values
	IdentifierToken,
	// InToken represents in
	InToken,
	// LessThanToken represents less than
	LessThanToken,
	// NotEqualsToken represents not equal
	NotEqualsToken,
	// NotInToken represents not in
	NotInToken,
	// OpenParToken represents open parenthesis
	OpenParToken,
}

impl Default for Token {
    fn default() -> Self {
        return Self::ErrorToken;
    }
}

lazy_static! {
    // STRING2TOKEN contains the mapping between lexer Token and token literal
    // (except IdentifierToken, EndOfStringToken and ErrorToken since it makes no sense)
    pub static ref STRING2TOKEN : HashMap<String, Token> = [
        (")".to_owned(),     Token::ClosedParToken),
        (",".to_owned(),     Token::CommaToken),
        ("!".to_owned(),     Token::DoesNotExistToken),
        ("==".to_owned(),    Token::DoubleEqualsToken),
        ("=".to_owned(),     Token::EqualsToken),
        (">".to_owned(),     Token::GreaterThanToken),
        ("in".to_owned(),    Token::InToken),
        ("<".to_owned(),     Token::LessThanToken),
        ("!=".to_owned(),    Token::NotEqualsToken),
        ("notin".to_owned(), Token::NotInToken),
        ("(".to_owned(),     Token::OpenParToken),
    ].into_iter().collect();
}

// ScannedItem contains the Token and the literal produced by the lexer.
#[derive(Debug, Default)]
pub struct ScannedItem {
    pub tok: Token,
    pub literal: String,
}

// isWhitespace returns true if the rune is a space, tab, or newline.
pub fn IsWhitespace(ch: char) -> bool {
    return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
}

// isSpecialSymbol detects if the character ch can be an operator
pub fn IsSpecialSymbol(ch: char) -> bool {
    match ch {
        '=' | '!' | '(' | ')' | ',' | '>' | '<' => {
            return true;
        }
        _ => return false,
    }
}

// Lexer represents the Lexer struct for label selector.
// It contains necessary informationt to tokenize the input string
#[derive(Debug)]
pub struct Lexer {
    // s stores the string to be tokenized
    pub s: Vec<char>,
    // pos is the position currently tokenized
    pub pos: usize,
}

impl Lexer {
    // read returns the character currently lexed
    // increment the position and check the buffer overflow
    pub fn Read(&mut self) -> char {
        let mut b = '\0';
        if self.pos < self.s.len() {
            b = self.s[self.pos];
            self.pos += 1;
        }

        return b;
    }

    // unread 'undoes' the last read character
    pub fn Unread(&mut self) {
        self.pos -= 1;
    }

    // scanIDOrKeyword scans string to recognize literal token (for example 'in') or an identifier.
    pub fn ScanIDOrKeyword(&mut self) -> (Token, String) {
        let mut buf : Vec<char> = Vec::new();
        loop {
            let ch = self.Read();
            if ch == '\0' {
                break;
            } else if IsSpecialSymbol(ch) || IsWhitespace(ch) {
                self.Unread();
                break;
            } else {
                buf.push(ch);
            }
        }

        let s : String = buf.iter().collect();
        match STRING2TOKEN.get(&s) {
            None => (),
            Some(v) => {
                return (v.clone(), s)
            }
        }

        return (Token::IdentifierToken, s)
    }

    // scanSpecialSymbol scans string starting with special symbol.
    // special symbol identify non literal operators. "!=", "==", "="
    pub fn ScanSpecialSymbol(&mut self) -> (Token, String) {
        let mut lastScannedItem = ScannedItem::default();
        let mut buf : Vec<char> = Vec::new();

        loop {
            let ch = self.Read();
            if ch == '\0' {
                break;
            } else if IsSpecialSymbol(ch) {
                buf.push(ch);
                let s : String = buf.iter().collect();
                match STRING2TOKEN.get(&s) {
                    Some(token) => {
                        lastScannedItem = ScannedItem {
                            tok: token.clone(),
                            literal: s,
                        }
                    }
                    None => {
                        if lastScannedItem.tok != Token::ErrorToken {
                            self.Unread();
                            break;
                        }
                    }
                }
            } else {
                self.Unread();
                break;
            }
        }

        let s : String = buf.iter().collect();
        if lastScannedItem.tok == Token::ErrorToken {
            return (Token::ErrorToken, format!("error expected: keyword found '{}'", s))
        }

        return (lastScannedItem.tok, lastScannedItem.literal)
    }

    // skipWhiteSpaces consumes all blank characters
    // returning the first non blank character
    pub fn SkipWhiteSpaces(&mut self, ch: char) -> char {
        let mut ch = ch;
        loop {
            if !IsWhitespace(ch) {
                return ch
            }
            ch = self.Read();
        }
    }

    // Lex returns a pair of Token and the literal
    // literal is meaningfull only for IdentifierToken token
    pub fn Lex(&mut self) -> (Token, String) {
        let ch = self.Read();
        let ch = self.SkipWhiteSpaces(ch);
        if ch == '\0' {
            return (Token::EndOfStringToken, "".to_owned());
        } else if IsSpecialSymbol(ch) {
            self.Unread();
            return self.ScanSpecialSymbol();
        } else {
            self.Unread();
            return self.ScanIDOrKeyword();
        }
    }
}

// Parser data structure contains the label selector parser data structure
#[derive(Debug)]
pub struct Parser {
    pub l: Lexer,
    pub scanItems: Vec<ScannedItem>,
    pub position: usize,
}

// ParserContext represents context during parsing:
// some literal for example 'in' and 'notin' can be
// recognized as operator for example 'x in (a)' but
// it can be recognized as value for example 'value in (in)'
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParserContext {
    KeyAndOperator,
	Values,
}

impl Parser {
    // lookahead func returns the current token and string. No increment of current position
    pub fn Lookahead(&self, context: ParserContext) -> (Token, String) {
        let (mut tok, lit) = (self.scanItems[self.position].tok, self.scanItems[self.position].literal.clone());
        if context == ParserContext::Values {
            if tok == Token::InToken || tok == Token::NotInToken {
                tok = Token::IdentifierToken
            }
        }

        return (tok, lit)
    }

    // consume returns current token and string. Increments the position
    pub fn Consume(&mut self, context: ParserContext) -> (Token, String) {
        self.position += 1;
        let (mut tok, lit) = (self.scanItems[self.position-1].tok, self.scanItems[self.position-1].literal.clone());
        if context == ParserContext::Values {
            if tok == Token::InToken || tok == Token::NotInToken {
                tok = Token::IdentifierToken
            }
        }
        
        return (tok, lit)
    }

    // scan runs through the input string and stores the ScannedItem in an array
    // Parser can now lookahead and consume the tokens
    pub fn Scan(&mut self) {
        loop {
            let (token, literal) = self.l.Lex();
            self.scanItems.push(ScannedItem { tok: token, literal: literal });
            if token == Token::EndOfStringToken {
                break;
            }
        }
    }

    pub fn Parse(&mut self) -> Result<Selector> {
        self.Scan();
        let mut requirements = Selector::default();
        loop {
            let (tok, lit) = self.Lookahead(ParserContext::Values);
            match tok {
                Token::IdentifierToken | Token::DoesNotExistToken => {
                    let r = self.ParseRequirement()?;
                    requirements.Add(r);
                    let (t, l) = self.Consume(ParserContext::Values);
                    match t {
                        Token::EndOfStringToken => {
                            return Ok(requirements)
                        }
                        Token::CommaToken => {
                            let (t2, l2) = self.Lookahead(ParserContext::Values);
                            if t2 != Token::IdentifierToken && t2 != Token::DoesNotExistToken {
                                return Err(Error::CommonError(format!("found '{}', expected: identifier after ','", l2)))
                            }
                        }
                        _ => {
                            return Err(Error::CommonError(format!("found '{}', expected: ',' or 'end of string'", l)))
                        }
                    }
                }
                Token::EndOfStringToken => {
                    return Ok(requirements)
                }
                _ => {
                    return Err(Error::CommonError(format!("found '{}', expected: !, identifier, or 'end of string'", lit)))
                }
            }
        }
    }

    pub fn ParseRequirement(&mut self) -> Result<Requirement> {
        let (key, operator) = self.ParseKeyAndInferOperator()?;
        if operator == SelectionOp::Exists || operator == SelectionOp::DoesNotExist {
            return Ok(Requirement::New(&key, operator, Vec::new())?);
        }

        let operator = self.ParseOperator()?;
        let mut values = HashSet::new();
        match operator {
            SelectionOp::In | SelectionOp::NotIn => {
                values = self.ParseValues()?;
            }
            SelectionOp::Equals | 
            SelectionOp::DoubleEquals | 
            SelectionOp::NotEquals | 
            SelectionOp::GreaterThan | 
            SelectionOp::LessThan => {
                values = self.ParseExactValue()?;
            }
            _ => ()
        }

        return Ok(Requirement::New(&key, operator, values.into_iter().collect())?);
    }

    // parseKeyAndInferOperator parses literals.
    // in case of no operator '!, in, notin, ==, =, !=' are found
    // the 'exists' operator is inferred
    pub fn ParseKeyAndInferOperator(&mut self) -> Result<(String, SelectionOp)> {
        let mut operator = SelectionOp::None;
        let (mut tok, mut literal) = self.Consume(ParserContext::Values);
        if tok == Token::DoesNotExistToken {
            operator = SelectionOp::DoesNotExist;
            let (tt, tl) = self.Consume(ParserContext::Values);
            tok = tt;
            literal = tl;
        }

        if tok != Token::IdentifierToken {
            return Err(Error::CommonError(format!("found '{}', expected: identifier", literal)));
        }

        ValidateLabelKey(&literal)?;

        let (t, _) = self.Lookahead(ParserContext::Values);
        if t == Token::EndOfStringToken || t == Token::CommaToken {
            if operator != SelectionOp::DoesNotExist {
                operator = SelectionOp::Exists;
            }
        }

        return Ok((literal, operator))
    }

    // parseOperator returns operator and eventually matchType
    // matchType can be exact
    pub fn ParseOperator(&mut self) -> Result<SelectionOp> {
        let op;
        let (tok, lit) = self.Consume(ParserContext::KeyAndOperator);
        match tok {
            Token::InToken => op = SelectionOp::In,
            Token::EqualsToken => op = SelectionOp::Equals,
            Token::DoubleEqualsToken => op = SelectionOp::DoubleEquals,
            Token::GreaterThanToken => op = SelectionOp::GreaterThan,
            Token::LessThanToken => op = SelectionOp::LessThan,
            Token::NotInToken => op = SelectionOp::NotIn,
            Token::NotEqualsToken => op = SelectionOp::NotEquals,
            _ => {
                return Err(Error::CommonError(format!("found '{}', expected: In, NotIn ...", lit)));
            }
        }

        return Ok(op)
    }

    // parseValues parses the values for set based matching (x,y,z)
    pub fn ParseValues(&mut self) -> Result<HashSet<String>> {
        let (tok, lit) = self.Consume(ParserContext::Values);
        if tok != Token::OpenParToken {
            return Err(Error::CommonError(format!("found '{}' expected: '('", lit)));
        }

        let (tok, lit) = self.Lookahead(ParserContext::Values);
        match tok {
            Token::IdentifierToken | Token::CommaToken => {
                let s = self.ParseIdentifiersList()?;
                let (tok, _) = self.Consume(ParserContext::Values);
                if tok != Token::ClosedParToken {
                    return Err(Error::CommonError(format!("found '{}', expected: ')'", lit)))
                }
                return Ok(s)
            }
            Token::ClosedParToken => {
                self.Consume(ParserContext::Values);
                return Ok(HashSet::new());
            }
            _ => {
                return Err(Error::CommonError(format!("found '{}', expected: ',', ')' or identifier", lit)));
            }
        }
    }

    pub fn ParseIdentifiersList(&mut self) -> Result<HashSet<String>> {
        let mut s = HashSet::new();
        loop {
            let (tok, lit) = self.Consume(ParserContext::Values);
            match tok {
                Token::IdentifierToken => {
                    s.insert(lit);
                    let (tok2, lit2) = self.Lookahead(ParserContext::Values);
                    match tok2 {
                        Token::CommaToken => continue,
                        Token::ClosedParToken => return Ok(s),
                        _ => return Err(Error::CommonError(format!("found '{}', expected: ',' or ')'", lit2))),
                    }
                }
                Token::CommaToken => {  // handled here since we can have "(,"
                    if s.len() == 0 {
                        s.insert("".to_owned()); // to handle (,
                    }

                    let (tok2, _lit2) = self.Lookahead(ParserContext::Values);
                    if tok == Token::ClosedParToken {
                        s.insert("".to_owned()); // to handle ,)  Double "" removed by StringSet
                        return Ok(s)
                    }
                    if tok2 == Token::CommaToken {
                        self.Consume(ParserContext::Values);
                        s.insert("".to_owned()); // to handle ,, Double "" removed by StringSet
                    }
                }
                _ => {
                    return Err(Error::CommonError(format!("found '{}', expected: ',', or identifier", lit)))
                }
            }
        }
    }

    // parseExactValue parses the only value for exact match style
    pub fn ParseExactValue(&mut self) -> Result<HashSet<String>> {
        let mut s = HashSet::new();
        let (tok, _) = self.Lookahead(ParserContext::Values);
        if tok == Token::EndOfStringToken || tok == Token::CommaToken {
            s.insert("".to_owned());
            return Ok(s);
        }

        let (tok, lit) = self.Consume(ParserContext::Values);
        if tok == Token::IdentifierToken {
            s.insert(lit);
            return Ok(s);
        }

        return Err(Error::CommonError(format!("found '{}', expected: identifier", lit)))
    }
}

// Parse takes a string representing a selector and returns a selector
// object, or an error. This parsing function differs from ParseSelector
// as they parse different selectors with different syntaxes.
// The input will cause an error if it does not follow this form:
//
//	<selector-syntax>         ::= <requirement> | <requirement> "," <selector-syntax>
//	<requirement>             ::= [!] KEY [ <set-based-restriction> | <exact-match-restriction> ]
//	<set-based-restriction>   ::= "" | <inclusion-exclusion> <value-set>
//	<inclusion-exclusion>     ::= <inclusion> | <exclusion>
//	<exclusion>               ::= "notin"
//	<inclusion>               ::= "in"
//	<value-set>               ::= "(" <values> ")"
//	<values>                  ::= VALUE | VALUE "," <values>
//	<exact-match-restriction> ::= ["="|"=="|"!="] VALUE
//
// KEY is a sequence of one or more characters following [ DNS_SUBDOMAIN "/" ] DNS_LABEL. Max length is 63 characters.
// VALUE is a sequence of zero or more characters "([A-Za-z0-9_-\.])". Max length is 63 characters.
// Delimiter is white space: (' ', '\t')
// Example of valid syntax:
//
//	"x in (foo,,baz),y,z notin ()"
//
// Note:
//  1. Inclusion - " in " - denotes that the KEY exists and is equal to any of the
//     VALUEs in its requirement
//  2. Exclusion - " notin " - denotes that the KEY is not equal to any
//     of the VALUEs in its requirement or does not exist
//  3. The empty string is a valid VALUE
//  4. A requirement with just a KEY - as in "y" above - denotes that
//     the KEY exists and can be any VALUE.
//  5. A requirement with just !KEY requires that the KEY not exist.
pub fn Parse(selector: &str) -> Result<Selector> {
    let mut p =  Parser {
        l: Lexer { s: selector.chars().collect(), pos: 0 },
        scanItems: Vec::new(),
        position: 0,
    };

    let mut items = p.Parse()?;
    items.Sort();
    return Ok(items)
}

pub fn ValidateLabelKey(k: &str) -> Result<()> {
    return IsQualifiedName(k);
}

pub fn ValidateLabelValue(_k: &str, v: &str) -> Result<()> {
    return IsValidLabelValue(v);
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil and empty Sets are considered equivalent to Everything().
// It does not perform any validation, which means the server will reject
// the request if the Set contains invalid values.
pub fn SelectorFromSet(ls : &Labels) -> Selector {
    return SelectorFromValidatedSet(ls);
}

// ValidatedSelectorFromSet returns a Selector which will match exactly the given Set. A
// nil and empty Sets are considered equivalent to Everything().
// The Set is validated client-side, which allows to catch errors early.
pub fn ValidatedSelectorFromSet(ls: &Labels) -> Result<Selector> {
    let mut rs = Selector::default();
    if ls.0.len() ==  0 {
        return Ok(rs);
    }

    for (label, value) in ls.as_ref() {
        let r = Requirement::New(label, SelectionOp::Equals, vec![value.clone()])?;
        rs.0.push(r);
    }

    rs.Sort();
    return Ok(rs);
}

// SelectorFromValidatedSet returns a Selector which will match exactly the given Set.
// A nil and empty Sets are considered equivalent to Everything().
// It assumes that Set is already validated and doesn't do any validation.
// Note: this method copies the Set; if the Set is immutable, consider wrapping it with ValidatedSetSelector
// instead, which does not copy.
pub fn SelectorFromValidatedSet(ls: &Labels) -> Selector {
    let mut rs = Selector::default();
    if ls.0.len() ==  0 {
        return rs;
    }

    for (label, value) in ls.as_ref() {
        rs.0.push(Requirement { key: label.clone(), op: SelectionOp::Equals, strVals: vec![value.clone()] });
    }

    rs.Sort();
    return rs;
}

#[cfg(test1)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn matches(ls: &Labels, want: &str) {
        assert_eq!(ls.String(), want);
    }

    #[test]
    fn TestSetString() {
        matches(&Labels([("x".to_string(), "y".to_string())].into_iter().collect()), "x=y");
        matches(&Labels([("foo".to_string(), "bar".to_string())].into_iter().collect()), "foo=bar");
        matches(&Labels([("foo".to_string(), "bar".to_string()), ("x".to_string(), "y".to_string())].into_iter().collect()), "foo=bar,x=y");
    }

    #[test]
    fn TestLabelHas() {
        struct Test {
            Ls: Labels,
            Key: String,
            Has: bool,
        }

        let labelHasTests: [Test; 3] = [
            Test {
                Ls: Labels([("x".to_string(), "y".to_string())].into_iter().collect()),
                Key: "x".to_owned(),
                Has: true,
            },
            Test {
                Ls: Labels([("x".to_string(), "".to_string())].into_iter().collect()),
                Key: "x".to_owned(),
                Has: true,
            },
            Test {
                Ls: Labels([("x".to_string(), "y".to_string())].into_iter().collect()),
                Key: "foo".to_owned(),
                Has: false,
            },
        ];

        for lh in labelHasTests {
            let has = lh.Ls.Has(&lh.Key);
            assert_eq!(has, lh.Has, "{:?}.Has({}) => {}, expected {}", lh.Ls, lh.Key, has, lh.Has);
        }
    }

    #[test]
    fn TestLabelConflict() {
        struct Test {
            labels1: Labels,
            labels2: Labels,
            conflict: bool,
        }

        let tests = [
            Test {
                labels1: Labels([].into_iter().collect()),
                labels2: Labels([].into_iter().collect()),
                conflict: false
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string())].into_iter().collect()),
                labels2: Labels([("infra".to_string(), "true".to_string())].into_iter().collect()),
                conflict: false
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string())].into_iter().collect()),
                labels2: Labels([("infra".to_string(), "true".to_string()), ("env".to_string(), "test".to_string())].into_iter().collect()),
                conflict: false
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string())].into_iter().collect()),
                labels2: Labels([("env".to_string(), "test1".to_string())].into_iter().collect()),
                conflict: true
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string()), ("infra".to_string(), "false".to_string())].into_iter().collect()),
                labels2: Labels([("infra".to_string(), "true".to_string()), ("env".to_string(), "test".to_string())].into_iter().collect()),
                conflict: true
            },
        ];

        for t in &tests {
            let conflict = t.labels1.Conflict(&t.labels2);
            assert_eq!(conflict, t.conflict);
        }
    }

    #[test]
    fn TestLabelMerge() {
        struct Test {
            labels1: Labels,
            labels2: Labels,
            mergedLabels: Labels,
        }

        let tests = [
            Test {
                labels1: Labels([].into_iter().collect()),
                labels2: Labels([].into_iter().collect()),
                mergedLabels: Labels([].into_iter().collect()),
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string())].into_iter().collect()),
                labels2: Labels([].into_iter().collect()),
                mergedLabels: Labels([("env".to_string(), "test".to_string())].into_iter().collect()),
            },
            Test {
                labels1: Labels([("env".to_string(), "test".to_string()), ("infra".to_string(), "false".to_string())].into_iter().collect()),
                labels2: Labels([("infra".to_string(), "true".to_string()), ("a".to_string(), "b".to_string())].into_iter().collect()),
                mergedLabels: Labels([("infra".to_string(), "true".to_string()), ("env".to_string(), "test".to_string()), ("a".to_string(), "b".to_string())].into_iter().collect()),
            },
        ];

        for t in &tests {
            let mergedLabels = t.labels1.Merge(&t.labels2);
            assert_eq!(mergedLabels.Equals(&t.mergedLabels), true);
        }
    }

    #[test]
    fn TestLabelSelectorParse() {
        struct Test {
            selector: String,
            labels: Labels,
            valid: bool,
        }

        let tests = [
            Test {
                selector: "".to_owned(),
                labels: Labels([].into_iter().collect()),
                valid: true,
            },
            Test {
                selector: ",".to_owned(),
                labels: Labels([].into_iter().collect()),
                valid: false,
            },
            Test {
                selector: "x=y".to_owned(),
                labels: Labels([("x".to_string(), "y".to_string())].into_iter().collect()),
                valid: true,
            },
            Test {
                selector: "test= dddy, a =b, c=d".to_owned(),
                labels: Labels([("test".to_string(), "dddy".to_string()),("a".to_string(), "b".to_string()),("c".to_string(), "d".to_string())].into_iter().collect()),
                valid: true,
            },
        ];

        for t in &tests {
            let labels = match Labels::New(&t.selector) {
                Err(_) => {
                    assert_eq!(t.valid, false);
                    continue;
                }
                Ok(l) => l,
            };
            assert_eq!(labels.Equals(&t.labels), t.valid);
        }
    }

    #[test]
    fn TestSelectorParse() {
        let testGoodStrings = [
            "x=a,y=b,z=c",
            "",
            "x!=a,y=b",
            "x=",
            "x= ",
            "x=,z= ",
            "x= ,z= ",
            "!x",
            "x>1",
            "x>1,z<5",
        ];

        for t in &testGoodStrings {
            let lq = Parse(t).unwrap();
            assert_eq!(t.to_string().replace(" ", ""), lq.String());
        }

        let testBadStrings = [
            "x=a||y=b",
            "x==a==b",
            "!x=a",
            "x<a",
        ];

        for t in &testBadStrings {
            match Parse(t) {
                Err(_) => continue,
                Ok(_) => {
                    assert!(false);
                    continue;
                }
            }
        }
    }

    #[test]
    fn TestDeterministicParse() {
        let s1 = Parse("x=a,a=x").unwrap();
        let s2 = Parse("a=x,x=a").unwrap();
        assert_eq!(s1.String(), s2.String());
    }

    fn ExepctMatch(selector: &str, ls: &Labels) {
        let lq = match Parse(selector) {
            Ok(lq) => lq,
            Err(e) => {
                assert!(false, "error {:?}", e); 
                return;
            }
        };

        assert!(lq.Match(ls),  "Wanted '{:?}' to match '{:?}', but it did not.", &lq, ls)
    }

    fn ExepctNoMatch(selector: &str, ls: &Labels) {
        let lq = match Parse(selector) {
            Ok(lq) => lq,
            Err(e) => {
                assert!(false, "error {:?}", e); 
                return;
            }
        };

        assert!(!lq.Match(ls),  "Wanted '{}' to not match '{:?}', but it did.", selector, ls)
    }

    #[test]
    fn TestSelectorMatches() {
        ExepctMatch("", &Labels([("x".to_string(), "y".to_string())].into_iter().collect()));
        ExepctMatch("x=y", &Labels([("x".to_string(), "y".to_string())].into_iter().collect()));
        ExepctMatch("x>1", &Labels([("x".to_string(), "2".to_string())].into_iter().collect()));
        ExepctMatch("x<1", &Labels([("x".to_string(), "0".to_string())].into_iter().collect()));
        ExepctMatch("x", &Labels([("x".to_string(), "y".to_string())].into_iter().collect()));
        ExepctMatch("!x", &Labels([("z".to_string(), "y".to_string())].into_iter().collect()));
        ExepctMatch("in=notin", &Labels([("in".to_string(), "notin".to_string())].into_iter().collect()));
        ExepctMatch("x=y,z=w", &Labels([("x".to_string(), "y".to_string()), ("z".to_string(), "w".to_string())].into_iter().collect()));
        ExepctMatch("x!=y,z!=w", &Labels([("x".to_string(), "z".to_string()), ("z".to_string(), "a".to_string())].into_iter().collect()));

        ExepctNoMatch("x=y", &Labels([].into_iter().collect()));
        ExepctNoMatch("x=y", &Labels([("x".to_string(), "z".to_string())].into_iter().collect()));
        ExepctNoMatch("x=y,z=w", &Labels([("x".to_string(), "w".to_string()), ("z".to_string(), "w".to_string())].into_iter().collect()));
        ExepctNoMatch("x!=y,z!=w", &Labels([("x".to_string(), "z".to_string()), ("z".to_string(), "w".to_string())].into_iter().collect()));
        ExepctNoMatch("x", &Labels([("z".to_string(), "y".to_string())].into_iter().collect()));
        ExepctNoMatch("!x", &Labels([("x".to_string(), "y".to_string())].into_iter().collect()));
        ExepctNoMatch("x>1", &Labels([("x".to_string(), "0".to_string())].into_iter().collect()));
        ExepctNoMatch("x<1", &Labels([("x".to_string(), "1".to_string())].into_iter().collect()));
        
        let labelSet = Labels([("foo".to_string(), "bar".to_string()), ("baz".to_string(), "blah".to_string())].into_iter().collect());

        ExepctMatch("foo=bar", &labelSet);
        ExepctMatch("baz=blah", &labelSet);
        ExepctMatch("foo=bar,baz=blah", &labelSet);

        ExepctNoMatch("foo=blah", &labelSet);
        ExepctNoMatch("baz=bar", &labelSet);
        ExepctNoMatch("foo=bar,foobar=bar,baz=blah", &labelSet);
    }

    fn ExpectMatchDirect(selector: &Labels, ls: &Labels) {
        assert!(SelectorFromSet(&selector).Match(&ls), 
            "Wanted {:?} to match '{:?}', but it did not.\n", selector, ls);
    }

    fn ExpectNoMatchDirect(selector: &Labels, ls: &Labels) {
        assert!(!SelectorFromSet(&selector).Match(&ls), 
            "Wanted {:?} to not match '{:?}', but it did.\n", selector, ls);
    }

    #[test]
    fn TestSetMatches() {
        let labelSet = Labels([("foo".to_string(), "bar".to_string()), ("baz".to_string(), "blah".to_string())].into_iter().collect());

        ExpectMatchDirect(&Labels([].into_iter().collect()), &labelSet);
        ExpectMatchDirect(&Labels([("foo".to_string(), "bar".to_string())].into_iter().collect()), &labelSet);
        ExpectMatchDirect(&Labels([("baz".to_string(), "blah".to_string())].into_iter().collect()), &labelSet);
        ExpectMatchDirect(&Labels([("foo".to_string(), "bar".to_string()), ("baz".to_string(), "blah".to_string())].into_iter().collect()), &labelSet);
    }

    #[test]
    fn TestSetIsEmpty() {
        let labelSet = Labels::default();
        assert!(labelSet.Empty());

    }

    #[test]
    fn TestLexer() {
        struct Test {
            s: String,
            t: Token,
        }

        let testcases = [
            Test {s: "".to_owned(), t: Token::EndOfStringToken},
            Test {s: ",".to_owned(), t: Token::CommaToken},
            Test {s: "notin".to_owned(), t: Token::NotInToken},
            Test {s: "in".to_owned(), t: Token::InToken},
            Test {s: "=".to_owned(), t: Token::EqualsToken},
            Test {s: "==".to_owned(), t: Token::DoubleEqualsToken},
            Test {s: ">".to_owned(), t: Token::GreaterThanToken},
            Test {s: "<".to_owned(), t: Token::LessThanToken},
            Test {s: "!".to_owned(), t: Token::DoesNotExistToken},
            Test {s: "!=".to_owned(), t: Token::NotEqualsToken},
            Test {s: "(".to_owned(), t: Token::OpenParToken},
            Test {s: ")".to_owned(), t: Token::ClosedParToken},
            Test {s: "~".to_owned(), t: Token::IdentifierToken},
            Test {s: "||".to_owned(), t: Token::IdentifierToken},
        ];

        for t in &testcases {
            let mut l = Lexer {
                s: t.s.chars().collect(),
                pos: 0
            };

            let (token, lit) = l.Lex();
            assert_eq!(token, t.t, "Got {:?} it should be {:?} for '{}'", token, t.t, t.s);

            if t.t != Token::ErrorToken {
                assert_eq!(lit, t.s)
            }
        }
    }

    fn GetRequirement(key: &str, op: SelectionOp, vals: Vec<String>) -> Requirement {
        let req = match Requirement::New(key, op, vals) {
            Err(_e) => {
                //assert!(false, "error is {:?}", e);
                Requirement::default()
            }
            Ok(r) => r,
        };

        return req;
    }

    #[test]
    fn TestSetSelectorParser() {
        struct Test {
            In: String,
            Out: Selector,
            Match: bool,
            Valid: bool,
        }

        let tests = [
            Test {
                In: "".to_owned(),
                Out: Selector::default(),
                Match: true,
                Valid: true
            },
            Test {
                In: "\rx".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::Exists, vec![])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "this-is-a-dns.domain.com/key-with-dash".to_owned(),
                Out: Selector(vec![GetRequirement("this-is-a-dns.domain.com/key-with-dash", SelectionOp::Exists, vec![])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "this-is-another-dns.domain.com/key-with-dash in (so,what)".to_owned(),
                Out: Selector(vec![GetRequirement("this-is-another-dns.domain.com/key-with-dash", SelectionOp::In, vec!["so".to_owned(), "what".to_owned()])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "0.1.2.domain/99 notin (10.10.100.1, tick.tack.clock)".to_owned(),
                Out: Selector(vec![GetRequirement("0.1.2.domain/99", SelectionOp::NotIn, vec!["10.10.100.1".to_owned(), "tick.tack.clock".to_owned()])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "foo  in	 (abc)".to_owned(),
                Out: Selector(vec![GetRequirement("foo", SelectionOp::In, vec!["abc".to_owned()])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x notin\n (abc)".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::NotIn, vec!["abc".to_owned()])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x  notin	\t	(abc,def)".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::NotIn, vec!["abc".to_owned(), "def".to_owned()])]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x in (abc,def)".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::In, vec!["abc".to_owned(), "def".to_owned()])]),
                Match: true,
                Valid: true
            },
            // todo: debug
            Test {
                In: "x in (abc,)".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::In, vec!["abc".to_owned()])]),
                Match: true,
                Valid: false
            },
            Test {
                In: "x in ()".to_owned(),
                Out: Selector(vec![GetRequirement("x", SelectionOp::In, vec![])]),
                Match: true,
                Valid: false
            },
            Test {
                In: "x=a,y!=b,z in (h,i,j)".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::Equals, vec!["a".to_owned()]),
                    GetRequirement("y", SelectionOp::NotEquals, vec!["b".to_owned()]),
                    GetRequirement("z", SelectionOp::In, vec!["h".to_owned(), "i".to_owned(), "j".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x,y in (a)".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::Exists, vec![]),
                    GetRequirement("y", SelectionOp::In, vec!["a".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x=a".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::Equals, vec!["a".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x>1".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::GreaterThan, vec!["1".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x<1".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::LessThan, vec!["1".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x=a,y!=b".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::Equals, vec!["a".to_owned()]),
                    GetRequirement("y", SelectionOp::NotEquals, vec!["b".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x=a,y!=b,z in (h,i,j)".to_owned(),
                Out: Selector(vec![
                    GetRequirement("x", SelectionOp::Equals, vec!["a".to_owned()]),
                    GetRequirement("y", SelectionOp::NotEquals, vec!["b".to_owned()]),
                    GetRequirement("z", SelectionOp::In, vec!["h".to_owned(), "i".to_owned(), "j".to_owned()]),
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "x=a||y=b".to_owned(),
                Out: Selector(vec![                   
                ]),
                Match: false,
                Valid: false
            },
            Test {
                In: "x,,y".to_owned(),
                Out: Selector(vec![                   
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: ",x,y".to_owned(),
                Out: Selector(vec![                   
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "x nott in (y)".to_owned(),
                Out: Selector(vec![                   
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "x notin ( )".to_owned(),
                Out: Selector(vec![   
                    GetRequirement("x", SelectionOp::NotIn, vec![]),                
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "x notin (, a)".to_owned(),
                Out: Selector(vec![    
                    GetRequirement("x", SelectionOp::NotIn, vec!["".to_owned(), "a".to_owned()]),        
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "a in (xyz),".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "a in (xyz)b notin ()".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "a ".to_owned(),
                Out: Selector(vec![  
                    GetRequirement("a", SelectionOp::Exists, vec![]),         
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "a in (x,y,notin, z,in)".to_owned(),
                Out: Selector(vec![  
                    GetRequirement("a", SelectionOp::In, vec!["in".to_owned(), "notin".to_owned(),"x".to_owned(), "y".to_owned(),"z".to_owned()]),         
                ]),
                Match: true,
                Valid: true
            },
            Test {
                In: "a in (xyz abc)".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "a notin(".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "a (".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
            Test {
                In: "(".to_owned(),
                Out: Selector(vec![           
                ]),
                Match: true,
                Valid: false
            },
        ];

        for t in &tests {
            match Parse(&t.In) {
                Err(e) =>  {
                    assert!(!t.Valid, "Parse({}) => {:?} expected no error", t.In, e);
                    continue;
                }
                Ok(sel) => {
                    assert!(t.Valid, "Parse({}) => {:#?} expected error", t.In, sel);
                    if t.Match {
                        assert!(sel.Equ(&t.Out), "Parse({:#?}) => parse output '{:#?}' doesn't match '{:#?}' expected match", t.In, sel, t.Out)
                    }
                }
            }
        }
    }
}
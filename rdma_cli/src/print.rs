// Copyright (c) 2021 Quark Container Authors
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

#[macro_export]
macro_rules! raw {
 // macth like arm for macro
    ($a:expr,$b:expr,$c:expr)=>{
        {
           error!("raw:: {:x}/{:x}/{:x}", $a, $b, $c);
        }
    }
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        println!("{}", &format!("{}\n",&s));
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        println!("Print {}", &s);
    });
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        println!("ERROR {}", &s);
    });
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        println!("INFO {}", &s);
    });
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => ({
        let s = &format!($($arg)*);
        println!("DEBUG {}", &s);
    });
}
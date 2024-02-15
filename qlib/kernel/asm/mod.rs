// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

cfg_x86_64! {
    mod x86_64;
    pub use self::x86_64::*;
}

cfg_aarch64! {
    pub mod aarch64;
    pub use self::aarch64::*;
}

// muldiv64 multiplies two 64-bit numbers, then divides the result by another
// 64-bit number.
//
// It requires that the result fit in 64 bits, but doesn't require that
// intermediate values do; in particular, the result of the multiplication may
// require 128 bits.
//
// It returns !ok if divisor is zero or the result does not fit in 64 bits.
#[inline(always)]
pub fn muldiv64(value: u64, multiplier: u64, divisor: u64) -> (u64, bool) {
    let val = value as u128 * multiplier as u128;
    let res = val / divisor as u128;
    if res > core::u64::MAX as u128 {
        return (0, false);
    }

    return (res as u64, true);
}

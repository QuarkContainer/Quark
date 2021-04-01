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

use super::super::qlib::platform::defs_impl::*;

#[derive(Default)]
pub struct DefaultPlatform {}

impl DefaultPlatform {
    // SupportsAddressSpaceIO implements platform.Platform.SupportsAddressSpaceIO.
    pub fn SupportsAddressSpaceIO(&self) -> bool {
        return false;
    }

    // MapUnit implements platform.Platform.MapUnit.
    pub fn MapUint(&self) -> u64 {
        // We greedily creates PTEs in MapFile, so extremely large mappings can
        // be expensive. Not _that_ expensive since we allow super pages, but
        // even though can get out of hand if you're creating multi-terabyte
        // mappings. For this reason, we limit mappings to an arbitrary 16MB.
        return 16 << 20
    }

    // MinUserAddress returns the lowest available address.
    pub fn MinUserAddress(&self) -> u64 {
        return MIN_USER_ADDR;
    }

    pub fn MaxUserAddress(&self) -> u64 {
        return MAX_USER_ADDR;
    }
}
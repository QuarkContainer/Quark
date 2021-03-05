// Copyright (c) 2021 QuarkSoft LLC
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

use alloc::sync::Arc;

use super::super::super::qlib::mem::seq::*;
use super::transport::unix::*;

// EndpointWriter implements safemem.Writer that writes to a transport.Endpoint.
pub struct EndpointWriter {
    pub Endpoint: Arc<Endpoint>,
    pub Control: SCMControlMessages,
    pub To: BoundEndpoint,
}
/*
impl BlockSeqWriter for EndpointWriter {
    fn WriteFromBlocks(&mut self, srcs: &[IoVec]) -> Result<usize> {

    }
}
*/
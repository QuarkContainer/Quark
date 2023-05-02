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

use k8s_openapi::api::core::v1 as k8s;
use qobjs::{pb_gen::node_mgr_pb::{self as NmMsg, node_agent_message::MessageBody}, runtime_types::{PodToString, QuarkPod, PodState, NodeToString}};
use qobjs::common::*;

use crate::node::QuarkNode;

pub fn BuildNodeAgentNodeState(node: &QuarkNode, revision: i64) -> Result<NmMsg::NodeAgentMessage> {
    let mut podStates = Vec::new();
    for v in node.pods.lock().unwrap().values() {
        let s = BuildNodeMgrGrpcPodState(revision, v)?;
        let body = s.message_body.as_ref().unwrap();
        let state = match body {
            MessageBody::PodState(state) => state.clone(),
            _ => panic!("impossible"),
        };
        podStates.push(state);
    }

    let ns = NmMsg::NodeState {
        node_revision: revision,
        node: NodeToString(&*node.node.lock().unwrap())?,
        pod_states: podStates,
    };

    return Ok(NmMsg::NodeAgentMessage {
        message_body: Some(NmMsg::node_agent_message::MessageBody::NodeState(ns)),
        ..Default::default()
    })
}

pub fn BuildNodeAgentNodeReady(node: &QuarkNode, revision: i64) -> Result<NmMsg::NodeAgentMessage> {
    let mut podStates = Vec::new();
    for v in node.pods.lock().unwrap().values() {
        let s = BuildNodeMgrGrpcPodState(revision, v)?;
        let body = s.message_body.as_ref().unwrap();
        let state = match body {
            MessageBody::PodState(state) => state.clone(),
            _ => panic!("impossible"),
        };
        podStates.push(state);
    }

    let ns = NmMsg::NodeReady {
        node_revision: revision,
        node: NodeToString(&*node.node.lock().unwrap())?,
        pod_states: podStates,
    };

    return Ok(NmMsg::NodeAgentMessage {
        message_body: Some(NmMsg::node_agent_message::MessageBody::NodeReady(ns)),
        ..Default::default()
    })
}

pub fn BuildNodeMgrPodStateForFailedPod(nodeRev: i64, pod: &k8s::Pod) -> Result<NmMsg::NodeAgentMessage> {
    let state = NmMsg::pod_state::State::Terminated;

    let s: NmMsg::PodState = NmMsg::PodState {
        node_revision: nodeRev,
        state: state as i32,
        pod: PodToString(pod)?,
        resource: Some(NmMsg::PodResource::default()),
    };

    return Ok(NmMsg::NodeAgentMessage {
        message_body: Some(NmMsg::node_agent_message::MessageBody::PodState(s)),
        ..Default::default()
    })
}

pub fn BuildNodeMgrGrpcPodState(nodeRev: i64, pod: &QuarkPod) -> Result<NmMsg::NodeAgentMessage> {
    let s: NmMsg::PodState = NmMsg::PodState {
        node_revision: nodeRev,
        state: PodStateToNodeAgentState(pod) as i32,
        pod: PodToString(&(*pod.Pod().read().unwrap()))?,
        resource: Some(NmMsg::PodResource::default()),
    };

    return Ok(NmMsg::NodeAgentMessage {
        message_body: Some(NmMsg::node_agent_message::MessageBody::PodState(s)),
        ..Default::default()
    })
}

pub fn PodStateToNodeAgentState(pod: &QuarkPod) -> NmMsg::pod_state::State {
    match pod.PodState() {
        PodState::Creating => {
            return NmMsg::pod_state::State::Creating;
        }
        PodState::Created => {
            return NmMsg::pod_state::State::Standby;
        }
        PodState::Running => {
            return NmMsg::pod_state::State::Creating;
        }
        PodState::Terminating => {
            return NmMsg::pod_state::State::Terminating;
        }
        PodState::Terminated => {
            return NmMsg::pod_state::State::Terminated;
        }
        PodState::Cleanup => {
            return NmMsg::pod_state::State::Terminated;
        }
        PodState::Failed => {
            let mut allContainerTerminated = true;
            let containers = pod.Containers();
            for v in &containers {
                allContainerTerminated = allContainerTerminated && v.ContainerExit();
            }
            
            if allContainerTerminated {
                return NmMsg::pod_state::State::Terminated;
            } else {
                return NmMsg::pod_state::State::Terminating;
            }       
        }
        _ => {
            return NmMsg::pod_state::State::Creating;
        }
    }

}
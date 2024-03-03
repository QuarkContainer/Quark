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
// limitations under

use std::sync::Arc;

use pyo3::prelude::*;


#[pyclass]
pub struct TensorInner {
    pub type_: u8, // 1: CPU 2: GPU
    pub id: u8, // GPU ID
    pub offset: u64,
    pub len: u64,
}

#[pyclass]
#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

#[pyclass]
pub struct QObject {
    #[pyo3(get, set)]
    pub data: Vec<u8>,
    #[pyo3(get, set)]
    pub tensors: Vec<Tensor>
}

#[pyclass]
#[derive(Clone)]
pub struct QObjectList(Arc<Vec<QObject>>);

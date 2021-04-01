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

use alloc::vec::Vec;
use core::ops::Deref;

#[derive(Default)]
pub struct View(pub Vec<u8>);

impl Deref for View {
    type Target = Vec<u8>;

    fn deref(&self) -> &Vec<u8> {
        &self.0
    }
}

impl View {
    // NewView allocates a new buffer and returns an initialized view that covers
    // the whole buffer.
    pub fn New(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0);
        return Self(data)
    }

    // NewViewFromBytes allocates a new buffer and copies in the given bytes.
    pub fn NewFromBytes(b: Vec<u8>) -> Self {
        return Self(b)
    }

    // TrimFront removes the first "count" bytes from the visible section of the
    // buffer.
    pub fn TrimFront(&mut self, count: usize) {
        let vec2 = self.0.split_off(count);
        self.0 = vec2;
    }

    // CapLength irreversibly reduces the length of the visible section of the
    // buffer to the value specified.
    pub fn CapLength(&mut self, length: usize) {
        self.0.truncate(length);
    }

    // get the buffer
    pub fn Get(&mut self) -> Self {
        return Self(self.0.split_off(0));
    }
}

// VectorisedView is a vectorised version of View using non contigous memory.
// It supports all the convenience methods supported by View.
pub struct VectorisedView {
    pub views: Vec<View>,
    pub size: usize,
}

impl VectorisedView {
    // NewVectorisedView creates a new vectorised view from an already-allocated slice
    // of View and sets its size.
    pub fn New(size: usize, views: Vec<View>) -> Self {
        return Self {
            views: views,
            size: size,
        }
    }

    // TrimFront removes the first "count" bytes of the vectorised view.
    pub fn TrimFront(&mut self, count: usize) {
        let mut count = count;

        while count > 0 && self.views.len() > 0 {
            if count < self.views[0].len() {
                self.size -= count;
                self.views[0].TrimFront(count);
                return
            }

            count -= self.views[0].len();
            self.RemoveFirst();
        }
    }

    // CapLength irreversibly reduces the length of the vectorised view.
    pub fn CapLength(&mut self, length: usize) {
        if self.size < length {
            return
        }

        let mut length = length;

        self.size = length;
        for i in 0..self.views.len() {
            let len = self.views[i].len();
            if len >= length {
                if length == 0 {
                    self.views.truncate(i);
                } else {
                    self.views[i].CapLength(length);
                    self.views.truncate(i+1);
                }
            }

            length -= len;
        }
    }

    // Clone returns a clone of this VectorisedView.
    // If the buffer argument is large enough to contain all the Views of this VectorisedView,
    // the method will avoid allocations and use the buffer to store the Views of the clone.
    pub fn Clone(mut self, mut buf: Vec<View>) -> Self {
        buf.clear();
        buf.append(&mut self.views);
        return Self {
            views: buf,
            size: self.size,
        }
    }

    // First returns the first view of the vectorised view.
    pub fn First(&self) -> Option<&View> {
        if self.views.len() == 0 {
            return None;
        }

        return Some(&self.views[0])
    }

    // RemoveFirst removes the first view of the vectorised view.
    pub fn RemoveFirst(&mut self) {
        if self.views.len() == 0 {
            return
        }

        self.size -= self.views[0].len();
        self.views = self.views.split_off(1);
    }

    // Size returns the size in bytes of the entire content stored in the vectorised view.
    pub fn Size(&self) -> usize {
        return self.size;
    }

    // ToView returns a single view containing the content of the vectorised view.
    //
    // If the vectorised view contains a single view, that view will be returned
    // directly.
    pub fn ToView(mut self) -> View {
        if self.views.len() == 1{
            let data = self.views.pop();
            return data.unwrap();
        }

        let mut data = Vec::with_capacity(self.size);
        for i in 0..self.views.len() {
            data.append(&mut self.views[i].0)
        };

        return View(data)
    }

    // Views returns the slice containing the all views.
    pub fn Views(&self) -> &Vec<View> {
        return &self.views
    }

    // Append appends the views in a vectorised view to this vectorised view.
    pub fn Append(&mut self, mut vv2: VectorisedView) {
        self.size += vv2.size;
        self.views.append(&mut vv2.views);
    }
}
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

use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::ToString;

use super::super::task::*;
use super::super::super::common::*;
use super::super::super::range::*;
use super::super::super::addr::*;

// Mappable represents a memory-mappable object, a mutable mapping from uint64
// offsets to (platform.File, uint64 File offset) pairs.
pub trait Mappable {
    // AddMapping notifies the Mappable of a mapping from addresses ar in ms to
    // offsets [offset, offset+ar.Length()) in this Mappable.
    //
    // The writable flag indicates whether the backing data for a Mappable can
    // be modified through the mapping. Effectively, this means a shared mapping
    // where Translate may be called with at.Write == true. This is a property
    // established at mapping creation and must remain constant throughout the
    // lifetime of the mapping.
    //
    // Preconditions: offset+ar.Length() does not overflow.
    fn AddMapping(&self, task: &Task, ms: &Arc<MappingSpace>, ar: &Range, offset: u64, writeable: bool) -> Result<()>;

    // RemoveMapping notifies the Mappable of the removal of a mapping from
    // addresses ar in ms to offsets [offset, offset+ar.Length()) in this
    // Mappable.
    //
    // Preconditions: offset+ar.Length() does not overflow. The removed mapping
    // must exist. writable must match the corresponding call to AddMapping.
    fn RemoveMapping(&self, task: &Task, ms: &Arc<MappingSpace>, ar: &Range, offset: u64, writeable: bool);

    // CopyMapping notifies the Mappable of an attempt to copy a mapping in ms
    // from srcAR to dstAR. For most Mappables, this is equivalent to
    // AddMapping. Note that it is possible that srcAR.Length() != dstAR.Length(),
    // and also that srcAR.Length() == 0.
    //
    // CopyMapping is only called when a mapping is copied within a given
    // MappingSpace; it is analogous to Linux's vm_operations_struct::mremap.
    //
    // Preconditions: offset+srcAR.Length() and offset+dstAR.Length() do not
    // overflow. The mapping at srcAR must exist. writable must match the
    // corresponding call to AddMapping.
    fn CopyMapping(&self, ms: &Arc<MappingSpace>, srcAR: &Range, dstAR: &Range, offset: u64, writeable: bool) -> Result<()>;

    // Translate returns the Mappable's current mappings for at least the range
    // of offsets specified by required, and at most the range of offsets
    // specified by optional. at is the set of access types that may be
    // performed using the returned Translations. If not all required offsets
    // are translated, it returns a non-nil error explaining why.
    //
    // Translations are valid until invalidated by a callback to
    // MappingSpace.Invalidate or until the caller removes its mapping of the
    // translated range. Mappable implementations must ensure that at least one
    // reference is held on all pages in a platform.File that may be the result
    // of a valid Translation.
    //
    // Preconditions: required.Length() > 0. optional.IsSupersetOf(required).
    // required and optional must be page-aligned. The caller must have
    // established a mapping for all of the queried offsets via a previous call
    // to AddMapping. The caller is responsible for ensuring that calls to
    // Translate synchronize with invalidation.
    //
    // Postconditions: See CheckTranslateResult.
    fn Translate(&self, ms: &Arc<MappingSpace>, required: &Range, optional: &Range, at: &AccessType) -> (Vec<Translation>, Result<()>);

    // InvalidateUnsavable requests that the Mappable invalidate Translations
    // that cannot be preserved across save/restore.
    //
    // Invariant: InvalidateUnsavable never races with concurrent calls to any
    // other Mappable methods.
    fn InvalidateUnsavable(&self) -> Result<()>;
}

pub trait HostFile {
    // All pages in a File are reference-counted.

    // IncRef increments the reference count on all pages in fr.
    //
    // Preconditions: fr.Start and fr.End must be page-aligned. fr.Length() >
    // 0. At least one reference must be held on all pages in fr. (The File
    // interface does not provide a way to acquire an initial reference;
    // implementors may define mechanisms for doing so.)
    fn IncrRef(&self, fr: &Range);

    // DecRef decrements the reference count on all pages in fr.
    //
    // Preconditions: fr.Start and fr.End must be page-aligned. fr.Length() >
    // 0. At least one reference must be held on all pages in fr.
    fn DecrRef(&self, fr: &Range);

    // MapInternal returns a mapping of the given file offsets in the invoking
    // process' address space for reading and writing.
    //
    // Note that fr.Start and fr.End need not be page-aligned.
    //
    // Preconditions: fr.Length() > 0. At least one reference must be held on
    // all pages in fr.
    //
    // Postconditions: The returned mapping is valid as long as at least one
    // reference is held on the mapped pages.
    fn MapInternal(&self, fr: &Range, at: &AccessType) -> Result<Vec<Range>>;

    // FD returns the file descriptor represented by the File.
    //
    // The only permitted operation on the returned file descriptor is to map
    // pages from it consistent with the requirements of AddressSpace.MapFile.
    fn FD(&self) -> i32;
}

pub trait MappingSpace: Send + Sync {
    // Invalidate is called to notify the MappingSpace that values returned by
    // previous calls to Mappable.Translate for offsets mapped by addresses in
    // ar are no longer valid.
    //
    // Preconditions: ar.Length() != 0. ar must be page-aligned.

    // InvalidatePrivate is true if private pages in the invalidated region
    // should also be discarded, causing their data to be lost.
    fn Invalidate(&self, ar: &Range, invalidatePrivate: bool);

    //MappingSpace ID, used for cmp
    fn ID(&self) -> u64;
}

#[derive(Clone)]
pub struct Translation {
    // Source is the translated range in the Mappable.
    pub Source: Range,

    // File is the mapped file.
    pub File: Arc<HostFile>,

    // Offset is the offset into File at which this Translation begins.
    pub Offset: u64,

    // Perms is the set of permissions for which platform.AddressSpace.MapFile
    // and platform.AddressSpace.MapInternal on this Translation is permitted.
    pub Perms: AccessType,
}

impl Translation {
    // FileRange returns the FileRange represented by t.
    pub fn FileRange(&self) -> Range {
        return Range::New(self.Offset, self.Source.Len());
    }

    // CheckTranslateResult returns an error if (ts, terr) does not satisfy all
    // postconditions for Mappable.Translate(required, optional, at).
    pub fn CheckTranslateResult(&self, required: &Range, optional: &Range, at: &AccessType, ts: &Vec<Translation>, terr: Result<()>) -> Result<()> {
        if !Addr(required.Start()).IsPageAligned() || !Addr(required.End()).IsPageAligned() {
            panic!("unaligned required range: {:?}", required);
        }

        if !optional.IsSupersetOf(&required) {
            panic!("optional range {:?} is not a superset of required range {:?}", optional, required);
        }

        if !Addr(optional.Start()).IsPageAligned() || !Addr(optional.End()).IsPageAligned() {
            panic!("unaligned optional range: {:?}", optional);
        }

        // The first Translation must include required.Start.
        if ts.len() != 0 && !ts[0].Source.Contains(required.Start()) {
            return Err(Error::Common(format!("first Translation {:?} does not cover start of required range {:?}", &ts[0].Source, required)));
        }

        for i in 0..ts.len() {
            let t = &ts[i];

            if !Addr(t.Source.Start()).IsPageAligned() || !Addr(t.Source.End()).IsPageAligned() {
                return Err(Error::Common(format!("Translation {:?} has unaligned Source", t.Source)))
            }

            if !Addr(t.Offset).IsPageAligned() {
                return Err(Error::Common(format!("Translation {:x} has unaligned Offset", t.Offset)))
            }

            // Translations must be contiguous and in increasing order of
            // Translation.Source.
            if i > 0 && ts[i - 1].Source.End() != t.Source.Start() {
                return Err(Error::Common(format!("Translations {:?} and {:?} are not contiguous",
                                                 ts[i - 1].Source, t.Source)))
            }

            // Translations must be constrained to the optional range.
            if !optional.IsSupersetOf(&t.Source) {
                return Err(Error::Common(format!("Translation {:?} lies outside optional range {:?}",
                                                 t.Source, optional)))
            }

            // Each Translation must permit a superset of requested accesses.
            if !t.Perms.SupersetOf(at) {
                return Err(Error::Common(format!("Translation {:?} does not permit all requested access types {:?}",
                                                 &t.Perms, at)))
            }
        }

        // If the set of Translations does not cover the entire required range,
        // Translate must return a non-nil error explaining why.
        match terr {
            Ok(()) => {
                if ts.len() == 0 {
                    return Err(Error::Common("no Translations and no error".to_string()));
                }

                let t = &ts[ts.len() - 1];
                if !t.Source.Contains(required.End() - 1) {
                    return Err(Error::Common(format!("last Translation {:?} does not reach end of required range {:?}, but Translate returned no error",
                                                     t.Source, required)))
                }
            }
            _ => (),
        }

        return Ok(())
    }
}
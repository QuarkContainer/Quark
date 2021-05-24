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

// Inotify events observable by userspace. These directly correspond to
// filesystem operations and there may only be a single of them per inotify
// event read from an inotify fd.

// IN_ACCESS indicates a file was accessed.
pub const IN_ACCESS : u32 = 0x00000001;
// IN_MODIFY indicates a file was modified.
pub const IN_MODIFY : u32 = 0x00000002;
// IN_ATTRIB indicates a watch target's metadata changed.
pub const IN_ATTRIB : u32 = 0x00000004;
// IN_CLOSE_WRITE indicates a writable file was closed.
pub const IN_CLOSE_WRITE : u32 = 0x00000008;
// IN_CLOSE_NOWRITE indicates a non-writable file was closed.
pub const IN_CLOSE_NOWRITE : u32 = 0x00000010;
// IN_OPEN indicates a file was opened.
pub const IN_OPEN : u32 = 0x00000020;
// IN_MOVED_FROM indicates a file was moved from X.
pub const IN_MOVED_FROM : u32 = 0x00000040;
// IN_MOVED_TO indicates a file was moved to Y.
pub const IN_MOVED_TO : u32 = 0x00000080;
// IN_CREATE indicates a file was created in a watched directory.
pub const IN_CREATE : u32 = 0x00000100;
// IN_DELETE indicates a file was deleted in a watched directory.
pub const IN_DELETE : u32 = 0x00000200;
// IN_DELETE_SELF indicates a watch target itself was deleted.
pub const IN_DELETE_SELF : u32 = 0x00000400;
// IN_MOVE_SELF indicates a watch target itself was moved.
pub const IN_MOVE_SELF : u32 = 0x00000800;
// IN_ALL_EVENTS is a mask for all observable userspace events.
pub const IN_ALL_EVENTS : u32 = 0x00000fff;


// Inotify control events. These may be present in their own events, or ORed
// with other observable events.

// IN_UNMOUNT indicates the backing filesystem was unmounted.
pub const IN_UNMOUNT : u32 = 0x00002000;
// IN_Q_OVERFLOW indicates the event queued overflowed.
pub const IN_Q_OVERFLOW : u32 = 0x00004000;
// IN_IGNORED indicates a watch was removed, either implicitly or through
// inotify_rm_watch(2).
pub const IN_IGNORED : u32 = 0x00008000;
// IN_ISDIR indicates the subject of an event was a directory.
pub const IN_ISDIR : u32 = 0x40000000;



// Feature flags for inotify_add_watch(2).

// IN_ONLYDIR indicates that a path should be watched only if it's a
// directory.
pub const IN_ONLYDIR : u32 = 0x01000000;
// IN_DONT_FOLLOW indicates that the watch path shouldn't be resolved if
// it's a symlink.
pub const IN_DONT_FOLLOW : u32 = 0x02000000;
// IN_EXCL_UNLINK indicates events to this watch from unlinked objects
// should be filtered out.
pub const IN_EXCL_UNLINK : u32 = 0x04000000;
// IN_MASK_ADD indicates the provided mask should be ORed into any existing
// watch on the provided path.
pub const IN_MASK_ADD : u32 = 0x20000000;
// IN_ONESHOT indicates the watch should be removed after one event.
pub const IN_ONESHOT : u32 = 0x80000000;


// Feature flags for inotify_init1(2).

// IN_CLOEXEC is an alias for O_CLOEXEC. It indicates that the inotify
// fd should be closed on exec(2) and friends.
pub const IN_CLOEXEC : u32 = 0x00080000;
// IN_NONBLOCK is an alias for O_NONBLOCK. It indicates I/O syscall on the
// inotify fd should not block.
pub const IN_NONBLOCK : u32 = 0x00000800;


// ALL_INOTIFY_BITS contains all the bits for all possible inotify events. It's
// defined in the Linux source at "include/linux/inotify.h".
pub const ALL_INOTIFY_BITS : u32 = IN_ACCESS | IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE |
    IN_CLOSE_NOWRITE | IN_OPEN | IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE |
    IN_DELETE | IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT | IN_Q_OVERFLOW |
    IN_IGNORED | IN_ONLYDIR | IN_DONT_FOLLOW | IN_EXCL_UNLINK | IN_MASK_ADD |
    IN_ISDIR | IN_ONESHOT;

//! The `io_uring` library for Rust.
//!
//! The crate only provides a summary of the parameters.
//! For more detailed documentation, see manpage.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    non_snake_case
)]

#[macro_use]
pub mod util;
pub mod cqueue;
//pub mod opcode;
pub mod porting;
// mod register;
// pub mod squeue;
// pub mod submit;
pub mod sys;

// pub use self::cqueue::CompletionQueue;
// use self::porting::*;
// pub use self::register::Probe;
// pub use self::squeue::SubmissionQueue;
// pub use self::submit::Submitter;
// use self::util::{Fd, Mmap};
// use super::common::*;
// use super::mutex::*;

// #[derive(Default)]
// pub struct Submission {
//     pub fd: Fd,
//     pub params: Parameters,
//     pub memory: MemoryMap,
//     pub sq: SubmissionQueue,
// }

// #[derive(Default)]
// pub struct Completion {
//     pub cq: CompletionQueue,
// }



// #[allow(dead_code)]
// #[derive(Copy, Clone, Default, Debug)]
// pub struct MemoryMap {
//     pub sq_mmap: Mmap,
//     pub sqe_mmap: Mmap,
//     pub cq_mmap: Option<Mmap>,
// }

// /// IoUring build params
// #[derive(Clone, Default)]
// pub struct Builder {
//     pub dontfork: bool,
//     pub params: sys::io_uring_params,
// }

// #[derive(Clone, Copy, Default, Debug)]
// pub struct Parameters(pub sys::io_uring_params);

// impl Builder {
//     pub fn dontfork(&mut self) -> &mut Self {
//         self.dontfork = true;
//         self
//     }

//     /// Perform busy-waiting for an I/O completion,
//     /// as opposed to getting notifications via an asynchronous IRQ (Interrupt Request).
//     pub fn setup_iopoll(&mut self) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_IOPOLL;
//         self
//     }

//     /// When this flag is specified, a kernel thread is created to perform submission queue polling.
//     /// An io_uring instance configured in this way enables an application to issue I/O
//     /// without ever context switching into the kernel.
//     pub fn setup_sqpoll(&mut self, idle: impl Into<Option<u32>>) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_SQPOLL;
//         self.params.sq_thread_idle = idle.into().unwrap_or(0);
//         self
//     }

//     /// If this flag is specified,
//     /// then the poll thread will be bound to the cpu set in the value.
//     /// This flag is only meaningful when [Builder::setup_sqpoll] is enabled.
//     pub fn setup_sqpoll_cpu(&mut self, n: u32) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_SQ_AFF;
//         self.params.sq_thread_cpu = n;
//         self
//     }

//     /// Create the completion queue with struct `io_uring_params.cq_entries` entries.
//     /// The value must be greater than entries, and may be rounded up to the next power-of-two.
//     pub fn setup_cqsize(&mut self, n: u32) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_CQSIZE;
//         self.params.cq_entries = n;
//         self
//     }

//     pub fn setup_clamp(&mut self) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_CLAMP;
//         self
//     }

//     pub fn setup_attach_wq(&mut self, fd: RawFd) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_ATTACH_WQ;
//         self.params.wq_fd = fd as _;
//         self
//     }

//     #[cfg(feature = "unstable")]
//     pub fn setup_r_disabled(&mut self) -> &mut Self {
//         self.params.flags |= sys::IORING_SETUP_R_DISABLED;
//         self
//     }
// }

// impl Parameters {
//     pub fn is_setup_sqpoll(&self) -> bool {
//         self.0.flags & sys::IORING_SETUP_SQPOLL != 0
//     }

//     pub fn is_setup_iopoll(&self) -> bool {
//         self.0.flags & sys::IORING_SETUP_IOPOLL != 0
//     }

//     /// If this flag is set, the two SQ and CQ rings can be mapped with a single `mmap(2)` call.
//     /// The SQEs must still be allocated separately.
//     /// This brings the necessary `mmap(2)` calls down from three to two.
//     pub fn is_feature_single_mmap(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_SINGLE_MMAP != 0
//     }

//     /// If this flag is set, io_uring supports never dropping completion events. If a completion
//     /// event occurs and the CQ ring is full, the kernel stores the event internally until such a
//     /// time that the CQ ring has room for more entries.
//     pub fn is_feature_nodrop(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_NODROP != 0
//     }

//     /// If this flag is set, applications can be certain that any data for async offload has been consumed
//     /// when the kernel has consumed the SQE
//     pub fn is_feature_submit_stable(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_SUBMIT_STABLE != 0
//     }

//     /// If this flag is set, applications can specify offset == -1 with
//     /// `IORING_OP_{READV,WRITEV}`, `IORING_OP_{READ,WRITE}_FIXED`, and `IORING_OP_{READ,WRITE}`
//     /// to mean current file position, which behaves like `preadv2(2)` and `pwritev2(2)` with offset == -1.
//     /// It’ll use (and update) the current file position.
//     ///
//     /// This obviously comes with the caveat that if the application has multiple reads or writes in flight,
//     /// then the end result will not be as expected.
//     /// This is similar to threads sharing a file descriptor and doing IO using the current file position.
//     pub fn is_feature_rw_cur_pos(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_RW_CUR_POS != 0
//     }

//     /// If this flag is set, then io_uring guarantees that both sync and async execution of
//     /// a request assumes the credentials of the task that called `io_uring_enter(2)` to queue the requests.
//     /// If this flag isn’t set, then requests are issued with the credentials of the task that originally registered the io_uring.
//     /// If only one task is using a ring, then this flag doesn’t matter as the credentials will always be the same.
//     /// Note that this is the default behavior,
//     /// tasks can still register different personalities through
//     /// `io_uring_register(2)` with `IORING_REGISTER_PERSONALITY` and specify the personality to use in the sqe.
//     pub fn is_feature_cur_personality(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_CUR_PERSONALITY != 0
//     }

//     #[cfg(feature = "unstable")]
//     pub fn is_feature_fast_poll(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_FAST_POLL != 0
//     }

//     #[cfg(feature = "unstable")]
//     pub fn is_feature_poll_32bits(&self) -> bool {
//         self.0.features & sys::IORING_FEAT_POLL_32BITS != 0
//     }

//     pub fn sq_entries(&self) -> u32 {
//         self.0.sq_entries
//     }

//     pub fn cq_entries(&self) -> u32 {
//         self.0.cq_entries
//     }
// }

// pub type c_void = u64;

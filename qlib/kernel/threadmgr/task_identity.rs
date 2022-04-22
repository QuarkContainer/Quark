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

use alloc::vec::Vec;

use super::super::super::auth::cap_set::*;
use super::super::super::auth::id::*;
use super::super::super::auth::userns::*;
use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;

impl ThreadInternal {
    pub fn setKUIDsUncheckedLocked(&mut self, newR: KUID, newE: KUID, newS: KUID) {
        let oldcreds = self.creds.clone();
        let root = oldcreds.lock().UserNamespace.MapToKUID(ROOT_UID);

        let oldR = oldcreds.lock().RealKUID;
        let oldE = oldcreds.lock().EffectiveKUID;
        let oldS = oldcreds.lock().SavedKUID;

        self.creds = oldcreds.Fork();
        let mut newcreds = self.creds.lock();

        newcreds.RealKUID = newR;
        newcreds.EffectiveKUID = newE;
        newcreds.SavedKUID = newS;

        // "1. If one or more of the real, effective or saved set user IDs was
        // previously 0, and as a result of the UID changes all of these IDs have a
        // nonzero value, then all capabilities are cleared from the permitted and
        // effective capability sets." - capabilities(7)
        if (oldR == root || oldE == root || oldS == root)
            && (newR != root && newE != root && newS != root)
        {
            // prctl(2): "PR_SET_KEEPCAP: Set the state of the calling thread's
            // "keep capabilities" flag, which determines whether the thread's permitted
            // capability set is cleared when a change is made to the
            // thread's user IDs such that the thread's real UID, effective
            // UID, and saved set-user-ID all become nonzero when at least
            // one of them previously had the value 0.  By default, the
            // permitted capability set is cleared when such a change is
            // made; setting the "keep capabilities" flag prevents it from
            // being cleared." (A thread's effective capability set is always
            // cleared when such a credential change is made,
            // regardless of the setting of the "keep capabilities" flag.)
            if !newcreds.KeepCaps {
                newcreds.PermittedCaps = CapSet(0);
                newcreds.EffectiveCaps = CapSet(0);
            }
        }

        // """
        // 2. If the effective user ID is changed from 0 to nonzero, then all
        // capabilities are cleared from the effective set.
        //
        // 3. If the effective user ID is changed from nonzero to 0, then the
        // permitted set is copied to the effective set.
        // """
        if oldE == root && newE != root {
            newcreds.EffectiveCaps = CapSet(0);
        } else if oldE != root && newE == root {
            newcreds.EffectiveCaps = newcreds.PermittedCaps;
        }

        // "4. If the filesystem user ID is changed from 0 to nonzero (see
        // setfsuid(2)), then the following capabilities are cleared from the
        // effective set: ..."
        // (filesystem UIDs aren't implemented, nor are any of the capabilities in
        // question)

        // Not documented, but compare Linux's kernel/cred.c:commit_creds().
        if oldE != newE {
            self.parentDeathSignal = Signal(0);
        }
    }

    pub fn setKGIDsUncheckedLocked(&mut self, newR: KGID, newE: KGID, newS: KGID) {
        let creds = self.creds.clone();
        let oldE = creds.lock().EffectiveKGID;
        creds.lock().RealKGID = newR;
        creds.lock().EffectiveKGID = newE;
        creds.lock().SavedKGID = newS;

        // Not documented, but compare Linux's kernel/cred.c:commit_creds().
        if oldE != newE {
            self.parentDeathSignal = Signal(0);
        }
    }

    // updateCredsForExec updates t.creds to reflect an execve().
    //
    // NOTE(b/30815691): We currently do not implement privileged executables
    // (set-user/group-ID bits and file capabilities). This allows us to make a lot
    // of simplifying assumptions:
    //
    // - We assume the no_new_privs bit (set by prctl(SET_NO_NEW_PRIVS)), which
    // disables the features we don't support anyway, is always set. This
    // drastically simplifies this function.
    //
    // - We don't implement AT_SECURE, because no_new_privs always being set means
    // that the conditions that require AT_SECURE never arise. (Compare Linux's
    // security/commoncap.c:cap_bprm_set_creds() and cap_bprm_secureexec().)
    //
    // - We don't check for CAP_SYS_ADMIN in prctl(PR_SET_SECCOMP), since
    // seccomp-bpf is also allowed if the task has no_new_privs set.
    //
    // - Task.ptraceAttach does not serialize with execve as it does in Linux,
    // since no_new_privs being set has the same effect as the presence of an
    // unprivileged tracer.
    //
    // Preconditions: t.mu must be locked.
    pub fn updateCredsForExecLocked(&mut self) {
        // """
        // During an execve(2), the kernel calculates the new capabilities of
        // the process using the following algorithm:
        //
        //     P'(permitted) = (P(inheritable) & F(inheritable)) |
        //                     (F(permitted) & cap_bset)
        //
        //     P'(effective) = F(effective) ? P'(permitted) : 0
        //
        //     P'(inheritable) = P(inheritable)    [i.e., unchanged]
        //
        // where:
        //
        //     P         denotes the value of a thread capability set before the
        //               execve(2)
        //
        //     P'        denotes the value of a thread capability set after the
        //               execve(2)
        //
        //     F         denotes a file capability set
        //
        //     cap_bset  is the value of the capability bounding set
        //
        // ...
        //
        // In order to provide an all-powerful root using capability sets, during
        // an execve(2):
        //
        // 1. If a set-user-ID-root program is being executed, or the real user ID
        // of the process is 0 (root) then the file inheritable and permitted sets
        // are defined to be all ones (i.e. all capabilities enabled).
        //
        // 2. If a set-user-ID-root program is being executed, then the file
        // effective bit is defined to be one (enabled).
        //
        // The upshot of the above rules, combined with the capabilities
        // transformations described above, is that when a process execve(2)s a
        // set-user-ID-root program, or when a process with an effective UID of 0
        // execve(2)s a program, it gains all capabilities in its permitted and
        // effective capability sets, except those masked out by the capability
        // bounding set.
        // """ - capabilities(7)
        // (ambient capability sets omitted)
        //
        // As the last paragraph implies, the case of "a set-user-ID root program
        // is being executed" also includes the case where (namespace) root is
        // executing a non-set-user-ID program; the actual check is just based on
        // the effective user ID.
        let mut newPermitted: CapSet = CapSet::default();
        let mut fileEffective = false;
        let root = self.creds.lock().UserNamespace.MapToKUID(ROOT_UID);
        let EffectiveKUID = self.creds.lock().EffectiveKUID;
        let RealKUID = self.creds.lock().RealKUID;
        if EffectiveKUID == root || RealKUID == root {
            let InheritableCaps = self.creds.lock().InheritableCaps;
            let BoundingCaps = self.creds.lock().BoundingCaps;
            newPermitted.0 = InheritableCaps.0 | BoundingCaps.0;
            if EffectiveKUID == root {
                fileEffective = true
            }
        }

        self.creds = self.creds.Fork();
        // Now we enter poorly-documented, somewhat confusing territory. (The
        // accompanying comment in Linux's security/commoncap.c:cap_bprm_set_creds
        // is not very helpful.) My reading of it is:
        //
        // If at least one of the following is true:
        //
        // A1. The execing task is ptraced, and the tracer did not have
        // CAP_SYS_PTRACE in the execing task's user namespace at the time of
        // PTRACE_ATTACH.
        //
        // A2. The execing task shares its FS context with at least one task in
        // another thread group.
        //
        // A3. The execing task has no_new_privs set.
        //
        // AND at least one of the following is true:
        //
        // B1. The new effective user ID (which may come from set-user-ID, or be the
        // execing task's existing effective user ID) is not equal to the task's
        // real UID.
        //
        // B2. The new effective group ID (which may come from set-group-ID, or be
        // the execing task's existing effective group ID) is not equal to the
        // task's real GID.
        //
        // B3. The new permitted capability set contains capabilities not in the
        // task's permitted capability set.
        //
        // Then:
        //
        // C1. Limit the new permitted capability set to the task's permitted
        // capability set.
        //
        // C2. If either the task does not have CAP_SETUID in its user namespace, or
        // the task has no_new_privs set, force the new effective UID and GID to
        // the task's real UID and GID.
        //
        // But since no_new_privs is always set (A3 is always true), this becomes
        // much simpler. If B1 and B2 are false, C2 is a no-op. If B3 is false, C1
        // is a no-op. So we can just do C1 and C2 unconditionally.
        let EffectiveKGID = self.creds.lock().EffectiveKGID;
        let RealKGID = self.creds.lock().RealKGID;
        if EffectiveKUID != RealKUID || EffectiveKGID != RealKGID {
            self.creds.lock().EffectiveKUID = RealKUID;
            self.creds.lock().EffectiveKGID = RealKGID;
            self.parentDeathSignal = Signal(0);
        }

        // (Saved set-user-ID is always set to the new effective user ID, and saved
        // set-group-ID is always set to the new effective group ID, regardless of
        // the above.)
        self.creds.lock().SavedKUID = RealKUID;
        self.creds.lock().SavedKGID = RealKGID;
        self.creds.lock().PermittedCaps.0 &= newPermitted.0;
        if fileEffective {
            let PermittedCaps = self.creds.lock().PermittedCaps;
            self.creds.lock().EffectiveCaps = PermittedCaps
        } else {
            self.creds.lock().EffectiveCaps = CapSet(0);
        }

        // prctl(2): The "keep capabilities" value will be reset to 0 on subsequent
        // calls to execve(2).
        self.creds.lock().KeepCaps = false;

        // "The bounding set is inherited at fork(2) from the thread's parent, and
        // is preserved across an execve(2)". So we're done.
    }

    pub fn setUserNamespace(&mut self, ns: &UserNameSpace) -> Result<()> {
        let t = self;

        // "A process reassociating itself with a user namespace must have the
        // CAP_SYS_ADMIN capability in the target user namespace." - setns(2)
        //
        // If t just created ns, then t.creds is guaranteed to have CAP_SYS_ADMIN
        // in ns (by rule 3 in auth.Credentials.HasCapability).
        if !t.creds.HasCapabilityIn(Capability::CAP_SYS_ADMIN, ns) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        t.creds = t.creds.Fork();
        t.creds.lock().UserNamespace = ns.clone();
        // "The child process created by clone(2) with the CLONE_NEWUSER flag
        // starts out with a complete set of capabilities in the new user
        // namespace. Likewise, a process that creates a new user namespace using
        // unshare(2) or joins an existing user namespace using setns(2) gains a
        // full set of capabilities in that namespace."
        t.creds.lock().PermittedCaps = ALL_CAP;
        t.creds.lock().InheritableCaps = CapSet(0);
        t.creds.lock().EffectiveCaps = ALL_CAP;
        t.creds.lock().BoundingCaps = ALL_CAP;
        // "A call to clone(2), unshare(2), or setns(2) using the CLONE_NEWUSER
        // flag sets the "securebits" flags (see capabilities(7)) to their default
        // values (all flags disabled) in the child (for clone(2)) or caller (for
        // unshare(2), or setns(2)." - user_namespaces(7)
        t.creds.lock().KeepCaps = false;

        return Ok(());
    }

    pub fn hasCapabilityIn(&self, cp: u64, ns: &UserNameSpace) -> bool {
        return self.creds.HasCapabilityIn(cp, ns);
    }

    pub fn hasCapability(&self, cp: u64) -> bool {
        return self.creds.HasCapability(cp);
    }
}

impl Thread {
    pub fn Credentials(&self) -> Credentials {
        return self.lock().creds.clone();
    }

    pub fn UserNamespace(&self) -> UserNameSpace {
        return self.lock().creds.lock().UserNamespace.clone();
    }

    pub fn HasCapabilityIn(&self, cp: u64, ns: &UserNameSpace) -> bool {
        return self.lock().creds.HasCapabilityIn(cp, ns);
    }

    pub fn HasCapability(&self, cp: u64) -> bool {
        return self.lock().creds.HasCapability(cp);
    }

    pub fn SetUID(&self, uid: UID) -> Result<()> {
        if !uid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mut t = self.lock();
        let kuid = t.creds.lock().UserNamespace.MapToKUID(uid);
        if !kuid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // "setuid() sets the effective user ID of the calling process. If the
        // effective UID of the caller is root (more precisely: if the caller has
        // the CAP_SETUID capability), the real UID and saved set-user-ID are also
        // set." - setuid(2)
        if t.creds.HasCapability(Capability::CAP_SETUID) {
            t.setKUIDsUncheckedLocked(kuid, kuid, kuid);
            return Ok(());
        }

        // "EPERM: The user is not privileged (Linux: does not have the CAP_SETUID
        // capability) and uid does not match the real UID or saved set-user-ID of
        // the calling process."
        let ruid = t.creds.lock().RealKUID;
        let suid = t.creds.lock().SavedKUID;

        if kuid != ruid && kuid != suid {
            return Err(Error::SysError(SysErr::EPERM));
        }

        t.setKUIDsUncheckedLocked(ruid, kuid, suid);
        return Ok(());
    }

    pub fn SetREUID(&self, r: UID, e: UID) -> Result<()> {
        let mut t = self.lock();
        let creds = t.creds.clone();

        // "Supplying a value of -1 for either the real or effective user ID forces
        // the system to leave that ID unchanged." - setreuid(2)
        let mut newR = creds.lock().RealKUID;
        if r.Ok() {
            newR = creds.lock().UserNamespace.MapToKUID(r);
            if !newR.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let mut newE = creds.lock().EffectiveKUID;
        if e.Ok() {
            newE = creds.lock().UserNamespace.MapToKUID(e);
            if !newE.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let real = creds.lock().RealKUID;
        let effective = creds.lock().EffectiveKUID;
        let save = creds.lock().SavedKUID;
        if !creds.HasCapability(Capability::CAP_SETUID) {
            // "Unprivileged processes may only set the effective user ID to the
            // real user ID, the effective user ID, or the saved set-user-ID."
            if newE != real && newE != effective && newE != save {
                return Err(Error::SysError(SysErr::EPERM));
            }
            // "Unprivileged users may only set the real user ID to the real user
            // ID or the effective user ID."
            if newR != real && newR != effective {
                return Err(Error::SysError(SysErr::EPERM));
            }
        }

        // "If the real user ID is set (i.e., ruid is not -1) or the effective user
        // ID is set to a value not equal to the previous real user ID, the saved
        // set-user-ID will be set to the new effective user ID."
        let mut newS = creds.lock().SavedKUID;
        if r.Ok() || (e.Ok() && newE != effective) {
            newS = newE;
        }

        t.setKUIDsUncheckedLocked(newR, newE, newS);
        return Ok(());
    }

    pub fn SetRESUID(&self, r: UID, e: UID, s: UID) -> Result<()> {
        let mut t = self.lock();
        let creds = t.creds.clone();

        // "Unprivileged user processes may change the real UID, effective UID, and
        // saved set-user-ID, each to one of: the current real UID, the current
        // effective UID or the current saved set-user-ID. Privileged processes (on
        // Linux, those having the CAP_SETUID capability) may set the real UID,
        // effective UID, and saved set-user-ID to arbitrary values. If one of the
        // arguments equals -1, the corresponding value is not changed." -
        // setresuid(2)
        let mut newR = creds.lock().RealKUID;
        if r.Ok() {
            newR = creds.UseUID(r)?
        }

        let mut newE = creds.lock().EffectiveKUID;
        if e.Ok() {
            newE = creds.UseUID(e)?
        }

        let mut newS = creds.lock().SavedKUID;
        if s.Ok() {
            newS = creds.UseUID(s)?
        }

        t.setKUIDsUncheckedLocked(newR, newE, newS);
        return Ok(());
    }

    // SetGID implements the semantics of setgid(2).
    pub fn SetGID(&self, gid: GID) -> Result<()> {
        if !gid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mut t = self.lock();
        let creds = t.creds.clone();

        let kgid = creds.lock().UserNamespace.MapToKGID(gid);
        if !kgid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if creds.HasCapability(Capability::CAP_SETGID) {
            t.setKGIDsUncheckedLocked(kgid, kgid, kgid);
            return Ok(());
        }

        let r = creds.lock().RealKGID;
        let s = creds.lock().SavedKGID;

        if kgid != r || kgid != s {
            return Err(Error::SysError(SysErr::EPERM));
        }

        t.setKGIDsUncheckedLocked(r, kgid, s);
        return Ok(());
    }

    // SetREGID implements the semantics of setregid(2).
    pub fn SetREGID(&self, r: GID, e: GID) -> Result<()> {
        let mut t = self.lock();
        let creds = t.creds.clone();
        let userns = creds.lock().UserNamespace.clone();

        let mut newR = creds.lock().RealKGID;
        if r.Ok() {
            newR = userns.MapToKGID(r);
            if !newR.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let mut newE = creds.lock().EffectiveKGID;
        if e.Ok() {
            newE = userns.MapToKGID(e);
            if !newE.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let RealKGID = creds.lock().RealKGID;
        let EffectiveKGID = creds.lock().EffectiveKGID;
        let SavedKGID = creds.lock().SavedKGID;
        if !creds.HasCapability(Capability::CAP_SETGID) {
            if newE.0 != RealKGID.0 && newE.0 != EffectiveKGID.0 && newE.0 != SavedKGID.0 {
                return Err(Error::SysError(SysErr::EPERM));
            }

            if newR.0 != RealKGID.0 && newR.0 != EffectiveKGID.0 {
                return Err(Error::SysError(SysErr::EPERM));
            }
        }

        let mut newS = SavedKGID;
        if r.Ok() || (e.Ok() && newE != EffectiveKGID) {
            newS = newE;
        }

        t.setKGIDsUncheckedLocked(newR, newE, newS);
        return Ok(());
    }

    // SetRESGID implements the semantics of the setresgid(2) syscall.
    pub fn SetRESGID(&self, r: GID, e: GID, s: GID) -> Result<()> {
        let mut t = self.lock();
        let creds = t.creds.clone();

        let mut newR = creds.lock().RealKGID;
        if r.Ok() {
            newR = creds.UseGID(r)?;
            if !newR.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let mut newE = creds.lock().EffectiveKGID;
        if e.Ok() {
            newE = creds.UseGID(e)?;
            if !newE.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        let mut newS = creds.lock().SavedKGID;
        if s.Ok() {
            newS = creds.UseGID(s)?;
            if !newS.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }

        t.setKGIDsUncheckedLocked(newR, newE, newS);
        return Ok(());
    }

    pub fn SetExtraGIDs(&self, gids: &[GID]) -> Result<()> {
        let mut t = self.lock();
        info!("SetExtraGIDs 1");
        if !t.creds.HasCapability(Capability::CAP_SETGID) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        info!("SetExtraGIDs 2");
        let mut kgids = Vec::with_capacity(gids.len());
        let userns = t.creds.lock().UserNamespace.clone();
        for gid in gids {
            let kgid = userns.MapToKGID(*gid);
            if !kgid.Ok() {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            kgids.push(kgid)
        }

        info!("SetExtraGIDs 3");
        t.creds = t.creds.Fork();
        t.creds.lock().ExtraKGIDs = kgids;
        return Ok(());
    }

    pub fn SetCapabilitySets(
        &self,
        permitted: CapSet,
        inheritable: CapSet,
        effective: CapSet,
    ) -> Result<()> {
        let mut t = self.lock();

        // "Permitted: This is a limiting superset for the effective capabilities
        // that the thread may assume." - capabilities(7)
        if effective.0 & !permitted.0 != 0 {
            return Err(Error::SysError(SysErr::EPERM));
        }

        // "It is also a limiting superset for the capabilities that may be added
        // to the inheritable set by a thread that does not have the CAP_SETPCAP
        // capability in its effective set."
        let InheritableCaps = t.creds.lock().InheritableCaps;
        let PermittedCaps = t.creds.lock().PermittedCaps;
        let BoundingCaps = t.creds.lock().BoundingCaps;
        if !t.creds.HasCapability(Capability::CAP_SETPCAP)
            && (inheritable.0 & !(inheritable.0 | PermittedCaps.0)) != 0
        {
            return Err(Error::SysError(SysErr::EPERM));
        }

        // "If a thread drops a capability from its permitted set, it can never
        // reacquire that capability (unless it execve(2)s ..."
        if permitted.0 & !PermittedCaps.0 != 0 {
            return Err(Error::SysError(SysErr::EPERM));
        }

        // "... if a capability is not in the bounding set, then a thread can't add
        // this capability to its inheritable set, even if it was in its permitted
        // capabilities ..."
        if inheritable.0 & !(InheritableCaps.0 | BoundingCaps.0) != 0 {
            return Err(Error::SysError(SysErr::EPERM));
        }

        t.creds = t.creds.Fork();
        t.creds.lock().PermittedCaps = permitted;
        t.creds.lock().InheritableCaps = inheritable;
        t.creds.lock().EffectiveCaps = effective;

        let task = Task::GetTask(t.taskId);
        task.creds = t.creds.clone();

        return Ok(());
    }

    pub fn DropBoundingCapability(&self, cp: u64) -> Result<()> {
        let mut t = self.lock();
        if !t.creds.HasCapability(Capability::CAP_SETPCAP) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        t.creds = t.creds.Fork();
        t.creds.lock().BoundingCaps.0 &= !CapSetOf(cp).0;
        return Ok(());
    }

    pub fn SetUserNamespace(&self, ns: &UserNameSpace) -> Result<()> {
        let mut t = self.lock();

        return t.setUserNamespace(ns);
    }

    // SetKeepCaps will set the keep capabilities flag PR_SET_KEEPCAPS.
    pub fn SetKeepCaps(&self, k: bool) {
        let mut t = self.lock();
        t.creds = t.creds.Fork();
        t.creds.lock().KeepCaps = k;
    }
}

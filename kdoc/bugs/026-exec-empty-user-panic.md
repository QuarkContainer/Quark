# 026 — quark exec panic on empty --user

## Symptom

`quark exec` without `--user` panics:

```
index out of bounds: the len is 1 but the index is 1
```

at `qvisor/src/runc/cmd/exec.rs:215`.

## Cause

Default `--user` is `""`. UID parse failure used `ids[1]` in the error path when only `ids[0]` exists.

## Fix

Treat empty `--user` as uid/gid 0. Return `Err` instead of panic on malformed uid/gid strings.

## Files

- `qvisor/src/runc/cmd/exec.rs`

## Verify

```bash
quark create test -b /tmp/bundle && quark start test
quark exec test -- /bin/true
quark delete --force test
```

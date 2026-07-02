# 035 — CRI pause sandbox booted with all host vCPUs

**Tags:** upstream

## Symptom

`crictl run` creates the pod sandbox VM but guest kernel panics in `vcpu_mgr.rs` (`get panic : state is 0`) during multi-vCPU bring-up. With 1 vCPU, Start hangs on `StartRootContainer`.

## Cause

1. CRI pause containers have no `cpu.quota`; Quark defaulted to all host vCPUs → scheduler race on vCPU 1.
2. Guest boot only starts the control socket accept loop on vCPU 1; a 1-vCPU sandbox never runs `ControllerProcess`.

## Fix

- Sandboxed pods with no CPU limit: boot with 1 vCPU (bypass `min_vcpu_amount=2`).
- qkernel: when `vcpuCnt == 1`, run bootstrap + `ControllerProcess` on vCPU 0 instead of vCPU 1.

## Files

- `qvisor/src/runc/runtime/vm_type/{noncc,emulcc,sevsnp}.rs`
- `qkernel/src/lib.rs`

## Verify

```bash
sudo crictl run container.json pod.json
keska-lab-network-preflight
```

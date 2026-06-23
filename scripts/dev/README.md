# Quark Development CLI

macOS development workflow for Quark using a Lima VM (`quark`). All runtime work (build, Docker, KVM) runs **inside the VM** — not Docker Desktop.

**Entry point:** `python3 quark.py` from repo root.

```
repo root/
  quark.py                  ← Python CLI (dev + bench)
  lima-quark.yaml           ← Lima VM definition
  benchmark/                ← perf benchmarks + JSON results
  scripts/dev/
    README.md               ← this file
    bin/                    ← VM shell scripts (limactl, rsync, docker)
```

## Prerequisites

```bash
brew install lima
```

Python 3 is bundled with macOS. Optional tools: `fswatch` (sync --watch), `entr` (watch build).

The Lima VM must mount this repo. Default in [`lima-quark.yaml`](../../lima-quark.yaml):

```yaml
mounts:
  - location: "~"
    writable: true
```

## First-time setup

```bash
python3 quark.py vm create
python3 quark.py dev start
python3 quark.py docker setup
python3 quark.py install-config
python3 quark.py rebuild
python3 quark.py test smoke
```

## Daily workflow

```bash
python3 quark.py vm start           # if stopped overnight
python3 quark.py rebuild            # after code changes
python3 quark.py logs tail          # in another terminal
python3 quark.py run python
python3 quark.py vm stop            # when done
```

Debug session:

```bash
python3 quark.py dev loop           # config:debug → rebuild → clear logs → hello → show logs
```

Run `python3 quark.py help` or see [benchmark/README.md](../../benchmark/README.md).

## Commands

### Workflows

| Command | Description |
|---------|-------------|
| `quark.py help` | Grouped command reference |
| `quark.py quick` | Short cheat sheet |
| `quark.py doctor` | Full preflight (Lima, KVM, docker, binaries) |
| `quark.py dev start` | `vm start` + `diag all` |
| `quark.py dev stop` / `dev down` | Stop Lima VM |
| `quark.py dev up` | Alias for `dev start` |
| `quark.py dev loop` | End-to-end debug workflow |

### VM

| Command | Description |
|---------|-------------|
| `quark.py vm start` | Start Lima VM |
| `quark.py vm stop` | Stop Lima VM |
| `quark.py vm restart` | Restart (refreshes docker group) |
| `quark.py vm shell` | Interactive shell in VM |
| `quark.py vm status` | Show VM state |
| `quark.py vm create` | Create VM from `lima-quark.yaml` |
| `quark.py vm delete --confirm` | Delete VM |

### Build

| Command | Description |
|---------|-------------|
| `quark.py sync` | rsync mount → `~/quark` (excludes target, build, .git) |
| `quark.py sync --watch` | Auto-sync on change (needs fswatch) |
| `quark.py build` | `make debug` (default) |
| `quark.py build release` | `make release` |
| `quark.py build qvisor` / `qkernel` / `vdso` | Partial rebuilds |
| `quark.py install` | `sudo make install` |
| `quark.py rebuild` | sync + build + install + docker restart |
| `quark.py build cleanall --confirm` | Full clean |

### Run (inside VM)

| Command | Description |
|---------|-------------|
| `quark.py run hello` | hello-world smoke test |
| `quark.py run python` | Python 3.12 snippet |
| `quark.py run shell` | Interactive ubuntu bash |
| `quark.py run ubuntu --cmd "uname -a"` | One-shot command |
| `quark.py run exec -- --rm -it ubuntu:24.04 bash` | Arbitrary docker run |
| `quark.py run busybox` | Memory overhead probe |

### Logs & config

| Command | Description |
|---------|-------------|
| `quark.py logs tail` | Follow `/var/log/quark/quark.log` |
| `quark.py logs cat --lines 500` | Print recent lines |
| `quark.py logs clear --confirm` | Clear log directory |
| `quark.py config show` | Print `/etc/quark/config.json` |
| `quark.py config debug` | Set DebugLevel to Debug |
| `quark.py config quiet` | Set DebugLevel to Error |
| `quark.py config shim --shim off` | Toggle ShimMode |
| `quark.py install-config` | Copy repo `config.json` to VM |

### Docker

| Command | Description |
|---------|-------------|
| `quark.py docker setup` | Merge runtimes from `doc/daemon.json` |
| `quark.py docker restart` | Restart docker in VM |
| `quark.py docker info` | Show docker info |
| `quark.py docker pull --image ubuntu:24.04` | Pull image in VM |

### Diagnostics & tests

| Command | Description |
|---------|-------------|
| `quark.py diag all` | Run all checks |
| `quark.py diag kvm` | Verify `/dev/kvm` |
| `quark.py diag docker` | Verify quark runtime registered |
| `quark.py test smoke` | hello-world + python |
| `quark.py test startup` | runc vs quark timing |
| `quark.py test rust` | Build `test/rust` |
| `quark.py test c --test-name fork` | Run C test in container |

### Watch

| Command | Description |
|---------|-------------|
| `quark.py watch logs` | Alias for `logs tail` |
| `quark.py watch ps` | Loop docker ps + quark processes |
| `quark.py watch build` | Auto-rebuild on source change (needs entr) |

### Benchmarks

See [benchmark/README.md](../../benchmark/README.md).

## Environment variables

```bash
RUNTIME=quark python3 quark.py run hello
VERBOSE=1 python3 quark.py rebuild
CONFIRM=1 python3 quark.py logs clear
AUTO_START=0 python3 quark.py run hello
SKIP_SYNC=1 python3 quark.py build
```

## Scripts (`bin/`)

| Script | Role |
|--------|------|
| `vm-exec.sh` | Core primitive — run any command in Lima VM |
| `sync-repo.sh` | rsync virtiofs mount → `~/quark` |
| `preflight.sh` | Host + VM checks for `quark.py doctor` |
| `docker-runtime.sh` | Register quark runtimes in daemon.json |
| `config-level.sh` | Set DebugLevel / ShimMode |
| `log-tail.sh` | Tail / cat / clear quark logs |
| `help.sh` | Text for `quark.py help` |
| `diag-*.sh` | Individual diagnostic checks |
| `test-*.sh` | Benchmark and C test helpers |

## Architecture

```
macOS host                         Lima VM "quark"
──────────                         ────────────────
python3 quark.py
  └─ scripts/dev/bin/vm-exec.sh ──►  ~/quark (build tree)
                                      docker + quark_d
```

Build tree lives at `~/quark` on VM disk (fast). Source is synced from the virtiofs mount of your Mac home directory.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `unknown runtime quark_d` | You ran docker on macOS. Use `quark.py run …` or `quark.py vm shell`. |
| `permission denied` on docker.sock | Run `quark.py vm restart` after docker group changes. |
| Build is slow | Run `quark.py sync` — ensure build uses `~/quark`, not mount. |
| `diag docker` fails | Run `quark.py docker setup` then `quark.py docker restart`. |

## Extending

- Add commands in `quark.py` (argparse subcommands).
- Add VM logic in `scripts/dev/bin/` — keep Python thin, shell handles limactl/docker quoting.

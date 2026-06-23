#!/usr/bin/env bash
# Print grouped help for the Quark dev CLI.
set -euo pipefail

mode="${1:-full}"

if [[ "$mode" == "quick" ]]; then
  cat <<'EOF'
Quark Development CLI

  python3 quark.py vm start
  python3 quark.py rebuild
  python3 quark.py run hello
  python3 quark.py bench run
  python3 quark.py doctor

  python3 quark.py help       full command groups
  python3 quark.py --help     argparse reference

Docs: scripts/dev/README.md  |  benchmark/README.md
EOF
  exit 0
fi

cat <<'EOF'
Quark Development CLI — command groups

  python3 quark.py <command> ...

Getting started
  quark.py dev start        Start VM + run diagnostics
  quark.py doctor           Full environment check
  quark.py vm create        First-time Lima VM setup
  quark.py docker setup     Register quark / quark_d runtimes
  quark.py install-config   Install config.json
  quark.py rebuild          Sync → build → install → docker restart
  quark.py test smoke       hello-world + python

VM (Lima)
  quark.py vm start|stop|restart|shell|status
  quark.py vm create        Create from lima-quark.yaml
  quark.py vm delete --confirm

Build & install
  quark.py sync [--watch]
  quark.py build [debug|release|qvisor|qkernel|vdso|cuda|snp|clean|cleanall]
  quark.py install
  quark.py rebuild

Run containers (inside VM — not Docker Desktop)
  quark.py run hello|python|shell|ubuntu|busybox|compare
  quark.py run exec -- --rm -it ubuntu:24.04 bash

Benchmarks
  quark.py bench run                  dev profile (all suites)
  quark.py bench run --profile stress
  quark.py bench run --suite cold_start
  quark.py bench setup
  quark.py bench results
  quark.py bench compare a.json b.json

Logs & config
  quark.py logs tail
  quark.py logs cat --lines 500
  quark.py logs clear --confirm
  quark.py config show|debug|quiet|shim
  quark.py docker setup|restart|info|pull

Diagnostics & workflows
  quark.py diag all
  quark.py dev loop             Debug session workflow
  quark.py watch logs|ps|build

Variables: RUNTIME=quark_d  SKIP_SYNC=1  CONFIRM=1  LINES=500  IMAGE=ubuntu:24.04

Full reference: scripts/dev/README.md  |  benchmark/README.md
EOF

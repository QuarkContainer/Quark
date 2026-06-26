"""IPython magics."""

from __future__ import annotations

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from keska_lab.display import console


@magics_class
class KeskaLabMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.lab = shell.user_ns.get("lab")

    @line_magic
    def lab(self, line: str = "") -> None:
        if line.strip() == "ping":
            console.print(self.lab.ping())
            return
        console.print(f"[bold]Host[/bold]     {self.lab.config.ssh_target}")
        console.print(f"[bold]Local repo[/bold] {self.lab.config.resolve_local_repo()}")
        if self.lab.last_report:
            r = self.lab.last_report
            console.print(
                f"[bold]Last bench[/bold] {r.backend}/{r.profile} — "
                f"p50 {r.stats.p50} ms ({r.stats.samples} samples)"
            )

    @line_magic
    def probe(self, line: str = "") -> dict:
        info = self.lab.probe()
        for name, data in info["backends"].items():
            mark = "[green]ready[/green]" if data.get("ready") else "[yellow]not ready[/yellow]"
            console.print(f"  {name}: {mark}  {data}")
        return info

    @line_magic
    def backend(self, line: str = "") -> None:
        name = line.strip() or "quark"
        env = self.lab.use(name)
        console.print(f"Active: [cyan]{name}[/cyan] — {env.probe()}")

    @line_magic
    def bootstrap(self, line: str = "") -> None:
        """Install Rust + build deps on lab: %bootstrap"""
        from keska_lab.display import print_setup_report
        from keska_lab.provision import ProvisionContext, ensure_apt_deps, ensure_rust
        from keska_lab.setup.base import SetupReport, StepResult
        import time

        cfg = self.lab.config
        ctx = ProvisionContext(self.lab.remote, cfg, cfg.resolve_local_repo(), stream=True)
        report = SetupReport(pipeline="bootstrap")
        for name, fn in [("ensure-apt-deps", lambda: ensure_apt_deps(ctx)), ("ensure-rust", lambda: ensure_rust(ctx))]:
            t0 = time.perf_counter()
            try:
                msg = fn()
                report.steps.append(StepResult(name, True, msg, time.perf_counter() - t0))
            except Exception as e:
                report.steps.append(StepResult(name, False, str(e), time.perf_counter() - t0))
                break
        print_setup_report(report)
        report.raise_if_failed()

    @line_magic
    def build(self, line: str = "") -> None:
        """Build/deploy Quark: %build"""
        self.lab.quark.run(stream=True)

    @magic_arguments()
    @argument("mode", nargs="?", default="tti")
    @argument("-n", type=int, default=None)
    @argument("--wave-size", type=int, default=None)
    @argument("--waves", type=int, default=None)
    @argument("--image", default=None)
    @argument("-b", "--backend", default="quark", help="quark | kata")
    @argument("--no-setup", action="store_true", help="skip setup pipeline")
    @argument("-q", action="store_true")
    @line_magic
    def bench(self, line: str = "") -> None:
        args = parse_argstring(self.bench, line)
        self.lab.bench(
            args.mode,
            backend=args.backend,
            n=args.n,
            wave_size=args.wave_size,
            waves=args.waves,
            image=args.image,
            setup=not args.no_setup,
            verbose=not args.q,
        )

    @line_magic
    def setup(self, line: str = "") -> None:
        """Run setup pipeline: %setup quark | %setup kata"""
        name = line.strip() or "quark"
        env = self.lab.use(name)
        if name == "quark":
            env.prepare(stream=True)
        else:
            env.run(stream=True)

    @line_magic
    def cleanup(self, line: str = "") -> None:
        self.lab.cleanup()
        console.print("Cleanup done.")

    @line_magic
    def remote(self, line: str = "") -> str:
        if not line.strip():
            console.print("Usage: %remote <command>")
            return ""
        out = self.lab.ssh(line, stream=True)
        console.print(out)
        return out


def register_magics(shell) -> None:
    shell.register_magics(KeskaLabMagics)

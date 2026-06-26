"""IPython lab shell."""

from __future__ import annotations

import sys

from IPython.terminal.embed import InteractiveShellEmbed
from IPython.terminal.ipapp import load_default_config
from rich.console import Console

from keska_lab.magics import register_magics
from keska_lab.session import LabSession

console = Console()

BANNER = """
[bold cyan]Keska Lab[/bold cyan] — IPython sandbox laboratory @ lab.keska.vpn

  [bold]Quark[/bold]
    lab.quark.run()              build + install on lab (streams logs)
    lab.quark.bench("tti", n=100)

  [bold]Kata[/bold]
    lab.kata.run()
    lab.kata.bench("calm", n=10)

  [bold]CLI[/bold]
    keska-lab-provision          same as lab.quark.run(), from terminal

  [bold]Magics[/bold]
    %probe  %build  %setup quark  %bench tti  %remote docker ps
"""


def main() -> None:
    lab = LabSession()
    cfg = load_default_config()

    user_ns = {
        "lab": lab,
        "session": lab,
    }

    shell = InteractiveShellEmbed(config=cfg, user_ns=user_ns)
    register_magics(shell)

    console.print(BANNER)
    try:
        r = lab.remote.ping()
        if r.ok:
            console.print(f"[green]SSH OK[/green]  {lab.config.ssh_target}")
            console.print(f"Local repo: {lab.config.resolve_local_repo()}")
        else:
            console.print(f"[yellow]SSH warning:[/yellow] {r.stderr.strip()}")
    except Exception as e:
        console.print(f"[yellow]SSH warning:[/yellow] {e}")

    console.print("")
    shell()


if __name__ == "__main__":
    sys.exit(main() or 0)

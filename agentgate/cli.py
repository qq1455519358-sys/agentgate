"""AgentGate CLI — command-line interface.

Entry point: ``agentgate`` (installed via pyproject.toml console_scripts).

Commands:
  - approve   — interactively review and approve recorded entries
  - export    — export approved records to golden dataset
  - run       — execute a test suite from a Python module
  - info      — show version and configuration
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

try:
    import click
except ImportError:
    print(
        "AgentGate CLI requires 'click'. Install it with: pip install click",
        file=sys.stderr,
    )
    sys.exit(1)


@click.group()
@click.version_option(package_name="agentgate", prog_name="agentgate")
def cli() -> None:
    """AgentGate — AI Agent pre-deployment quality gate. 🚀"""
    pass


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "-f", "--file",
    default="agentgate_records.jsonl",
    show_default=True,
    help="Path to the records JSONL file.",
)
@click.option(
    "-o", "--output",
    default=None,
    help="Export approved entries to this golden JSONL file.",
)
@click.option(
    "--auto", "auto_approve",
    is_flag=True,
    default=False,
    help="Auto-approve all pending entries without review.",
)
def approve(file: str, output: str | None, auto_approve: bool) -> None:
    """Review and approve recorded agent interactions."""
    from .recorder import approve as do_approve

    count = do_approve(file, output, auto_approve_all=auto_approve)
    if count:
        click.echo(f"\n🎉 {count} entries approved.")
    else:
        click.echo("Nothing to approve.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "-f", "--file",
    default="agentgate_records.jsonl",
    show_default=True,
    help="Path to the records JSONL file.",
)
@click.option(
    "-o", "--output",
    default="golden.jsonl",
    show_default=True,
    help="Output golden dataset file.",
)
def export(file: str, output: str) -> None:
    """Export approved records to a golden dataset JSONL."""
    from .recorder import RecordStore

    store = RecordStore(file)
    count = store.export_golden(output)
    click.echo(f"📦 Exported {count} approved entries to {output}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("module_path")
@click.option(
    "--suite", "suite_attr",
    default="suite",
    show_default=True,
    help="Attribute name of the TestSuite in the module.",
)
@click.option(
    "--agent", "agent_attr",
    default=None,
    help="Attribute name of the agent function in the module.",
)
@click.option(
    "--junit", "junit_path",
    default="agentgate-report.xml",
    show_default=True,
    help="JUnit XML output path.",
)
@click.option(
    "--tag", "tags",
    multiple=True,
    help="Filter tests by tag (repeatable).",
)
@click.option(
    "--pass-k", "pass_k",
    type=int,
    default=None,
    help="Run each test K times (PassK strategy).",
)
def run(
    module_path: str,
    suite_attr: str,
    agent_attr: str | None,
    junit_path: str,
    tags: tuple[str, ...],
    pass_k: int | None,
) -> None:
    """Run a test suite from a Python module.

    MODULE_PATH is a dotted Python path like 'tests.test_agent'.

    Example:
        agentgate run tests.test_agent --suite suite --agent my_agent
    """
    # Add cwd to path so local modules are importable
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        click.echo(f"❌ Cannot import module '{module_path}': {exc}", err=True)
        sys.exit(1)

    suite_obj = getattr(mod, suite_attr, None)
    if suite_obj is None:
        click.echo(
            f"❌ Module '{module_path}' has no attribute '{suite_attr}'",
            err=True,
        )
        sys.exit(1)

    agent_fn = None
    if agent_attr:
        agent_fn = getattr(mod, agent_attr, None)
        if agent_fn is None:
            click.echo(
                f"❌ Module '{module_path}' has no attribute '{agent_attr}'",
                err=True,
            )
            sys.exit(1)

    from .ci import run_regression

    tag_list = list(tags) if tags else None
    result = run_regression(
        suite_obj,
        agent_fn,
        junit_path=junit_path,
        tags=tag_list,
        pass_k=pass_k,
    )
    sys.exit(0 if result.all_passed else 1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@cli.command()
def info() -> None:
    """Show AgentGate version and environment info."""
    import platform

    click.echo("AgentGate — AI Agent Quality Gate 🛡️\n")
    click.echo(f"  Python:    {platform.python_version()}")
    click.echo(f"  Platform:  {platform.platform()}")

    # Check optional dependencies
    deps = {
        "sentence-transformers": "sentence_transformers",
        "numpy": "numpy",
        "openai": "openai",
        "click": "click",
    }
    click.echo("\n  Dependencies:")
    for name, module in deps.items():
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "installed")
            click.echo(f"    ✅ {name}: {ver}")
        except ImportError:
            click.echo(f"    ❌ {name}: not installed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()

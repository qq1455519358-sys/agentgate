"""AgentGate recorder — capture agent I/O for golden dataset creation.

Provides:
  - @record decorator  — wraps agent functions to log inputs/outputs to JSONL
  - approve()          — interactive CLI to review and approve recorded entries
  - RecordStore        — manage the JSONL record file programmatically
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .types import RecordEntry

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_RECORD_PATH = Path("agentgate_records.jsonl")


# ---------------------------------------------------------------------------
# RecordStore — low-level JSONL management
# ---------------------------------------------------------------------------

class RecordStore:
    """Append-only JSONL store for agent interaction records."""

    def __init__(self, path: str | Path = _DEFAULT_RECORD_PATH) -> None:
        self.path = Path(path)

    def append(self, entry: RecordEntry) -> None:
        """Append a single record entry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._to_dict(entry), ensure_ascii=False) + "\n")

    def load_all(self) -> list[RecordEntry]:
        """Load all entries from the file."""
        if not self.path.exists():
            return []
        entries: list[RecordEntry] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                entries.append(self._from_dict(obj))
        return entries

    def overwrite(self, entries: list[RecordEntry]) -> None:
        """Rewrite the entire file with the given entries."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(self._to_dict(entry), ensure_ascii=False) + "\n")

    def export_golden(self, output_path: str | Path) -> int:
        """Export only approved entries as a golden dataset JSONL.

        Returns the number of exported entries.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        entries = [e for e in self.load_all() if e.approved]
        with output.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(
                    json.dumps(
                        {
                            "input": entry.input,
                            "expected": entry.output,
                            "name": f"golden_{entry.id}",
                            "meta": entry.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(entries)

    @staticmethod
    def _to_dict(entry: RecordEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "input": entry.input,
            "output": entry.output,
            "timestamp": entry.timestamp,
            "agent_name": entry.agent_name,
            "meta": entry.meta,
            "approved": entry.approved,
        }

    @staticmethod
    def _from_dict(obj: dict[str, Any]) -> RecordEntry:
        return RecordEntry(
            id=obj.get("id", ""),
            input=obj["input"],
            output=obj["output"],
            timestamp=obj.get("timestamp", 0.0),
            agent_name=obj.get("agent_name", ""),
            meta=obj.get("meta", {}),
            approved=obj.get("approved", False),
        )


# ---------------------------------------------------------------------------
# @record decorator
# ---------------------------------------------------------------------------

def record(
    path: str | Path = _DEFAULT_RECORD_PATH,
    agent_name: str = "",
    capture_meta: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[F], F]:
    """Decorator to record agent function calls to a JSONL file.

    The wrapped function must accept a string as its first positional argument
    and return a string.

    Args:
        path: Output JSONL file path.
        agent_name: Label for the agent (defaults to function name).
        capture_meta: Optional callable that receives the same args/kwargs
                      and returns extra metadata dict.

    Usage::

        @record(path="my_records.jsonl", agent_name="chatbot")
        def my_agent(query: str) -> str:
            return "some response"
    """
    store = RecordStore(path)

    def decorator(fn: F) -> F:
        name = agent_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(input_text: str, *args: Any, **kwargs: Any) -> str:
            output = fn(input_text, *args, **kwargs)
            meta = capture_meta(input_text, *args, **kwargs) if capture_meta else {}
            entry = RecordEntry(
                input=input_text,
                output=str(output),
                agent_name=name,
                meta=meta,
                timestamp=time.time(),
            )
            store.append(entry)
            return output

        # Attach store reference for programmatic access
        wrapper._record_store = store  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# approve() — interactive review
# ---------------------------------------------------------------------------

def approve(
    path: str | Path = _DEFAULT_RECORD_PATH,
    output_path: str | Path | None = None,
    *,
    auto_approve_all: bool = False,
) -> int:
    """Interactively review recorded entries and mark them as approved.

    Args:
        path: Path to the records JSONL file.
        output_path: If given, export approved entries to this golden JSONL.
        auto_approve_all: Skip interactive review, approve everything.

    Returns:
        Number of newly approved entries.
    """
    store = RecordStore(path)
    entries = store.load_all()
    pending = [e for e in entries if not e.approved]

    if not pending:
        print("No pending entries to review.")
        if output_path:
            count = store.export_golden(output_path)
            print(f"Exported {count} approved entries to {output_path}")
        return 0

    print(f"\n📋 {len(pending)} pending entries to review\n")
    approved_count = 0

    for i, entry in enumerate(pending, 1):
        if auto_approve_all:
            entry.approved = True
            approved_count += 1
            continue

        print(f"--- [{i}/{len(pending)}] {entry.agent_name or 'agent'} ---")
        print(f"  INPUT:  {entry.input}")
        print(f"  OUTPUT: {entry.output}")
        print()

        while True:
            choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit? ").strip().lower()
            if choice in ("a", "approve"):
                entry.approved = True
                approved_count += 1
                print("  ✅ Approved\n")
                break
            elif choice in ("r", "reject"):
                # Mark as explicitly rejected by removing from list
                entries.remove(entry)
                print("  ❌ Rejected (removed)\n")
                break
            elif choice in ("s", "skip"):
                print("  ⏭️  Skipped\n")
                break
            elif choice in ("q", "quit"):
                print("  Quitting review.\n")
                store.overwrite(entries)
                if output_path:
                    count = store.export_golden(output_path)
                    print(f"Exported {count} approved entries to {output_path}")
                return approved_count
            else:
                print("  Invalid choice. Use a/r/s/q.")

    store.overwrite(entries)
    print(f"\n✅ Approved {approved_count} entries")

    if output_path:
        count = store.export_golden(output_path)
        print(f"📦 Exported {count} total approved entries to {output_path}")

    return approved_count

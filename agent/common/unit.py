"""The unit as the platform mounts it at `--task-dir`: `instruction.md`, `card.toml`, `manifest.json`,
`environment/Dockerfile` and every file under `environment/data/`, with `checks/` sealed (docs/challenge.md, "What
/input holds"). Read once, with the instruction's canary lines already out."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import canary


@dataclass(frozen=True)
class Unit:
    task_dir: Path
    instruction: str            # instruction.md with every canary line removed
    dropped: int                # the lines removed
    card: dict
    canaries: frozenset[str]    # every GUID-shaped token in the unit's input tree (canary.collect)
    dockerfile: str             # environment/Dockerfile, "" where absent

    @property
    def card_path(self) -> Path:
        return self.task_dir / "card.toml"

    @property
    def data_dir(self) -> Path:
        return self.task_dir / "environment" / "data"


def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read(task_dir: Path) -> Unit:
    task_dir = Path(task_dir)
    card = canary.read_card(task_dir / "card.toml")
    raw = text(task_dir / "instruction.md")
    dockerfile = text(task_dir / "environment" / "Dockerfile")
    known = canary.collect(task_dir)
    instruction, dropped = canary.strip(raw, known)
    return Unit(task_dir, instruction, dropped, card, known, dockerfile)

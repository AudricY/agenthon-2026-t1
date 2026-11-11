"""Maps each path the instruction names to where that file really is on the platform.

Step: 1.
Called by: agent/n11/solve.py (`build` in `N11.solve`), common/contract.py (`outputs`) and agent/n11/commands.py.
Start at: `build`, and `PathMap.text` for what the model is told.

The rules in detail
-------------------
The path map: where each path the instruction names is on this filesystem.

The instructions name the unit image's paths (`/app/data/<x>`, a bare `/app/<file>`), and the unit's
`environment/Dockerfile` `COPY` lines say which data file each one is, under another name where the copy renames it.
On the platform none of those paths exists: the unit is at `--task-dir`, read-only, and the output is `--out`. So each
named path is looked up by the `COPY` lines, then by name under the data directory, and stated to the model as "where
the instruction says X, read Y"; a path with no file is said to have none, and a script the instruction asks to be
saved at `/app/<name>.py` is answered by the runner, which saves the model's script under /tmp.
"""
from __future__ import annotations

import json
import posixpath
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

OUTPUTS = ("/app/output", "/output")
NAMED = re.compile(r"(?<![\w.~$-])(/(?:app|output|input)(?:/[^\s`'\"()<>\[\]{},;|*$\\]*)?)")
SCRIPTS = (".py", ".sh", ".ipynb")
# A deliverable's kind by its extension (a script's is none): a file name the instruction writes without a path, of
# one of these kinds, is read as an output's
KINDS = ("json", "jsonl", "csv", "tsv", "txt", "md", "html", "xml", "yaml", "yml", "parquet", "pqt", "xlsx", "png",
         "svg", "npy", "npz", "pkl")
BARE = re.compile(r"(?<![\w./~$-])([A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\.(?:" + "|".join(KINDS) + r"))\b")


@dataclass(frozen=True)
class Entry:
    named: str              # the path as the instruction names it, normalised
    kind: str               # "read", "output", "script" or "missing"
    path: Path | None       # the file or directory here; for "output", the file in --out

    def line(self) -> str:
        if self.kind == "read":
            what = "the directory" if self.path.is_dir() else "the file"
            return f"- where the instruction says `{self.named}`, read {what} `{self.path}`"
        if self.kind == "script":
            return (f"- `{self.named}`: the instruction asks for a script saved there, and that filesystem is "
                    f"read-only; do not save it anywhere: the script in your reply is saved under /tmp and run with "
                    f"python3 for you")
        if self.kind == "missing":
            return f"- `{self.named}`: no such file exists in this unit"
        return f"- where the instruction says `{self.named}`, write `{self.path}`"


@dataclass(frozen=True)
class PathMap:
    entries: tuple[Entry, ...]
    out: Path
    data_dir: Path

    def text(self) -> str:
        lines = ["The instruction's paths are those of the unit's own image, not this machine's. Here:",
                 f"- the output directory is `{self.out}`: write every output file directly into it (where the "
                 f"instruction says `/app/output/X` or `/output/X`, write `{self.out}/X`), and nothing anywhere else "
                 f"but /tmp",
                 f"- every input file is under `{self.data_dir}` (listed in full below)"]
        lines += [entry.line() for entry in self.entries if entry.kind != "output"]
        return "\n".join(lines)


def logical_lines(dockerfile: str) -> list[str]:
    """The Dockerfile's instructions, continuation lines joined and comments dropped."""
    out, pending = [], ""
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        out.append(pending + line)
        pending = ""
    if pending.strip():
        out.append(pending)
    return out


def arguments(rest: str) -> list[str]:
    rest = rest.strip()
    if rest.startswith("["):
        try:
            parsed = json.loads(rest)
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        except ValueError:
            pass
    try:
        return shlex.split(rest)
    except ValueError:
        return rest.split()


def copies(dockerfile: str, context: Path) -> dict[str, Path]:
    """Each container path a `COPY` or `ADD` line places, file or directory, mapped to its source under the build
    context (`environment/`); a relative destination resolves against the `WORKDIR` before it."""
    placed: dict[str, Path] = {}
    workdir = "/"
    for line in logical_lines(dockerfile):
        word, _, rest = line.partition(" ")
        word = word.upper()
        if word == "WORKDIR" and rest.strip():
            workdir = posixpath.normpath(posixpath.join(workdir, arguments(rest)[0]))
            continue
        if word not in ("COPY", "ADD"):
            continue
        args = [arg for arg in arguments(rest) if not arg.startswith("--")]
        if len(args) < 2:
            continue
        *sources, dest = args
        into = dest.endswith("/") or len(sources) > 1
        target = posixpath.normpath(posixpath.join(workdir, dest))
        for source in sources:
            found = sorted(context.glob(source.rstrip("/"))) if any(c in source for c in "*?[") else \
                [context / source.rstrip("/")]
            for host in found:
                if host.is_dir():
                    placed[target] = host
                    for inner in sorted(host.rglob("*")):
                        placed[posixpath.join(target, inner.relative_to(host).as_posix())] = inner
                elif host.is_file():
                    placed[posixpath.join(target, host.name) if into or len(found) > 1 else target] = host
    return placed


def named(instruction: str) -> list[str]:
    """Every `/app`, `/output` and `/input` path the instruction names, once each, in order, normalised; `/app`
    alone is the image's root and not a path to map."""
    seen: dict[str, None] = {}
    for match in NAMED.finditer(instruction):
        path = posixpath.normpath(match.group(1).rstrip(".,:;!?"))
        if path != "/app":
            seen.setdefault(path, None)
    return list(seen)


def under(path: str, root: str) -> str | None:
    """`path` relative to `root`, "" for the root itself; None where it is not under it."""
    if path == root:
        return ""
    return path[len(root) + 1:] if path.startswith(root + "/") else None


def build(task_dir: Path, instruction: str, dockerfile: str, out: Path) -> PathMap:
    task_dir, out = Path(task_dir), Path(out)
    data_dir = task_dir / "environment" / "data"
    placed = copies(dockerfile, task_dir / "environment")
    files = [path for path in sorted(data_dir.rglob("*"))] if data_dir.is_dir() else []
    entries = []
    for path in named(instruction):
        entries.append(locate(path, task_dir, placed, files, out))
    return PathMap(tuple(entries), out, data_dir)


def locate(path: str, task_dir: Path, placed: dict[str, Path], files: list[Path], out: Path) -> Entry:
    for root in OUTPUTS:
        rest = under(path, root)
        if rest is not None:
            return Entry(path, "output", out / rest if rest else out)
    rest = under(path, "/input")
    if rest is not None:
        here = task_dir / rest
        return Entry(path, "read", here) if here.exists() else Entry(path, "missing", None)
    if path in placed and placed[path].exists():
        return Entry(path, "read", placed[path])
    for root in sorted((p for p in placed if placed[p].is_dir()), key=len, reverse=True):
        rest = under(path, root)
        if rest and (placed[root] / rest).exists():
            return Entry(path, "read", placed[root] / rest)
    inside = [placed[p].parent for p in placed if under(p, path)]
    if inside and len(set(inside)) == 1:
        return Entry(path, "read", inside[0])          # a directory its files were copied into one by one
    name = posixpath.basename(path)
    same = [file for file in files if file.name == name]
    if len(same) == 1:
        return Entry(path, "read", same[0])
    if name.endswith(SCRIPTS):
        return Entry(path, "script", None)
    return Entry(path, "missing", None)


def outputs(instruction: str, inputs: Iterable[str] = ()) -> list[str]:
    """The output files the instruction names, relative to the output directory, once each, in the order first named:
    each `/app/output/<name>` and `/output/<name>` path, and each file name of a deliverable's kind (`KINDS`) written
    without a path ("### File 2: prices.csv", "`summary.json`") that is not an input's: neither one of `inputs` (the
    data files' names) nor the last part of another path the instruction names."""
    found: dict[str, int] = {}
    for match in NAMED.finditer(instruction):
        path = posixpath.normpath(match.group(1).rstrip(".,:;!?"))
        for root in OUTPUTS:
            rest = under(path, root)
            if rest:
                found.setdefault(rest, match.start())
    taken = {posixpath.basename(path) for path in named(instruction)
             if all(under(path, root) is None for root in OUTPUTS)}
    taken |= {posixpath.basename(str(name)) for name in inputs}
    taken |= {posixpath.basename(name) for name in found}
    for match in BARE.finditer(instruction):
        name = match.group(1)
        if name not in taken:
            found.setdefault(name, match.start())
    return sorted(found, key=found.get)

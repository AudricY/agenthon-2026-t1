"""The run's environment pre-empting the mechanical crashes (lane clock1 of 10-13; the owner's in707 and in709,
agenthon-t2; reader `time`, docs/research/time-1013.md section 3: 44 of the 154 failing runs crashed on a path, a
removed API, a numpy or pandas value given to `json`, or a standard module never imported, each pre-emptable at no
request).

Under `AGENTHON_N11_SHIM=on` (agent/n11/solve.py) each run's script starts with agent/n11/preload.py loaded before it,
as the interpreter's `sitecustomize`: the workspace's `shim/` directory (beside work/, never under it) holds it and
goes first on the run's `PYTHONPATH`; an interpreter's own `sitecustomize`, which that name hides, runs after it
(`CHAIN_ENV`). `environ` writes the run's spec there (the path table, commands.run_table, one table with the write
tool's forms; the standard names the script's files use unbound, `free`; the kinds), and `fired` reads back what the
run's shims did, which `said` turns into the run's observation's lines, one a kind, after its head: the model is told
what ran in its place, so it learns nothing false about the sandbox. Nothing else in the run changes: its limit, its
exit's reading, the observation's form. Standard library only.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from n11 import lint

KINDS = ("paths", "api", "json", "imports")
STANDARD = {"np": "numpy", "pd": "pandas", "qf": "qf", "scipy": "scipy", "arch_model": "arch:arch_model"}  # the
#                                         runtime card's names a file may use unbound, each with its module (`:` its
#                                         attribute)
DIR = "shim"                    # under the workspace's root, beside work/
SPEC, REPORT = "spec.json", "fired.jsonl"
SPEC_ENV = "N11_SHIM"           # the run's spec, as preload.py reads it
CHAIN_ENV = "N11_SHIM_CHAIN"    # the interpreter's own sitecustomize, run after the shim
PRELOAD = Path(__file__).resolve().parent / "preload.py"
SHOWN = 4                       # the mappings or calls a line names, at most: the harness's bound, never said
SAID = {
    "paths": "shim: paths in the unit image's form were opened here for this run: {what} (the task's paths section "
             "names this machine's)",
    "api": "shim: calls the runtime's versions removed ran as before for this run: {what}; write the new form",
    "json": "shim: json wrote numpy and pandas values as plain ones for this run ({what}); json alone refuses them: "
            "convert them (float(x), int(x), x.tolist(), x.isoformat())",
    "imports": "shim: names used and never imported were imported for this run: {what}; write the imports",
}


def free(files: list[Path], root: Path) -> dict[str, list[str]]:
    """The runtime card's standard names (`STANDARD`) each of `files` uses and never binds (agent/n11/lint.py `free`,
    a `*` import from a file beside it read through it), each with the files, as `work/<name>.py` under `root`'s
    work/."""
    found: dict[str, list[str]] = {}
    modules = {path.stem: path.read_text(encoding="utf-8", errors="replace")
               for path in sorted({path.parent for path in files}.pop().glob("*.py"))} if files else {}
    for path in files:
        try:
            tree = lint.parsed(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if tree is None:
            continue
        for name in lint.free(tree, modules):
            if name in STANDARD:
                found.setdefault(name, []).append(path.relative_to(root).as_posix())
    return found


def chained() -> str:
    """The interpreter's own `sitecustomize`, where it has one, which the shim's hides: run after it."""
    try:
        spec = importlib.util.find_spec("sitecustomize")
    except (ImportError, ValueError):
        return ""
    return str(spec.origin) if spec is not None and spec.origin and Path(spec.origin).is_file() else ""


def preloaded(root: Path, env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    """`root`'s `shim/` and the run's environment (`env`, the child's, otherwise as it is) loading preload.py first:
    the file there as `sitecustomize`, the directory first on `PYTHONPATH` (once, where the shim or the failure
    snapshot, agent/n11/snapshot.py, put it already), the interpreter's own `sitecustomize` named to run after it."""
    here = Path(root) / DIR
    here.mkdir(parents=True, exist_ok=True)
    loaded = here / "sitecustomize.py"
    if not loaded.is_file():
        loaded.write_text(PRELOAD.read_text(encoding="utf-8"), encoding="utf-8")
    env = dict(env)
    paths = env.get("PYTHONPATH", "")
    if paths.split(os.pathsep)[0] != str(here):
        env["PYTHONPATH"] = os.pathsep.join(part for part in (str(here), paths) if part)
    other = chained()
    if other:
        env[CHAIN_ENV] = other
    return here, env


def environ(root: Path, env: dict[str, str], table: tuple[list[tuple[str, str]], list[tuple[str, str]]],
            scripts: str, work: Path, names: dict[str, list[str]]) -> dict[str, str]:
    """The run's environment with the shim (`env`, the child's, otherwise as it is): preload.py loaded first
    (`preloaded`); the spec beside it (`table`, the path table and the rows read after the scripts' form,
    commands.run_table; `scripts`, the image's form of a script saved under work/; the standard `names` to import); its
    report emptied."""
    here, env = preloaded(root, env)
    (here / REPORT).unlink(missing_ok=True)
    spec = {"kinds": list(KINDS), "table": [list(pair) for pair in table[0]],
            "after": [list(pair) for pair in table[1]], "scripts": scripts, "work": str(work),
            "names": {name: STANDARD[name] for name in names}, "report": str(here / REPORT)}
    (here / SPEC).write_text(json.dumps(spec), encoding="utf-8")
    env[SPEC_ENV] = str(here / SPEC)
    return env


def fired(root: Path) -> list[tuple[str, str]]:
    """What the run's shims did, in order, each once: (kind, what); `environ` empties the report before each run."""
    path = Path(root) / DIR / REPORT
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    found: dict[tuple[str, str], None] = {}
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("kind") in KINDS:
            found[(str(item["kind"]), str(item.get("what", "")))] = None
    return list(found)


def said(found: list[tuple[str, str]], names: dict[str, list[str]]) -> list[str]:
    """The run's observation's lines, one a kind that fired, in `KINDS`' order: the paths mapped, the calls run as
    before and their new forms, the values json wrote plainly, the names imported and the files using them."""
    lines = []
    for kind in KINDS:
        whats = [what for each, what in found if each == kind][:SHOWN]
        if not whats:
            continue
        if kind == "paths":
            what = ", ".join(f"`{old.strip()}` as `{new.strip()}`" for old, _, new in
                             (each.partition(" -> ") for each in whats))
        elif kind == "api":
            what = "; ".join(f"`{old}` as {new}" for old, _, new in (each.partition("|") for each in whats))
        elif kind == "json":
            what = ", ".join(whats)
        else:
            what = "; ".join(f"`{name}` ({module.replace(':', '.')}) in {', '.join(names.get(name, ['work/']))}"
                             for name, _, module in (each.partition("|") for each in whats))
        lines.append(SAID[kind].format(what=what))
    return lines

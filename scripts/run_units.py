"""Runs the agent image over the organisers' Track 1 units the way the platform runs it, then scores each unit with
the unit's own checker, as step 6 of the kit's README does.

    python scripts/run_units.py --image agenthon-2026-t1:n11 [--kit vendor/track1-coding-public]
                                [--units NAME ...] [--jobs N] [--results DIR] [--base IMAGE]

The model is reached through MODEL_ENDPOINT, MODEL_NAME and MODEL_TOKEN, the variables the platform injects (see
REPRODUCE.md). The client settings in CLIENT_VARIABLES are forwarded too where they are set, for a hosted route that
differs from the platform's (OpenRouter's own field for thinking off, the platform's pace). Each container receives
every variable by name, so the token never appears on a command line. Needs Docker and the organisers'
qfbench2-common package, whose `split_unit` builds the tree the agent sees: the unit without its checks and reference
data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_VARIABLES = ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN")
CLIENT_VARIABLES = ("AGENTHON_MODEL_REASONING_FIELD", "AGENTHON_MODEL_PROVIDER_PIN", "AGENTHON_MODEL_MIN_INTERVAL_S",
                    "AGENTHON_MODEL_RATE_WAIT_S", "AGENTHON_MODEL_RATE_ATTEMPTS", "AGENTHON_MODEL_RATE_UNCHARGED",
                    "AGENTHON_MODEL_RATE_UNTIMED")
# The organisers' container settings: a read-only root, uid 65534 and a small /tmp.
PLATFORM = ("--read-only", "--user", "65534:65534", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--pids-limit", "256", "--ulimit", "nofile=1024:1024",
            "--ulimit", "nproc=256:256")
CHECKER_SECONDS = 1800
MEMORY_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def memory_bytes(text: str) -> int:
    text = str(text).strip().upper().removesuffix("IB").removesuffix("B")
    number, unit = (text[:-1], text[-1]) if text[-1:].isalpha() else (text, "")
    return int(float(number) * MEMORY_UNITS[unit])


def participant_tree(unit: Path, work: Path) -> tuple[Path, Path]:
    """The tree mounted at /input for the agent, and the complete copy the checker reads."""
    from qfbench2_common.dataset import split_unit

    split_unit(unit.resolve(), work / "mounted", work / "grader", "coding")
    mounted, grader = work / "mounted" / unit.name, work / "grader" / unit.name
    for tree in (mounted, grader):
        for path in [tree, *tree.rglob("*")]:   # the split leaves 0700 directories; uid 65534 must read them
            path.chmod(0o755 if path.is_dir() else 0o644)
    return mounted, grader


def run_container(argv: list[str], name: str, seconds: float) -> tuple[int | None, float]:
    """The exit code (None if the clock ran out and the container was killed) and the seconds it ran."""
    start = time.monotonic()
    try:
        done = subprocess.run(["docker", "run", "--rm", "--name", name, *argv], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=seconds)
        return done.returncode, time.monotonic() - start
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return None, time.monotonic() - start


def run_unit(unit: Path, image: str, base: str, work: Path) -> dict:
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    timeout = float(card["agent"]["timeout_sec"])
    environment = card.get("environment", {})
    cpus = min(float(environment.get("cpus", os.cpu_count() or 1)), float(os.cpu_count() or 1))
    memory = memory_bytes(environment.get("memory", "4G"))
    mounted, grader = participant_tree(unit, work)
    out = work / "output"
    out.mkdir()
    out.chmod(0o777)
    mounts = ["-v", f"{out}:/app/output", "-v", f"{out}:/output"]

    agent_argv = [*PLATFORM, "--cpus", f"{cpus:g}", "--memory", str(memory), "--memory-swap", str(memory),
                  "-v", f"{mounted}:/input:ro", *mounts, "-e", "QFBENCH_SEED=0", "-e", "QFBENCH_NETWORK=restricted",
                  *(word for name in MODEL_VARIABLES for word in ("-e", name)),
                  *(word for name in CLIENT_VARIABLES if os.environ.get(name) for word in ("-e", name)),
                  image, "solve", "--task-dir", "/input", "--out", "/app/output"]
    agent_exit, agent_seconds = run_container(agent_argv, f"t1-agent-{uuid.uuid4().hex[:12]}", timeout)

    checker_argv = ["--network", "none", "-e", "OUTPUT_DIR=/app/output", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-v", f"{grader}:/input:ro", *mounts, base, "bash", "/input/checks/test.sh"]
    checker_exit, _ = run_container(checker_argv, f"t1-check-{uuid.uuid4().hex[:12]}", CHECKER_SECONDS)

    reward_file = out / "reward.json"
    reward = json.loads(reward_file.read_text(encoding="utf-8")).get("reward") if reward_file.is_file() else None
    # Hand back to this user what the containers' users wrote, so the work directory can be removed.
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "0:0", "-v", f"{work}:/work", base,
                    "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/work"], capture_output=True)
    return {"unit": unit.name, "passed": reward == 1, "reward": reward, "agent_exit": agent_exit,
            "timed_out": agent_exit is None, "agent_seconds": round(agent_seconds, 1), "timeout_sec": timeout,
            "checker_exit": checker_exit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="the agent image, for example agenthon-2026-t1:n11")
    parser.add_argument("--kit", default=str(ROOT / "vendor" / "track1-coding-public"),
                        help="the organisers' kit, as scripts/fetch_vendor.sh checks it out")
    parser.add_argument("--units", nargs="+", help="unit names under the kit's units/ (default: every unit)")
    parser.add_argument("--jobs", type=int, default=4, help="units run at once (default 4)")
    parser.add_argument("--results", default=str(ROOT / "results"), help="where each run's results.jsonl goes")
    parser.add_argument("--base", default="finance-bench-sandbox:latest", help="the image the checkers run in")
    args = parser.parse_args()

    missing = [name for name in MODEL_VARIABLES if not os.environ.get(name)]
    if missing:
        sys.exit(f"set {', '.join(missing)} first (REPRODUCE.md, step 4)")
    units_dir = Path(args.kit) / "units"
    names = args.units or sorted(path.name for path in units_dir.iterdir() if (path / "card.toml").is_file())
    units = [units_dir / name for name in names]
    if absent := [unit.name for unit in units if not (unit / "card.toml").is_file()]:
        sys.exit(f"no such unit under {units_dir}: {', '.join(absent)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = Path(args.results) / stamp
    record.mkdir(parents=True)
    rows_file = record / "results.jsonl"
    print(f"{len(units)} units, {args.jobs} at a time; results in {rows_file}", flush=True)

    def one(unit: Path) -> dict:
        work = Path(tempfile.mkdtemp(prefix=f"t1-{unit.name}-"))
        try:
            return run_unit(unit, args.image, args.base, work)
        except Exception as error:   # a unit that cannot run is recorded as failed and the run goes on
            return {"unit": unit.name, "passed": False, "error": f"{type(error).__name__}: {error}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    passed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool, rows_file.open("w", encoding="utf-8") as rows:
        for done, row in enumerate(pool.map(one, units), start=1):
            rows.write(json.dumps(row) + "\n")
            rows.flush()
            passed += bool(row["passed"])
            print(f"[{done}/{len(units)}] {row['unit']}: {'pass' if row['passed'] else 'fail'}", flush=True)
    print(f"passed {passed} of {len(units)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

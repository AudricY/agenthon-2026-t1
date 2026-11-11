# n11: a-team's Agenthon 2026 Track 1 agent

This is the code of a-team's entry in Track 1 (Coding) of Agenthon 2026, a competition in the NeurIPS 2026
Competition Track. The entry placed first of 61 teams, with a verified Final score of 0.1786. Five teams scored
0.1786, and the published tie-break, the earlier upload, ranked this entry first.

The agent, `n11`, solves a quantitative-finance coding task in a short loop with the competition's model,
`nvidia/nemotron-3-super-120b-a12b`: the model writes Python, the agent runs it and reports back, and the model writes
again. The scripts can import a library of 316 functions written for the entry.

## Results

| Units | Run | Result |
|---|---|---|
| The organisers' private Final units | their verification rerun | 0.1786 |
| The organisers' private Final units | Codabench's run of the upload | 0.1071 |
| The kit's 86 public units | the Final image, 2026-11-03 | 27 of 86 |
| The kit's 86 public units | this repository's `main`, 2026-11-07 | 25 of 86 |

The two Final scores come from the same image, and the organisers did not publish why their rerun scored higher. The
model's answers vary from run to run: two identical runs once differed on 10 of 86 units.

## How it works

A unit is an instruction, a task card and data files. The agent must write the output files the instruction names,
and hidden checks grade them. `agent/n11/solve.py` runs each unit in five steps:

1. **Read the unit.** The instruction, with its canary lines removed, a digest of the data files, a map from the
   instruction's paths to the mounted files, and the list of output files it names (`agent/common/`).
2. **Pick library functions.** One request shows the task and an index of the library (`agent/qf/`). The reply names
   up to 8 functions, and their documentation goes into the loop's system prompt (`agent/n11/selection.py`).
3. **Draft once.** One request asks for a complete script, which runs once. Its files are held aside and written out
   only if the loop leaves nothing to grade (`agent/n11/drafts.py`).
4. **Loop.** The model replies with whole files in fenced blocks and `run` or `finish` lines. The agent saves and
   runs them, then answers with the ends of the output, the traceback and checks on the output files: that they
   exist, parse and hold finite numbers, and the rules the task states (`agent/n11/commands.py`, `props.py`,
   `sanity.py`). A finish after a failing run is turned away up to three times.
5. **Close.** The loop stops at a finish, at 20 requests, or when the clock runs out. Any output file that carries a
   canary is removed, and the process exits 0 (`agent/common/guard.py`).

Every request uses temperature 0, a fixed seed and at most 4,000 output tokens. Thinking is off except in step 2.
The clock is the card's timeout less 200 s, at most 600 s. `agent/model.py` is the model client.

[docs/architecture.md](docs/architecture.md) maps the modules, gives a reading order and defines the code's words.

## This code and the Final image

`main` is the Final image's agent with the code its settings never run removed: the Final carried 56 switches and 17
setting files from earlier experiments, and ran one value of each. The agent's Python went from 31,672 lines to
10,575, and the library is unchanged. The Final image's own code is this repository's first commit, tagged
`final-image`, and `git diff final-image main` shows every change.

Three checks tie `main` to the Final image:

- `scripts/verify_image.sh` finds the library, its index, the tokenizer, the `solve` wrapper and every Python
  package identical to the Final image's.
- A replay check fed both agents the model replies recorded in two live runs, offline and in order, and compared
  every request they sent. With the clock's readings masked, `main` sent the same requests and wrote the same files
  as the Final's code on 151 of the 152 units where the Final's code agreed with itself across two replays. Two files
  differ between any two runs, a plot's random ids and a test report's timings, and were left out. The exception is
  a unit whose script ran for 85 to 89 s against its 90 s limit, so whether it was stopped depended on the machine's
  load. On the same unit in the other recording, two replays of the Final's code split the same way.
- On the kit's 86 units, `main` solved 25, against the Final image's 27.

To build the Final image's exact files, check out `final-image` and follow its README.

## Reproducing

You need Linux on x86-64, Docker with BuildKit, git, curl, about 5 GB of disk, and Python 3.13 or later. Running units
also needs a key for an OpenAI-compatible endpoint that serves `nvidia/nemotron-3-super-120b-a12b`, and the
organisers' `qfbench2-common` package:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.6.0#subdirectory=common"
```

### Build and verify

```sh
scripts/fetch_vendor.sh
docker build --platform linux/amd64 -t agenthon-2026-t1:n11 .
scripts/verify_image.sh agenthon-2026-t1:n11
```

`fetch_vendor.sh` puts what it fetches under `vendor/`, which git ignores. It clones the organisers' Track 1 kit at
commit `84049db`, the commit the Final's base was built from, and builds the kit's sandbox base. It then pins the
base's packages to the Final image's (`docker/base.lock`), since the kit's Dockerfile asks only for minimum versions.
Last, it downloads the model's tokenizer from Hugging Face and the `tokenizers` 0.23.3 wheel from PyPI, each checked
against its sha256. The base is tagged `finance-bench-sandbox:latest`; set `BASE_TAG` to use another name.

On 2026-11-07 the check printed:

```
library, index, tokenizer and solve wrapper: identical (56 files)
python packages: identical
agent code: 31672 lines of Python in the Final image, 10575 built
```

The image's Python is 3.13.18, where the Final's was 3.13.16, because the kit's base starts from `python:3.13-slim`.
Since that check `main` has gained docstring text and documents only, so a build of it prints a larger count for
the same code.

### Run the units

The platform hands each container `MODEL_ENDPOINT`, `MODEL_NAME` and `MODEL_TOKEN`, and `scripts/run_units.py` does
the same, by name, so the key never appears on a command line. These settings use OpenRouter:

```sh
export MODEL_ENDPOINT=https://openrouter.ai/api/v1
export MODEL_NAME=nvidia/nemotron-3-super-120b-a12b
export MODEL_TOKEN=...                                  # your OpenRouter key
export AGENTHON_MODEL_REASONING_FIELD=1
export AGENTHON_MODEL_PROVIDER_PIN='{"allow_fallbacks": true}'
export AGENTHON_MODEL_MIN_INTERVAL_S=44
export AGENTHON_MODEL_RATE_WAIT_S=15 AGENTHON_MODEL_RATE_ATTEMPTS=6
export AGENTHON_MODEL_RATE_UNCHARGED=1 AGENTHON_MODEL_RATE_UNTIMED=1

python3 scripts/run_units.py --image agenthon-2026-t1:n11 --jobs 4
```

The `AGENTHON_MODEL_*` lines make OpenRouter behave like the platform's model, and the runner forwards them:

- `REASONING_FIELD=1`: OpenRouter drops `chat_template_kwargs`, the field that turns the model's thinking off, so the
  client also sends OpenRouter's own, `reasoning: {"enabled": false}`.
- `MIN_INTERVAL_S=44`: at least 44 s between two requests of a unit. The platform's model took 37 to 46 s at the
  median, so the 600 s clock allows about as many turns as it did there.
- `RATE_*`: on a 429 the client waits and tries again, up to 6 attempts. The retries do not count against the 20
  requests or the clock, since the platform has no such limit.
- `PROVIDER_PIN`: lets OpenRouter fall back to another provider of the same model.

Each unit runs with the platform's container settings (read-only root, uid 65534, the card's cpus) for at most the
card's timeout, and the unit's own checker then scores it in the base image. The runner prints `passed N of 86` and
writes a row per unit to `results/<time>/results.jsonl`. A run of all 86 units on 2026-11-07, four at a time, took
2 hours 10 minutes and about $1.94 of OpenRouter credit.

A local run differs from the platform's in four ways:

- **Serving.** The platform served the model itself, at fp8. OpenRouter routes to its own providers, so the replies
  differ even at temperature 0 and a fixed seed.
- **Scoring.** The runner uses each unit's checker. The platform's verifier also checks the output folder's limits
  and scans for canaries first.
- **Units.** The Final's units are private. The kit's 86 public units are the closest public set.
- **Network.** The platform's containers reach only the model's proxy. The runner uses Docker's default network, so a
  container can reach the internet. The agent's code calls nothing but the model.

## Contents

| Path | What it is |
|---|---|
| `docs/architecture.md` | The module map, the reading order and the glossary |
| `agent/` | The agent: `n11/` its steps, `common/` reading the unit, running scripts and guarding the output, `qf/` the library, `model.py` the client |
| `Dockerfile`, `docker/` | The image, on the organisers' sandbox base pinned to the Final image's packages |
| `scripts/fetch_vendor.sh` | Fetches the kit, the tokenizer and the `tokenizers` wheel, and builds the base |
| `scripts/verify_image.sh` | Compares a build with the Final image |
| `scripts/run_units.py` | Runs an image over the kit's units and scores each with its own checker |
| `scripts/qf_index.py` | Builds the library's index for the prompt; the image build runs it |

## Licence

The code written for this entry is under the MIT licence ([LICENSE](LICENSE)). What the build fetches keeps its own
terms, and none of it is committed:

| Material | Terms |
|---|---|
| The organisers' Track 1 kit (`Agenthon-2026/track1-coding-public` at `84049db`): units, checkers, the base's Dockerfile | CC BY-NC 4.0, Pawel Polak, except where a unit's `manifest.json` names another licence |
| The organisers' `qfbench2-common` (`Agenthon-2026/Agenthon2026-public` at `v2.6.0`) | CC BY-NC 4.0, Pawel Polak |
| The tokenizer of `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (revision `2dc98e2`) | NVIDIA Nemotron Open Model License |
| `tokenizers` 0.23.3 | Apache-2.0 |
| The base's packages (`docker/base.lock`) and `python:3.13-slim` | Each package's own licence |
| The model, reached over an API with your own key | The provider's terms; no weights are used or distributed |

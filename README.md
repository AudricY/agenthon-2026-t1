# n11: a-team's Agenthon 2026 Track 1 agent

This is the code of a-team's entry in Track 1 (Coding) of Agenthon 2026, a competition in the NeurIPS 2026
Competition Track. The entry placed first of 61 teams, with a verified Final score of 0.1786. Five teams scored
0.1786, and the published tie-break, the earlier upload, ranked this entry first.

The agent, `n11`, solves a quantitative-finance coding task in a short loop. The model writes Python files, the agent
runs them, and the model reads the bounded output and the output checks before it writes again. The scripts can import a
library of 316 vetted functions written for the entry. Every request goes to the competition's model,
`nvidia/nemotron-3-super-120b-a12b`. [METHOD.md](METHOD.md) describes the agent and the research behind it.

## Results

| Units | Run | Result |
|---|---|---|
| The organisers' private Final units | their verification rerun | 0.1786 |
| The organisers' private Final units | Codabench's run of the upload | 0.1071 |
| The kit's 86 public units | the last full run before the upload, 2026-10-23 | 25 of 86 |
| The kit's 86 public units | this repository's runner, 2026-11-03 | 27 of 86 |

The two Final scores come from the same image. The organisers did not publish why their rerun scored higher. The
model's answers vary from run to run, and two identical local runs once differed on 10 of 86 units.

## Quick start

```sh
scripts/fetch_vendor.sh
docker build --platform linux/amd64 -t agenthon-2026-t1:n11 .
scripts/verify_image.sh agenthon-2026-t1:n11
```

The last command compares the build with the image submitted to the Final, which is still public at
`ghcr.io/audricy/agenthon-t4@sha256:8dba17e4…`. Running the agent on units needs a model endpoint and a key; see
[REPRODUCE.md](REPRODUCE.md).

## Contents

| Path | What it is |
|---|---|
| `agent/` | The agent, byte for byte as it sits under `/opt/agent` in the Final image |
| `Dockerfile` | Builds that image on the organisers' sandbox base |
| `docker/` | Pins the sandbox base to the Python packages of the Final image |
| `scripts/fetch_vendor.sh` | Fetches the organisers' kit, the tokenizer and the `tokenizers` wheel, and builds the base |
| `scripts/verify_image.sh` | Compares a built image with the Final image |
| `scripts/run_units.py` | Runs an image over the kit's units and scores each unit with its own checker |
| `scripts/qf_index.py` | Builds the library index that n11 puts in its prompt; the image build runs it |
| `tests/` | The library's tests: 1,808 pass in the built image, and 8 skip without QuantLib |
| [METHOD.md](METHOD.md) | How n11 works, and what was tried before it |
| [REPRODUCE.md](REPRODUCE.md) | Building, verifying and running the agent |
| [THIRD-PARTY.md](THIRD-PARTY.md) | What the build fetches instead of committing, and under which terms |

## Licence

The code written for this entry is under the MIT licence ([LICENSE](LICENSE)). The organisers' kit, the task text in
`agent/n11/examples.jsonl`, the tokenizer and the base image's packages keep their own terms
([THIRD-PARTY.md](THIRD-PARTY.md)).

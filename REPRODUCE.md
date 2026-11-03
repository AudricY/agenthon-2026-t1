# Reproducing the entry

A build from this repository matches the Final image in every agent file and every Python package. The scores
reproduce only roughly, since the model's answers vary between runs and the Final's units are private.

## 1. What you need

- Linux on x86-64, Docker with BuildKit, git, curl and Python 3.13 or later.
- About 5 GB of disk for the kit, the images and Docker's build cache.
- To run units: a key for an OpenAI-compatible endpoint that serves `nvidia/nemotron-3-super-120b-a12b`, and the
  organisers' `qfbench2-common` package, which needs Python 3.13:

  ```sh
  python3 -m venv .venv && . .venv/bin/activate
  pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.6.0#subdirectory=common"
  ```

## 2. Build

```sh
scripts/fetch_vendor.sh
docker build --platform linux/amd64 -t agenthon-2026-t1:n11 .
```

`fetch_vendor.sh` puts everything it fetches under `vendor/`, which git ignores:

1. It clones the organisers' Track 1 kit (`Agenthon-2026/track1-coding-public`) at commit `84049db`, the commit the
   Final's base was built from.
2. It builds the kit's sandbox base, then pins its Python packages to the Final image's (`docker/base.lock`). The
   kit's Dockerfile asks only for minimum versions, so an unpinned build made on 2026-11-03 resolved newer releases of
   pandas, polars, pyarrow and 20 other packages.
3. It downloads the model's tokenizer from Hugging Face at a fixed revision and the `tokenizers` 0.23.3 wheel from
   PyPI, and checks each against its sha256.

The base is tagged `finance-bench-sandbox:latest`, the name the Dockerfile expects. To keep an existing image of that
name, set `BASE_TAG=other:tag`, build with `--build-arg BASE=other:tag`, and pass `--base other:tag` to
`run_units.py`, which runs the checkers in the base.

## 3. Verify against the Final image

```sh
scripts/verify_image.sh agenthon-2026-t1:n11
```

The script pulls the Final image and compares the sha256 of every file under `/opt/agent` and of the `solve` wrapper.
It then compares `pip freeze`. A build from a clean clone on 2026-11-03 printed:

```
agent files: 125 in the Final image, 125 built
agent files: identical
python packages: identical
```

The image's Python is 3.13.18, where the Final's was 3.13.16, because the kit's base starts from `python:3.13-slim`.

## 4. Run the units

The platform hands each container three variables: `MODEL_ENDPOINT`, `MODEL_NAME` and `MODEL_TOKEN`.
`scripts/run_units.py` hands over the same three, by name, so the key never appears on a command line. These settings
use OpenRouter:

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

The `AGENTHON_MODEL_*` lines make a hosted route behave like the platform's model, and the runner forwards them too:

- `REASONING_FIELD=1`: OpenRouter drops `chat_template_kwargs`, which is how the client turns the model's thinking
  off, so without this line the model thinks on every request and the output cap cuts its replies. With it, the
  client also sends OpenRouter's own field, `reasoning: {"enabled": false}`.
- `MIN_INTERVAL_S=44`: the client leaves at least 44 s between two requests of a unit. The platform's model took 37
  to 46 s at the median, so the agent's 600 s clock allows about as many turns as it did there.
- `RATE_*`: when OpenRouter answers 429, the client waits and tries again, up to 6 attempts in all. The attempt
  does not count against the 20 requests and the wait does not count against the clock, since the platform has no
  such limit.
- `PROVIDER_PIN`: lets OpenRouter fall back to another provider of the same model.

Each unit runs in the platform's container settings (read-only root, uid 65534, the card's cpus) for at most the
card's timeout. The unit's own checker (`checks/test.sh`) then scores the output in the base image, as step 6 of the
kit's README does. The runner prints `passed N of 86` and writes a row per unit to `results/<time>/results.jsonl`.

A run of all 86 units on 2026-11-03, four at a time, solved 27. It took 2.4 hours and about $1.62 of OpenRouter
credit.

## 5. Run the library's tests

```sh
docker run --rm --network none -v "$PWD:/repo:ro" -w /repo --entrypoint python3 agenthon-2026-t1:n11 \
    -m pytest -q -p no:cacheprovider tests
```

The tests run inside the built image, which has the agent's Python and packages. On 2026-11-03, 1,808 passed. The 8
that skip are cross-checks against QuantLib, which the image does not include.

## How a local run differs from the platform

- **The model's serving.** The platform served the model itself, at fp8. OpenRouter routes to its own providers, so
  the replies differ even at temperature 0 and a fixed seed.
- **The scoring.** The runner scores with each unit's checker. The platform's verifier also checks the output
  folder's limits and scans for canaries before it runs the checks.
- **The units.** The Final's units are private. The kit's 86 public units are the closest public set, and the image
  solved 25 of them in the last full run before the Final upload.
- **The network.** The platform's containers reach only the model's proxy. The runner uses Docker's default bridge
  network, so a container can reach the internet. The agent's own code calls nothing but the model.

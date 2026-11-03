#!/bin/sh
# Fetches what the build needs but the repository does not hold, each piece checked by its commit or hash:
# the organisers' Track 1 kit, the sandbox base image built from it and pinned to the Final's packages
# (docker/base-pin.Dockerfile), the model's tokenizer and the tokenizers wheel.
#   scripts/fetch_vendor.sh            # tags the base finance-bench-sandbox:latest, the name the Dockerfile expects
#   BASE_TAG=my-base:1 scripts/fetch_vendor.sh   # then: docker build --build-arg BASE=my-base:1 ...
set -eu
cd "$(dirname "$0")/.."

KIT_URL=https://github.com/Agenthon-2026/track1-coding-public.git
KIT_COMMIT=84049dbc066cc02cf39c2cc18105c2f92284978d
BASE_TAG=${BASE_TAG:-finance-bench-sandbox:latest}
KIT_TAG=${KIT_TAG:-finance-bench-sandbox:kit-84049db}
TOKENIZER_URL=https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/resolve/2dc98e2afe4face0e4ce40972a915c45368bd34a/tokenizer.json
TOKENIZER_SHA256=623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7
WHEEL=tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
WHEEL_SHA256=376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87

KIT=vendor/track1-coding-public
if [ ! -d "$KIT/.git" ]; then
    git clone --quiet "$KIT_URL" "$KIT"
fi
git -C "$KIT" checkout --quiet "$KIT_COMMIT"
docker build --platform linux/amd64 -t "$KIT_TAG" -f "$KIT/docker/sandbox.Dockerfile" "$KIT"
docker build --platform linux/amd64 --build-arg KIT_BASE="$KIT_TAG" -t "$BASE_TAG" -f docker/base-pin.Dockerfile .

mkdir -p vendor/nemotron3 vendor/tokenizers
curl -fsSL -o vendor/nemotron3/tokenizer.json "$TOKENIZER_URL"
echo "$TOKENIZER_SHA256  vendor/nemotron3/tokenizer.json" | sha256sum -c -

WHEEL_URL=$(python3 -c '
import json, sys, urllib.request
release = json.load(urllib.request.urlopen("https://pypi.org/pypi/tokenizers/0.23.3/json"))
print(next(f["url"] for f in release["urls"] if f["filename"] == sys.argv[1]))' "$WHEEL")
curl -fsSL -o "vendor/tokenizers/$WHEEL" "$WHEEL_URL"
echo "$WHEEL_SHA256  vendor/tokenizers/$WHEEL" | sha256sum -c -

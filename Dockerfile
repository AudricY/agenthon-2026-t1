# The image submitted to the Agenthon 2026 Track 1 Final: agent n11 on the organisers' sandbox base.
# scripts/fetch_vendor.sh first builds the base and fetches the tokenizer and the tokenizers wheel, then:
#   docker build --platform linux/amd64 -t agenthon-2026-t1:n11 .
# The image runs as `solve --task-dir /input --out /app/output` (SUBMISSION_CLI.md in the organisers' kit).
ARG BASE=finance-bench-sandbox:latest

FROM ${BASE} AS base
LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="coding"
LABEL qfbench2.verb="solve"
# HOME=/tmp: the platform runs the image as uid 65534, with no home directory and a read-only root.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONPATH=/opt/agent
COPY agent/requirements.txt /opt/agent/requirements.txt
RUN if grep -qv '^\s*\(#\|$\)' /opt/agent/requirements.txt; then pip install --no-cache-dir -r /opt/agent/requirements.txt; fi
COPY agent/ARTIFACT_PROVENANCE.md agent/ROSTER_CAP /opt/agent/

# The tokenizers wheel, mounted from a stage of its own so that no layer keeps it.
FROM scratch AS n11-wheel
COPY vendor/tokenizers/ /

FROM base AS n11-tokens
COPY agent/n11/tokenizers.txt /tmp/tokenizers.txt
RUN --mount=type=bind,from=n11-wheel,target=/tmp/wheels pip install --no-cache-dir --no-index --find-links /tmp/wheels \
        --no-deps --require-hashes -r /tmp/tokenizers.txt && rm /tmp/tokenizers.txt
COPY vendor/nemotron3/tokenizer.json /opt/agent/n11/tokenizer.json
RUN echo "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7  /opt/agent/n11/tokenizer.json" | sha256sum -c -

# The method library's index, which n11 puts in its system prompt, built where the script stays out of the image.
FROM n11-tokens AS n11-index
COPY agent/qf/ /build/agent/qf/
COPY agent/n11/ /build/agent/n11/
COPY scripts/qf_index.py /build/scripts/qf_index.py
RUN AGENTHON_N11_TOKENIZER=/opt/agent/n11/tokenizer.json python3 /build/scripts/qf_index.py --qf /build/agent/qf \
        --out /build/index.txt

# n0 and w1 come along because n11 imports them: n0's one-shot request and w1's first draft.
FROM n11-tokens AS agent-n11
COPY agent/common/ /opt/agent/common/
COPY agent/n11/ /opt/agent/n11/
COPY agent/n0/ /opt/agent/n0/
COPY agent/w1/ /opt/agent/w1/
COPY agent/qf/ /opt/agent/qf/
COPY --from=n11-index /build/index.txt /opt/agent/n11/index.txt
COPY agent/model.py /opt/agent/model.py
ENV AGENTHON_MODULE=n11.solve
LABEL agenthon.agent=n11
ENV AGENTHON_AGENT=n11
RUN printf '#!/bin/sh\nexec python3 -m %s solve "$@"\n' "$AGENTHON_MODULE" > /usr/local/bin/solve && \
    chmod +x /usr/local/bin/solve && \
    python3 -c "import importlib, os; importlib.import_module(os.environ['AGENTHON_MODULE'])" && \
    python3 -c "import model; from common import house; assert house.model is model" && \
    python3 -c "from n11 import measure; assert measure.UNIT.name == 'tokens', measure.UNIT.said()" && \
    useradd --create-home --uid 1000 runner
WORKDIR /tmp
USER runner
CMD ["solve", "--help"]

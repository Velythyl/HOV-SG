# HOV-SG (original release) + RAGMAP adapter, as an "object_mapping" container.
#
# CUDA 12.1 runtime matches the CUDA bundled with the PyTorch 2.3.1 wheels.
# Python 3.9 and the pinned packages follow upstream environment.yaml.
# habitat-sim is deliberately NOT installed: it is only needed to re-render
# HM3DSem walks, not to build a graph from posed RGB-D frames.
# Model weights are NOT baked in; see README "RAGMAP adapter" (mount /weights).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG UV_VERSION=0.9.18
ARG HOVSG_GIT_SHA=unknown
ENV DEBIAN_FRONTEND=noninteractive \
    HOVSG_ROOT=/opt/HOV-SG \
    VIRTUAL_ENV=/opt/HOV-SG/.venv \
    HOVSG_WEIGHTS=/weights \
    HOVSG_GIT_SHA=${HOVSG_GIT_SHA} \
    HF_HOME=/weights/hf \
    TORCH_HOME=/weights/torch \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/HOV-SG/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        libgl1 libegl1 libgomp1 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libsm6 libxt6 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --silent --show-error https://astral.sh/uv/${UV_VERSION}/install.sh | sh

WORKDIR ${HOVSG_ROOT}

COPY docker/requirements.txt /tmp/hovsg-requirements.txt
# scikit-fmm (navigation graph) ships no cp39 wheels, so a compiler is needed
# for this layer only.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && uv python install 3.9 \
    && uv venv --seed --python 3.9 ${VIRTUAL_ENV} \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch==2.3.1 torchvision==0.18.1 \
    && printf 'torch==2.3.1\ntorchvision==0.18.1\n' > /tmp/torch-constraints.txt \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python \
        --constraints /tmp/torch-constraints.txt -r /tmp/hovsg-requirements.txt \
    && uv cache clean \
    && apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

COPY . ${HOVSG_ROOT}
RUN uv pip install --python ${VIRTUAL_ENV}/bin/python --no-deps -e . \
    && python -c "import torch, open_clip, open3d, segment_anything, faiss, pyvista, skfmm, hydra; \
from hovsg.graph.graph import Graph; from ragmap_adapter.run import main; \
assert torch.version.cuda == '12.1', torch.version.cuda; print('imports ok')" \
    && ragmap-run --help >/dev/null

VOLUME ["/weights"]
WORKDIR /work
ENTRYPOINT ["ragmap-run"]
CMD ["--help"]

#!/usr/bin/env bash
# Smoke test of the published image on a synthetic scene.
#
#   smoke/run_smoke.sh [IMAGE] [RUNTIME]
#
# IMAGE defaults to ghcr.io/velythyl/hovsg:latest, RUNTIME to podman (docker in
# CI). Weights are cached in $HOVSG_WEIGHTS_CACHE (default ~/.cache/hovsg-weights).
# With a GPU (GPU_ARGS="--device nvidia.com/gpu=all" for podman,
# GPU_ARGS="--gpus all" for docker) it runs the production config; without one
# it runs the CPU smoke profile (SAM ViT-B, fewer SAM points, every 3rd frame),
# which exercises the same code path in ~10-20 minutes on a 4-core runner.
set -euo pipefail

IMAGE=${1:-ghcr.io/velythyl/hovsg:latest}
RUNTIME=${2:-podman}
WORK=${SMOKE_WORK:-$(mktemp -d)}
WEIGHTS=${HOVSG_WEIGHTS_CACHE:-$HOME/.cache/hovsg-weights}
UP_AXIS=${UP_AXIS:-z}
GPU_ARGS=${GPU_ARGS:-}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$WORK/scene" "$WORK/output" "$WEIGHTS"
# Generate the scene inside the image so the host needs no Python deps.
"$RUNTIME" run --rm -v "$HERE:/smoke:ro" -v "$WORK/scene:/scene" --entrypoint python "$IMAGE" \
  /smoke/make_synthetic_scene.py --out /scene --up-axis "$UP_AXIS" --frames "${FRAMES:-24}"

if [[ -z "$GPU_ARGS" ]]; then
  PROFILE=(models.sam.type=vit_b models.sam.points_per_side=6 models.sam.points_per_batch=36 pipeline.skip_frames=3)
else
  PROFILE=(pipeline.skip_frames=2)
fi

# shellcheck disable=SC2086
"$RUNTIME" run --rm $GPU_ARGS \
  -v "$WORK/scene:/input:ro" -v "$WORK/output:/output" -v "$WEIGHTS:/weights" \
  "$IMAGE" --input /input --output /output "${PROFILE[@]}" ${EXTRA_OVERRIDES:-}

"$RUNTIME" run --rm -v "$HERE:/smoke:ro" -v "$WORK/scene:/scene:ro" -v "$WORK/output:/output:ro" \
  --entrypoint python "$IMAGE" /smoke/check_output.py --output /output --scene /scene
echo "smoke outputs in $WORK"

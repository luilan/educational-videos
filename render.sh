#!/usr/bin/env bash
# Render one video.
#   ./render.sh <series> <video> [quality]
#   ./render.sh how-llms-work v05          # final 1080p60
#   ./render.sh foundations f07 -ql        # fast 480p draft
# Output: <series>/media/videos/<video>_scene/<quality>/<Class>.mp4
set -euo pipefail
series=${1:?series folder, e.g. how-llms-work}; video=${2:?video id, e.g. v05}; quality=${3:--qh}
cd "$(dirname "$0")/$series"
PYTHONPATH="../framework${PYTHONPATH:+:$PYTHONPATH}" manim "$quality" --disable_caching -a "${video}_scene.py"

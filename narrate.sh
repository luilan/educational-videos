#!/usr/bin/env bash
# Generate the narration for one video from its <video>_script.py:
#   ./narrate.sh how-llms-work v05
# Writes <series>/voice/<video>/s<N>.wav + sections.json (durations and word timestamps used as animation cues).
set -euo pipefail
series=${1:?series folder}; video=${2:?video id}
cd "$(dirname "$0")/$series"
python ../framework/voice.py "$video"

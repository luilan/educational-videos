#!/usr/bin/env bash
# Download the Kokoro text-to-speech model (~340 MB) into <repo>/tts/.
set -euo pipefail
dir="$(dirname "$0")/../tts"; mkdir -p "$dir"
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -f "$dir/$f" ] || curl -L -o "$dir/$f" "$base/$f"
done

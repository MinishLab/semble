#!/usr/bin/env bash
# Records the README demo GIFs (assets/images/tour-{dark,light}.gif) from dark.tape and light.tape.
set -euo pipefail

cd "$(dirname "$0")/../.."
command -v vhs >/dev/null || { echo "vhs not found, install it with: brew install vhs" >&2; exit 1; }
[ -x .venv/bin/semble ] || { echo ".venv/bin/semble not found, run: make install" >&2; exit 1; }
export SEMBLE_BIN_DIR="$PWD/.venv/bin"

seed=$(mktemp -d)
dark=$(mktemp -d)
light=$(mktemp -d)
trap 'rm -rf "$seed" "$dark" "$light"' EXIT

# Create an isolated $HOME with agent config dirs so `semble install` detects some agents.
mkdir -p "$seed"/.claude "$seed"/.cursor "$seed"/.codex
git -c advice.detachedHead=false clone -q --depth 1 --branch v2.13.5 https://github.com/pydantic/pydantic "$seed/pydantic"
# Warm the model cache so the recording doesn't hit the network or print an HF Hub warning.
env -i HOME="$seed" PATH="$SEMBLE_BIN_DIR:/usr/bin:/bin" "$SEMBLE_BIN_DIR/semble" search x "$seed/pydantic" -k 1 >/dev/null

# Record the demo for both dark and light themes. Each home gets its own agent dirs and repo copy, but shares the
# seed's .cache: grammar dylibs load slowly the first time from a new path, which stalls the start of indexing.
for home in "$dark" "$light"; do
  mkdir -p "$home"/.claude "$home"/.cursor "$home"/.codex
  cp -R "$seed/pydantic" "$home/pydantic"
  ln -s "$seed/.cache" "$home/.cache"
done
DEMO_HOME="$dark" vhs scripts/demo/dark.tape >/dev/null &
dark_pid=$!
DEMO_HOME="$light" vhs scripts/demo/light.tape >/dev/null &
light_pid=$!
wait "$dark_pid"
wait "$light_pid"
echo "Recorded assets/images/tour-dark.gif and assets/images/tour-light.gif"

# Generates the dark-theme "See it in action" GIF in the README.
#
# Prerequisites:
#   brew install vhs                          # also installs ttyd + ffmpeg
#   uv sync                                   # builds .venv/bin/semble
#
# Setup — an isolated demo $HOME so `semble install` never touches your
# real dotfiles, with a few agent config dirs pre-created so it shows
# some "(detected)" agents, plus a repo to search:
#
#   export DEMO_HOME=$(mktemp -d)
#   mkdir -p "$DEMO_HOME"/.claude "$DEMO_HOME"/.cursor "$DEMO_HOME"/.codex
#   git clone --depth 1 --branch 0.115.6 https://github.com/fastapi/fastapi "$DEMO_HOME"/fastapi
#   export SEMBLE_BIN_DIR="$(pwd)/.venv/bin"
#
# Warm the embedding model cache first, so the recording doesn't hit the
# network or show a "sending unauthenticated requests to HF Hub" warning:
#   env -i HOME="$DEMO_HOME" PATH="$SEMBLE_BIN_DIR:/usr/bin:/bin" \
#     "$SEMBLE_BIN_DIR/semble" search x "$DEMO_HOME/fastapi" -k 1 >/dev/null
#   rm -rf "$DEMO_HOME/Library/Caches/semble"   # drop the index cache, keep the model cache
#
# Render (from the repo root):
#   vhs .github/scripts/demo-dark.tape
#   vhs .github/scripts/demo-light.tape
#
# Both tapes exec into `env -i HOME="$DEMO_HOME" ... bash --norc --noprofile`
# as their first step, replacing the shell entirely so nothing here can read
# or write your real environment.

Output "assets/images/tour-dark.gif"

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 900
Set Theme "Catppuccin Mocha"
Set Padding 20
Set Margin 30
Set MarginFill "#11111b"
Set BorderRadius 12
Set WindowBar Colorful
Set WindowBarSize 36
Set TypingSpeed 45ms

Hide
Type "exec env -i HOME=$DEMO_HOME PATH=$SEMBLE_BIN_DIR:/usr/bin:/bin TERM=xterm-256color /bin/bash --norc --noprofile"
Enter
Sleep 800ms
Type "cd $HOME; PS1='$ '; clear"
Enter
Sleep 300ms
Show

Type "semble install"
Enter
Sleep 1200ms
Enter
Sleep 500ms
Enter
Sleep 500ms
Enter
Sleep 2200ms

Hide
Type "clear"
Enter
Sleep 300ms
Show

Type "cd fastapi"
Enter
Sleep 600ms

Type@25ms `semble search "OpenAPI schema generation" -k 1 --pretty`
Enter
Sleep 4500ms

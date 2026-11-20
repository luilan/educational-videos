#!/usr/bin/env bash
# The same five actions under Codex CLI's three built-in permission profiles (Linux: bubblewrap + seccomp).
#   ./sandbox_matrix.sh /path/to/codex
# Needs: Linux with unprivileged user namespaces (bubblewrap works), python3. Writes only to scratch folders it creates.
set -u
codex=${1:-codex}
work=$(mktemp -d -p "$HOME" sbx-project.XXXX)       # the "project folder"
outside=$(mktemp -d -p "$HOME" sbx-outside.XXXX)    # a folder outside it (not under /tmp)
export CODEX_HOME=$(mktemp -d -p "$HOME" sbx-home.XXXX)
for profile in :read-only :workspace :danger-full-access; do
  echo "## $profile"
  "$codex" sandbox -P "$profile" -C "$work" -- bash -c "
    { echo hi > inside.txt; } 2>/dev/null;        echo \"write in the project folder:  exit \$?\"
    { echo hi > $outside/x.txt; } 2>/dev/null;    echo \"write outside the project:    exit \$?\"
    { echo hi > /tmp/sbx-test.txt; } 2>/dev/null; echo \"write to /tmp:                exit \$?\"
    head -c1 /etc/hostname > /dev/null;           echo \"read a system file:           exit \$?\"
    python3 -c 'import urllib.request; urllib.request.urlopen(\"http://example.com\", timeout=5)' 2>/dev/null
                                                  echo \"open a network connection:    exit \$?\"" 2>&1 | grep -v "PATH aliases"
  rm -f "$work/inside.txt" "$outside/x.txt" /tmp/sbx-test.txt
done
rm -rf "$work" "$outside" "$CODEX_HOME"

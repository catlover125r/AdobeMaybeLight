#!/usr/bin/env bash
# Assemble a local, double-clickable AdobeMaybeLight.app from the release binary.
# Local use only: it dynamically links Homebrew's libraw, so it runs on a Mac
# that has `brew install libraw`. Not notarized / not for redistribution.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/aml"
APP="${1:-$ROOT/dist/AdobeMaybeLight.app}"

[ -x "$BIN" ] || { echo "build first: cargo build --release -p app"; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/aml"
cp "$ROOT/packaging/Info.plist" "$APP/Contents/Info.plist"

# Ad-hoc sign so Gatekeeper/library-validation is happy for local launch.
codesign --force --deep --sign - "$APP"

echo "built: $APP"

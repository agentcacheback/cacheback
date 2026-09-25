#!/usr/bin/env bash
# Install the tracked Git hooks; agent hooks require each application's trust review.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/pre-push .githooks/pre-merge-commit

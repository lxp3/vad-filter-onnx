# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, etc.) working in this repo.

## Git: branch rules

- `main` is the only branch that may be pushed to `origin` (GitHub) and `gitlab`.
- `main-20261019` is a local-only archive of the pre-squash history (about 137MB, includes
  removed `public/models/*.onnx` and `clang-format`). **Never push it to any remote**, and never
  merge it into `main` or cherry-pick/rebase from it.
  - Pushing it would make every `git clone` download the old history again.
  - Do not use `git push --all`, `git push --mirror`, or `git push --tags` (they would include it).
    Push explicitly: `git push origin main` / `git push gitlab main`.
- Do not recreate the old tags (`v0.1.0` ... `v2.1.0`); they point into the archived history.
- Do not force-push `main` unless the user explicitly asks.

## Repo size

- Do not commit large binaries (`*.onnx`, executables such as `clang-format`, archives).
  Models belong in `models/` / `public/models/`, which are not tracked; share them via a release
  or a download script instead.
- Before committing, check for large files: `git diff --cached --stat` and
  `git ls-files -s | awk '{print $2}' | git cat-file --batch-check | sort -k3 -n -r | head`.

## Backup

A full-history bundle exists outside the repo at
`/data/user/lxp/vad-filter-onnx-full-20261019.bundle` (restore with `git clone <bundle> <dir>`).

#!/usr/bin/env bash
# Clone-or-pull the dev-data dataset repos, re-stage from runs/, and stop.
#
# Deliberately stops before `git push`. The whole point of these repositories is
# that they hold things that cannot be recomputed cheaply; a script that pushes
# unattended is one bad glob away from committing a truncated file over a good
# one. Review the diff, then push by hand.
#
#   scripts/sync_dev_data.sh                    # both repos
#   scripts/sync_dev_data.sh quantik-dev-data   # one
set -euo pipefail

NAMESPACE="${QUANTIK_HF_NAMESPACE:-brpoplpush}"
PYTHON="${PYTHON:-.venv/bin/python}"
if [ "$#" -gt 0 ]; then
  REPOS=("$@")
else
  REPOS=(quantik-dev-data quantik-dev-runs)
fi

# git-lfs is not optional here: without it a clone yields pointer files that
# look like success — small text files where a 273 MB array should be.
command -v git-lfs >/dev/null || { echo "git-lfs is not installed; a clone would yield pointer files" >&2; exit 1; }

# A case, not an associative array: macOS ships bash 3.2, which has none.
for repo in "${REPOS[@]}"; do
  case "$repo" in
    quantik-dev-data) dir=runs/devdata ;;
    quantik-dev-runs) dir=runs/devruns ;;
    *) echo "unknown repo: $repo" >&2; exit 2 ;;
  esac

  if [ -d "$dir/.git" ]; then
    echo "== pulling $repo into $dir"
    git -C "$dir" pull --ff-only
  else
    echo "== cloning $repo into $dir"
    [ -e "$dir" ] && { echo "$dir exists but is not a clone; move it aside first" >&2; exit 1; }
    mkdir -p "$(dirname "$dir")"
    git clone "git@hf.co:datasets/$NAMESPACE/$repo" "$dir"
  fi

  echo "== staging $repo"
  # --prune so a renamed source cannot leave a stale copy behind. A stale file
  # in a backup is worse than a missing one: it still hashes fine.
  "$PYTHON" -m quantik_models.export.devdata "$dir" --repo "$repo" --prune

  echo "== $repo diff"
  git -C "$dir" add -A
  git -C "$dir" status --short | head -40
  echo
done

cat <<'MSG'
Nothing was pushed. Review the diffs above, then for each repo:

  cd runs/devdata   # or runs/devruns
  git commit -m "Re-stage from runs/"
  git push
MSG

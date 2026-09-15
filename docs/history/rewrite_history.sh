#!/usr/bin/env bash
# Einmaliger Umbau der Git-History -- ganz am Ende, nach dem letzten Commit:
#   1. Notebook-Ausgaben aus JEDER Version entfernen (die Versionen bleiben)
#   2. Autoren-Identitaeten ueber .mailmap vereinen
#   3. Co-Authored-By-/Session-Zeilen und die eine Erwaehnung im Text entfernen
# Getestet am 2026-09-14 auf einem Spiegel-Klon: Code identisch, 173 Commits,
# .git von 104 auf 15 MB. Danach ist ein Force-Push noetig und jeder Klon
# (auch der Linux-Rechner) muss neu geholt werden.
#
#   bash docs/history/rewrite_history.sh          # rechnet lokal, pusht NICHT
#   git push --force --all origin                 # erst nach Sichtpruefung
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Nur getrackte Aenderungen zaehlen -- docs/history selbst muss nicht ins Git.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Getrackte Dateien geaendert, aber nicht committet -- erst committen." >&2
  exit 1
fi

SICHERUNG="$ROOT/../pytorch-vor-rewrite-$(date +%Y%m%d-%H%M).git"
git clone --quiet --mirror "$ROOT" "$SICHERUNG"
echo "Sicherung: $SICHERUNG"

REMOTE="$(git remote get-url origin)"
python -m git_filter_repo --force \
  --mailmap .mailmap \
  --message-callback docs/history/clean_messages.py \
  --file-info-callback docs/history/strip_notebook_outputs.py
git remote add origin "$REMOTE"
git reflog expire --expire=now --all
git gc --prune=now --aggressive --quiet

echo
echo "Fertig. Pruefen:"
echo "  git log --oneline | head; git shortlog -sn HEAD; du -sh .git"
echo "  git log --all --format='%B%an' | grep -ic claude     # muss 0 sein"
echo "Dann: git push --force --all origin  (und die Klone neu holen)"

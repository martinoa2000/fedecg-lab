#!/usr/bin/env bash
# Publish the dashboard to GitHub Pages (the gh-pages branch).
#
# Exports the dashboard data from results/ and PTB-XL, builds the app for the
# repository's Pages path (/<repo>/), and pushes the built site as the only
# commit of the gh-pages branch, replacing whatever was published before.
# The site includes a few dozen PTB-XL recordings, which the dataset's CC BY
# 4.0 license allows with the attribution shown in the dashboard's footer.
#
# The Train page needs the local training server, so on the published site it
# explains how to run training locally instead.
#
# Usage (after downloading PTB-XL and running the experiments):
#   scripts/publish_dashboard.sh
# Then, once: GitHub > Settings > Pages > Source: branch gh-pages, folder /.

set -euo pipefail
cd "$(dirname "$0")/.."

remote=$(git remote get-url origin)
repo=$(basename "$remote" .git)

uv run python scripts/export_dashboard.py
npm --prefix app ci --silent
npm --prefix app run build -- --base "/$repo/"

site=$(mktemp -d)
trap 'rm -rf "$site"' EXIT
cp -R app/dist/. "$site"
touch "$site/.nojekyll"  # serve files as they are, no Jekyll processing

git -C "$site" init -q -b gh-pages
git -C "$site" add -A
git -C "$site" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -m "Publish dashboard from $(git rev-parse --short HEAD)"
git -C "$site" push -q --force "$remote" gh-pages

owner=$(basename "$(dirname "${remote/://}")")
echo "Published. Site: https://$owner.github.io/$repo/" >&2

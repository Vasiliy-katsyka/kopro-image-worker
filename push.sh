#!/bin/bash
# Commits local changes to github.com/Vasiliy-katsyka/kopro-image-worker and starts the GitHub build.
#   bash push.sh 2.2.0          -> ghcr.io/vasiliy-katsyka/kopro-image:2.2.0 (Qwen-Image: generate + edit)
#   bash push.sh 1.0.0 video    -> ghcr.io/vasiliy-katsyka/kopro-video:1.0.0 (LTX-2.3 video + audio)
set -e
cd "$(dirname "$0")"
T=$(cat /root/.gh_token); TAG=${1:-2.1.1}; WORKER=${2:-image}
git add Dockerfile handler.py push.sh .github video
git -c user.name="Vasiliy Katsyka" -c user.email="kviappsgames@gmail.com" commit -qm "worker update ($WORKER $TAG)" || true
git push -q "https://x-access-token:${T}@github.com/Vasiliy-katsyka/kopro-image-worker.git" main && echo "pushed"
sleep 8
curl -s -o /dev/null -w "start build: %{http_code} (204 = ok)\n" -H "Authorization: token $T" \
  https://api.github.com/repos/Vasiliy-katsyka/kopro-image-worker/actions/workflows/build.yml/dispatches \
  -d "{\"ref\":\"main\",\"inputs\":{\"tag\":\"$TAG\",\"worker\":\"$WORKER\"}}"

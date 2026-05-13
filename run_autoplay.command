#!/bin/bash
# Auto-launch: run autoplay-random for 300s with video recording.
# Invoked by double-clicking this file in Finder.

cd /Users/shukaihu/Code_Project_Local/Big2VisionAgent || exit 1

./.venv/bin/big2-agent autoplay-agent --timeout-seconds 300 --record-video
status=$?

echo ""
echo "=== Script finished. Exit code: $status ==="
echo "Terminal will stay open. You can close this window."

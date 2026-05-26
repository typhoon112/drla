#!/usr/bin/env bash
set -u
BASE=/data1/luyifei/drla/models/Cola-DLM
URL=https://huggingface.co/ByteDance-Seed/Cola-DLM/resolve/main
LOG=/data1/luyifei/drla/outputs/cola_download_curl.log
fetch() {
  rel="$1"
  mkdir -p "$BASE/$(dirname "$rel")"
  echo "[$(date -Is)] fetching $rel" >> "$LOG"
  curl -L --fail --continue-at - --retry 999 --retry-all-errors --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 -o "$BASE/$rel" "$URL/$rel" >> "$LOG" 2>&1
  echo "[$(date -Is)] done $rel" >> "$LOG"
}
fetch cola_dlm/cola_dit/model-00001-of-00002.safetensors
fetch cola_dlm/cola_dit/model-00002-of-00002.safetensors
fetch cola_dlm/cola_dit/model.safetensors.index.json
fetch cola_dlm/cola_vae/config.json
fetch cola_dlm/cola_vae/model.safetensors
fetch tokenizer.json

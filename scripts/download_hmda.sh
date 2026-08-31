#!/usr/bin/env bash
# Full HMDA public LAR (all jurisdictions x 2019-2023) from the CFPB data-browser API.
# Stores gzipped (~6x smaller); pandas reads .csv.gz natively. Resumable.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/dataset/hmda_raw"
LOG="${REPO_ROOT}/dataset/hmda_download.log"
mkdir -p "$OUT"

STATES="AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA PR RI SC SD TN TX UT VT VA WA WV WI WY"
YEARS="2019 2020 2021 2022 2023"
MIN_BYTES=2000
PARALLEL=4

fetch() {
  local yr=$1 st=$2
  local gz="$OUT/hmda_${yr}_${st}.csv.gz"
  local plain="$OUT/hmda_${yr}_${st}.csv"
  local tmp="$OUT/.tmp_${yr}_${st}"

  # already have gzipped copy
  [[ -f "$gz" && $(stat -c%s "$gz") -gt 500 ]] && { echo "SKIP  $yr $st"; return 0; }
  # have plain copy from earlier run -> just compress it
  if [[ -f "$plain" && $(stat -c%s "$plain") -gt $MIN_BYTES ]]; then
    gzip -f "$plain" && echo "ZIP   $yr $st $(du -h "$gz" | cut -f1)"; return 0
  fi

  local avail; avail=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
  [[ ${avail:-0} -lt 4 ]] && { echo "ABORT low disk (${avail}G)"; return 9; }

  for a in 1 2 3; do
    if curl -sL --max-time 1200 --retry 2 --retry-delay 5 \
         -o "$tmp" "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=${yr}&states=${st}"; then
      if [[ $(stat -c%s "$tmp" 2>/dev/null || echo 0) -gt $MIN_BYTES ]] && head -1 "$tmp" | grep -q activity_year; then
        gzip -c "$tmp" > "$gz" && rm -f "$tmp"
        echo "OK    $yr $st $(du -h "$gz" | cut -f1)"; return 0
      fi
    fi
    sleep $((a*10))
  done
  rm -f "$tmp"; echo "FAIL  $yr $st"; return 1
}
export -f fetch; export OUT MIN_BYTES

{ echo "=== HMDA download (gzip) started $(date) ==="
  for yr in $YEARS; do for st in $STATES; do echo "$yr $st"; done; done \
    | xargs -P $PARALLEL -n 2 bash -c 'fetch "$0" "$1"'
  echo "=== finished $(date) ==="
  echo "files: $(ls "$OUT"/hmda_*.csv.gz 2>/dev/null | wc -l)  size: $(du -sh "$OUT" | cut -f1)"
} >> "$LOG" 2>&1

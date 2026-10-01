#!/bin/bash
# arXiv corpus downloader for PaperWhisperer eval (T0.1)
#
# Usage:
#   bash download_arxiv.sh              # fetch anything missing (idempotent)
#   bash download_arxiv.sh --verify     # check the corpus against manifest.yaml
#   bash download_arxiv.sh --force      # re-fetch even if the file exists
#
# The script is idempotent: it never overwrites an existing non-empty file
# unless --force is given, and the download list below is kept in exact sync
# with eval/manifest.yaml. Re-running the documented reproduction sequence on
# an intact corpus is therefore a no-op, not a mutation.
#
# Environment:
#   ARXIV_PROXY      proxy URL passed to curl (e.g. http://127.0.0.1:10808).
#                    Unset by default -- a plain `curl` with no proxy is the
#                    portable default, so this works on a fresh clone.
#   EVAL_EXTRA_PATH  colon-separated directories to prepend to PATH. Only
#                    needed to pick up a Windows-side curl/sed under Git Bash;
#                    it is opt-in rather than hardcoded to one laptop.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$SCRIPT_DIR/corpus"

if [ -n "${EVAL_EXTRA_PATH:-}" ]; then
  PATH="$EVAL_EXTRA_PATH:$PATH"
fi
PROXY="${ARXIV_PROXY:-}"

# format: arxiv_id|category|filename|title_keyword
# `title_keyword` is a human-readable hint for what the PDF should be. It is
# checked by --verify (against the `preview` field in manifest.yaml, which is
# extracted first-page text), not by the downloader itself -- deciding whether a
# stream of compressed PDF bytes is the right paper is not a job for `grep`.
LIST="
1706.03762|digital_single_col|attention_is_all_you_need.pdf|Attention
2010.11929|digital_single_col|vit_an_image_is_worth_16x16.pdf|Image
1301.3781|digital_single_col|word2vec.pdf|word
1406.2661|digital_single_col|gan_generative_adversarial.pdf|Generative
1503.02531|digital_single_col|hinton_distillation.pdf|Distilling
1512.03385|digital_two_col|resnet.pdf|Deep
1505.04597|digital_two_col|unet.pdf|U-Net
1703.06870|digital_two_col|mask_rcnn.pdf|Mask
1506.02640|digital_two_col|yolov1.pdf|You
1506.01497|digital_two_col|faster_rcnn.pdf|Faster
1804.02767|digital_two_col|yolov3.pdf|YOLOv3
1612.03144|digital_two_col|fpn_feature_pyramid.pdf|Feature
1704.04861|digital_two_col|mobilenets.pdf|MobileNets
1409.4842|digital_two_col|googlenet_inception.pdf|Going
1502.03167|digital_two_col|batchnorm.pdf|Batch
1602.03837|digital_complex_layout|ligo_gw150914_prl.pdf|Observation
1910.11333|digital_complex_layout|quantum_supremacy_nature_style.pdf|Quantum
hep-th/9711201|formula_dense|maldacena_toroidal_compactification.pdf|Toroidal
hep-th/9711200|formula_dense|maldacena_large_n.pdf|Large
hep-th/9802109|formula_dense|witten_ads_holography.pdf|Gauge
math/0404188|formula_dense|green_tao_primes.pdf|Primes
hep-th/9905111|formula_dense|dbrane_theory_review.pdf|D-Branes
1804.07461|table_dense|glue_benchmark.pdf|GLUE
2211.09110|table_dense|helm_holistic_eval.pdf|Holistic
2302.13971|table_dense|llama_report.pdf|LLaMA
1905.11946|table_dense|efficientnet.pdf|EfficientNet
2303.08774|table_dense|gpt4_technical_report.pdf|GPT-4
2005.14165|long_paper|gpt3_language_models.pdf|Language
1807.06209|long_paper|planck2018_cosmo_params.pdf|Planck
2010.14533|long_paper|ligo_tests_of_gr.pdf|Tests
"

MODE="fetch"
case "${1:-}" in
  --verify) MODE="verify" ;;
  --force)  MODE="force" ;;
  "")       MODE="fetch" ;;
  *) echo "unknown argument: $1 (expected --verify or --force)" >&2; exit 2 ;;
esac

# Cross-check the download list against the manifest: every arXiv-sourced
# manifest entry must have exactly one list row pointing at the same id, and
# every list row must exist in the manifest. This is what keeps "re-run the
# documented sequence" from silently mutating the corpus, and it is also the
# cheapest way to confirm a corpus is intact without downloading anything.
if [ "$MODE" = "verify" ]; then
  # Exported rather than piped: the heredoc below *is* the python script's
  # stdin, so the list cannot also arrive that way.
  export LIST
  python3 - "$SCRIPT_DIR" <<'PY'
import hashlib
import os
import sys

import yaml

base = sys.argv[1]
manifest_path = os.path.join(base, "manifest.yaml")
if not os.path.isfile(manifest_path):
    print(f"[SKIP] {manifest_path} not present; run build_manifest.py first")
    raise SystemExit(0)

with open(manifest_path, encoding="utf-8") as fh:
    manifest = yaml.safe_load(fh)

rows = [ln.split("|") for ln in os.environ["LIST"].splitlines() if "|" in ln]

# PDF text layers routinely render Latin ligatures as single codepoints, so
# "EfficientNet" comes back as "Ef<0xFB01>cientNet" and a naive substring test
# fails on a correct document. Fold them back to ASCII before matching.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
             "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}


def normalise(text):
    text = str(text or "")
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)
    return text.lower()

entries = {}
for spec in manifest["categories"].values():
    for e in spec["entries"]:
        entries[e["file"]] = e

problems = []
hints = []
for arxiv_id, cat, name, keyword in rows:
    rel = f"{cat}/{name}"
    entry = entries.get(rel)
    if entry is None:
        problems.append(f"LIST-ONLY: {rel} is downloaded here but absent from manifest.yaml")
        continue
    if entry.get("source") != f"arxiv:{arxiv_id}":
        problems.append(
            f"ID MISMATCH: {rel} manifest says {entry.get('source')}, list says arxiv:{arxiv_id}"
        )
    path = os.path.join(base, "corpus", rel)
    if not os.path.isfile(path):
        problems.append(f"MISSING ON DISK: {rel}")
        continue
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != entry.get("sha256"):
        problems.append(f"SHA MISMATCH: {rel} on-disk content differs from manifest")
    elif keyword and keyword.lower() not in normalise(entry.get("preview")):
        # Advisory, not a failure. `preview` is the first ~90 characters of page
        # one, and on several papers that window is the arXiv stamp or a
        # publisher copyright notice rather than the title -- so a miss here is
        # usually an artefact of truncation, not evidence of a wrong download.
        # Identity is settled by the sha256 above; this only nudges a human to
        # eyeball the handful of files where the title could not be confirmed.
        hints.append(f"title not in truncated preview, check by eye: {rel} ({keyword!r})")

listed = {f"{cat}/{name}" for _, cat, name, _ in rows}
for rel, entry in sorted(entries.items()):
    if str(entry.get("source", "")).startswith("arxiv:") and rel not in listed:
        problems.append(f"MANIFEST-ONLY: {rel} ({entry['source']}) has no download row")

for h in hints:
    print(f"  [hint] {h}")
if problems:
    print("=== corpus verification FAILED ===")
    for p in problems:
        print(f"  {p}")
    raise SystemExit(1)
print(f"=== corpus verified: {len(rows)} arXiv rows match manifest ===")
print(f"corpus_version: {manifest.get('corpus_version', '<missing>')}")
PY
  exit $?
fi

echo "corpus dir: $BASE"
if [ -n "$PROXY" ]; then
  echo "proxy: $PROXY"
else
  echo "proxy: (none, direct)"
fi

CURL_PROXY_ARGS=()
[ -n "$PROXY" ] && CURL_PROXY_ARGS=(--proxy "$PROXY")

echo "$LIST" | while IFS='|' read -r id cat name kw; do
  [ -z "$id" ] && continue
  dest="$BASE/$cat/$name"
  mkdir -p "$BASE/$cat"
  if [ -s "$dest" ] && [ "$MODE" != "force" ]; then
    echo "[SKIP] $name already exists"
    continue
  fi
  ok=0
  for attempt in 1 2 3; do
    curl -sSL ${CURL_PROXY_ARGS[@]+"${CURL_PROXY_ARGS[@]}"} --max-time 180 --retry 1 \
      -H "User-Agent: PaperWhisperer-eval-corpus/0.1 (research; contact: eval@localhost)" \
      -o "$dest" "https://arxiv.org/pdf/$id" 2>/dev/null
    if [ -s "$dest" ] && head -c 5 "$dest" | grep -q "%PDF"; then
      sz=$(stat -c%s "$dest" 2>/dev/null || wc -c < "$dest")
      echo "[OK] $name ($(( sz / 1024 )) KB) attempt=$attempt"
      ok=1
      break
    else
      rm -f "$dest"
      echo "[RETRY] $name attempt=$attempt failed"
      sleep 3
    fi
  done
  [ "$ok" = "0" ] && echo "[FAIL] $name ($id)"
  sleep 3
done
echo "=== download pass complete ==="

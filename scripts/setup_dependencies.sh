#!/usr/bin/env bash
# Clones the external codebases VISPA builds on top of, pinned to the exact
# commits this repo was developed against. None of their code is vendored
# here (see steering/README.md and the top-level README for why) — this
# script is the only place that fetches it, into a gitignored third_party/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
mkdir -p "$THIRD_PARTY_DIR"

clone_pinned() {
    local name="$1" url="$2" commit="$3"
    local dest="$THIRD_PARTY_DIR/$name"

    if [ -d "$dest/.git" ]; then
        echo "[$name] already present at $dest, skipping clone (delete it to re-fetch)"
        return
    fi

    echo "[$name] cloning $url @ $commit"
    git clone "$url" "$dest"
    git -C "$dest" checkout "$commit"
}

# Probe-calibrated steering (the paper's main instantiation) and the
# ConVA-derived averaging/CAA baseline both live here.
clone_pinned "ConVA" "https://github.com/hr-jin/ConVA.git" "9484868882cd42752d1b9f27edcb1d9a1d41eb99"

# Projection-based (PCA / RepE) steering instantiation.
clone_pinned "representation-engineering" "https://github.com/andyzoujm/representation-engineering.git" "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e"

# ModPlural benchmark data (input/*.json) that pipeline/run_*.py --input points
# to. Vanilla/MoE/ModPlural/Ethos baseline numbers are cited directly from
# their papers, not reproduced, so nothing here imports this repo's code.
clone_pinned "modular_pluralism" "https://github.com/BunsenFeng/modular_pluralism.git" "56bd05d2e6d824e93a5c0bee5ac26b56aeb299aa"

echo ""
echo "Done. third_party/ now contains:"
ls "$THIRD_PARTY_DIR"
echo ""
echo "representation-engineering is pip-installable (MIT licensed): consider"
echo "    pip install -e third_party/representation-engineering"
echo "ConVA and modular_pluralism ship no LICENSE file (all-rights-reserved by"
echo "default upstream) and are NOT pip packages — this repo only imports from"
echo "them at runtime via sys.path, never copies or redistributes their code."

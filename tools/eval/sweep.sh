#!/usr/bin/env bash
# Sim-PPL sweep: for each layout, apply fork quant math to BF16 GGUF,
# uniformly Q8_0-store it (pure), evaluate with llama-perplexity.
set -euo pipefail
PY=/home/lio/.unsloth/studio/unsloth_studio/bin/python
QUANT=/home/lio/.unsloth/studio/unsloth_studio/bin/python
LP=/tmp/lcpp-build/bin/llama-perplexity
LQ=/tmp/lcpp-build/bin/llama-quantize
G=/run/media/lio/data/g
cd /home/lio/ninfer-a5000

for LAYOUT in "$@"; do
    echo "=== [$LAYOUT] quant_gguf $(date +%T)"
    $QUANT tools/eval/quant_gguf.py --layout "$LAYOUT" --out "$G/eval_${LAYOUT}.gguf"
    echo "=== [$LAYOUT] q8pure $(date +%T)"
    $LQ --pure "$G/eval_${LAYOUT}.gguf" "$G/eval_${LAYOUT}_q8.gguf" Q8_0 | tail -1
    rm -f "$G/eval_${LAYOUT}.gguf"
    echo "=== [$LAYOUT] ppl $(date +%T)"
    $LP -m "$G/eval_${LAYOUT}_q8.gguf" -f /tmp/wiki.test.raw -ngl 30 -c 512 -b 1024 \
        --chunks 50 2>&1 | grep "Final estimate" | sed "s/^/[$LAYOUT] /"
    echo "=== [$LAYOUT] done $(date +%T)"
done
echo SWEEP-DONE

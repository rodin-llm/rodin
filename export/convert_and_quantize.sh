#!/usr/bin/env bash
# convert_and_quantize.sh
# ----------------------------------------------------------------------
# RODIN-1B : conversion d'un dossier HF -> GGUF F16, puis quantization
# Q8_0 et Q4_K_M, via llama.cpp.
#
# Prerequis : un dossier HF VALIDE (export_rodin_to_hf.py + verify_hf_export.py).
# Ne PAS quantizer un export non verifie.
#
# Usage :
#   ./convert_and_quantize.sh <dossier_hf> <prefixe_sortie> [dossier_gguf_out]
# Exemples :
#   ./convert_and_quantize.sh ./rodin_hf_instruct rodin-1b-instruct ./gguf
#   ./convert_and_quantize.sh ./rodin_hf_base     rodin-1b-base     ./gguf
#
# Produit dans <dossier_gguf_out> :
#   <prefixe>-f16.gguf   (pivot, qualite max)
#   <prefixe>-Q8_0.gguf
#   <prefixe>-Q4_K_M.gguf
# ----------------------------------------------------------------------
set -euo pipefail

HF_DIR="${1:?usage: ./convert_and_quantize.sh <dossier_hf> <prefixe> [out]}"
PREFIX="${2:?prefixe de sortie manquant (ex: rodin-1b-instruct)}"
OUT_DIR="${3:-./gguf}"

LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
JOBS="$(nproc)"

mkdir -p "$OUT_DIR"

# --- 1. llama.cpp : clone + build si absent --------------------------
if [ ! -d "$LLAMA_DIR" ]; then
  echo "[build] clone llama.cpp -> $LLAMA_DIR"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  echo "[build] compilation llama.cpp (CPU suffit pour quantizer)"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release -j "$JOBS" \
        --target llama-quantize llama-cli
fi

# dependances python du convertisseur (dans le venv actif)
echo "[deps] verif dependances convert_hf_to_gguf.py"
pip install -q -r "$LLAMA_DIR/requirements.txt" 2>/dev/null || \
  pip install -q numpy sentencepiece safetensors transformers gguf protobuf

QUANT="$LLAMA_DIR/build/bin/llama-quantize"
CONVERT="$LLAMA_DIR/convert_hf_to_gguf.py"

F16="$OUT_DIR/${PREFIX}-f16.gguf"
Q8="$OUT_DIR/${PREFIX}-Q8_0.gguf"
Q4="$OUT_DIR/${PREFIX}-Q4_K_M.gguf"

# --- 2. HF -> GGUF F16 (pivot) ---------------------------------------
echo "[convert] $HF_DIR -> $F16"
python "$CONVERT" "$HF_DIR" --outfile "$F16" --outtype f16

# --- 3. quantization -------------------------------------------------
echo "[quant] -> Q8_0"
"$QUANT" "$F16" "$Q8" Q8_0
echo "[quant] -> Q4_K_M"
"$QUANT" "$F16" "$Q4" Q4_K_M

echo
echo "[OK] GGUF produits dans $OUT_DIR :"
ls -lh "$F16" "$Q8" "$Q4"
echo
echo "Test rapide (instruct) :"
echo "  $LLAMA_DIR/build/bin/llama-cli -m $Q4 -p '<|im_start|>user\\nBonjour<|im_end|>\\n<|im_start|>assistant\\n' -n 80 --temp 0.7"

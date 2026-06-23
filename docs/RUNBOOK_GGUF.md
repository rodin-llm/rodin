# RODIN-1B — Runbook GGUF (base + instruct, F16 / Q8_0 / Q4_K_M)

Objectif de cette session : produire les GGUF + safetensors, compatibles
Ollama ET LM Studio, pour les deux modeles (base + instruct v3).

Sur la machine de travail (GPU local), venv actif :
    source /path/to/rodin/.venv/bin/activate
    nvidia-smi -pl 280        # canicule
    pip install transformers safetensors gguf protobuf   # si absents

Place les 4 fichiers de cette session dans un dossier de travail, ex :
    /path/to/rodin/release/
    ├── export_rodin_to_hf.py
    ├── verify_hf_export.py
    ├── convert_and_quantize.sh   (chmod +x)
    ├── Modelfile.instruct
    ├── Modelfile.base
    └── 21_train_rodin.py         (copie, requis par verify_hf_export.py)

------------------------------------------------------------------------
ETAPE 1 — Export HF (les deux modeles)
------------------------------------------------------------------------
# INSTRUCT v3 (le livrable principal)
python export_rodin_to_hf.py \
  --weights /path/to/rodin/runs/sft_v3/rodin1b_instruct_bf16.pt \
  --tokenizer /path/to/rodin/bpe/rodin.model \
  --preset prod --variant instruct --dtype bf16 \
  --out ./rodin_hf_instruct

# BASE (pour ceux qui veulent reprendre le pretraining/SFT)
python export_rodin_to_hf.py \
  --weights /path/to/rodin/weights/rodin1b_weights_bf16.pt \
  --tokenizer /path/to/rodin/bpe/rodin.model \
  --preset prod --variant base --dtype bf16 \
  --out ./rodin_hf_base

# Les dossiers ./rodin_hf_* contiennent deja le safetensors BF16 + config +
# tokenizer HF : c'est exactement ce que tu publies pour "reprendre plus loin".

------------------------------------------------------------------------
ETAPE 2 — Verification (CRITIQUE, avant tout GGUF)
------------------------------------------------------------------------
python verify_hf_export.py \
  --weights /path/to/rodin/runs/sft_v3/rodin1b_instruct_bf16.pt \
  --hf ./rodin_hf_instruct \
  --train-file ./21_train_rodin.py --preset prod

# Attendu : "max-abs < 1e-2" ET "argmax match = 100.00%" -> [OK] mapping VALIDE.
# Idem pour la base avec --weights .../rodin1b_weights_bf16.pt --hf ./rodin_hf_base
# Si ECHEC -> NE PAS quantizer. Le mapping est faux, on debug avant.

------------------------------------------------------------------------
ETAPE 3 — Conversion + quantization GGUF
------------------------------------------------------------------------
chmod +x convert_and_quantize.sh
./convert_and_quantize.sh ./rodin_hf_instruct rodin-1b-instruct ./gguf
./convert_and_quantize.sh ./rodin_hf_base     rodin-1b-base     ./gguf

# Produit ./gguf/ :
#   rodin-1b-instruct-{f16,Q8_0,Q4_K_M}.gguf
#   rodin-1b-base-{f16,Q8_0,Q4_K_M}.gguf

------------------------------------------------------------------------
ETAPE 4 — Ollama
------------------------------------------------------------------------
ollama create rodin-1b-instruct -f Modelfile.instruct
ollama run rodin-1b-instruct "Explique-moi simplement la photosynthese."

ollama create rodin-1b-base -f Modelfile.base
ollama run rodin-1b-base "La France est"

------------------------------------------------------------------------
ETAPE 5 — LM Studio
------------------------------------------------------------------------
LM Studio lit les GGUF directement, sans Modelfile. Le chat_template ChatML
est deja embarque dans le GGUF instruct (via tokenizer_config.json a l'export),
donc LM Studio detecte le format ChatML automatiquement.

1. Copier les .gguf dans le dossier modeles de LM Studio :
   ~/.lmstudio/models/rodin-llm/rodin-1b/   (un sous-dossier par modele)
   ou utiliser "Import model" depuis l'UI et pointer le fichier .gguf.
2. Cote chat : selectionner le preset "ChatML" si non auto-detecte, et
   verifier que le stop "<|im_end|>" est present (il l'est via le template).
3. Le modele base : le charger en mode "completion" (pas chat), pas de template.

------------------------------------------------------------------------
ETAPE 6 — Arborescence de publication HuggingFace (rodin-llm/rodin-1b)
------------------------------------------------------------------------
Recommandation : UN repo avec sous-dossiers, ou deux repos (-base / -instruct).
Structure repo unique conseillee :

  rodin-llm/rodin-1b/
  ├── README.md                      (model card, cf. MODEL_CARD.md)
  ├── instruct/
  │   ├── config.json                \
  │   ├── model.safetensors           |  export HF BF16 (reprise possible)
  │   ├── tokenizer.model             |
  │   ├── tokenizer_config.json       |
  │   ├── special_tokens_map.json     /
  │   ├── rodin-1b-instruct-f16.gguf
  │   ├── rodin-1b-instruct-Q8_0.gguf
  │   └── rodin-1b-instruct-Q4_K_M.gguf
  └── base/
      ├── config.json + safetensors + tokenizer...
      ├── rodin-1b-base-f16.gguf
      ├── rodin-1b-base-Q8_0.gguf
      └── rodin-1b-base-Q4_K_M.gguf

Upload (huggingface_hub) :
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli upload rodin-llm/rodin-1b ./rodin_hf_instruct instruct
  huggingface-cli upload rodin-llm/rodin-1b ./gguf .   # ou par sous-dossier

------------------------------------------------------------------------
TABLEAU DES LIVRABLES
------------------------------------------------------------------------
                 | safetensors BF16 | GGUF f16 | GGUF Q8_0 | GGUF Q4_K_M
RODIN-1B base    |        X         |    X     |     X     |     X
RODIN-1B-Instruct|        X         |    X     |     X     |     X

Tailles indicatives (1.24B params) :
  safetensors bf16 : ~2.5 Go      f16 gguf : ~2.5 Go
  Q8_0 : ~1.3 Go                  Q4_K_M : ~0.8 Go

------------------------------------------------------------------------
PIEGES SPECIFIQUES RODIN (deja geres dans les scripts)
------------------------------------------------------------------------
- ChatML non-atomique : <|im_start|>/<|im_end|> se decomposent en 2 sous-tokens
  ([63769,4]/[63769,5]). Le stop est gere comme CHAINE par Ollama/LM Studio,
  pas comme token id. C'est pour ca qu'on n'ajoute pas de tokens au vocab.
- weight tying : safetensors interdit deux tenseurs partageant la memoire ->
  lm_head.weight est clone() a l'export. config.json garde tie_word_embeddings=true.
- RoPE rotate_half (non-interleaved) + n_kv_heads==n_heads : convention Llama
  exacte, AUCUNE permutation de poids. C'est pour ca que le verify doit passer
  a 100% d'argmax du premier coup.
- Le "assistant" parasite de sample.py n'apparait pas en runtime (le template
  est construit par Ollama/LM Studio).

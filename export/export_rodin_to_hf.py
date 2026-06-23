#!/usr/bin/env python3
# export_rodin_to_hf.py
# ----------------------------------------------------------------------
# RODIN-1B : export d'un checkpoint RodinLM (.pt) vers le format
# HuggingFace LlamaForCausalLM (config.json + model.safetensors + tokenizer).
#
# Pourquoi : llama.cpp (convert_hf_to_gguf.py) ne lit PAS un .pt arbitraire.
# Il faut un dossier HF standard. L'architecture de RodinLM est un Llama
# a l'octet pres (RoPE rotate_half non-interleaved, SwiGLU w2(silu(w1)*w3),
# RMSNorm, weight tying, pas de bias, n_kv_heads == n_heads), donc la
# conversion est un simple RENOMMAGE de cles, sans permutation de poids.
#
# Le .pt accepte est :
#   - soit un checkpoint complet {"model": state_dict, "cfg": {...}, ...}
#     (format save_checkpoint de 21_train_rodin.py),
#   - soit un poids "deja extrait" : {"model": sd} ou directement un sd.
#
# Tokens ChatML : <|im_start|> / <|im_end|> ne sont PAS des tokens
# SentencePiece uniques (ils se decomposent : [63769,4] / [63769,5]).
# On embarque donc le chat_template + on declarera le STOP comme la
# CHAINE "<|im_end|>" cote runtime (Modelfile / LM Studio), pas un id.
#
# Usage :
#   python export_rodin_to_hf.py \
#       --weights /path/to/rodin/.../rodin1b_instruct_v3_bf16.pt \
#       --tokenizer /path/to/rodin/bpe/rodin.model \
#       --preset prod \
#       --out ./rodin_hf_instruct \
#       --dtype bf16 \
#       --variant instruct
#
# --variant {base,instruct} ne change que le chat_template embarque
# (l'instruct recoit le template ChatML ; la base non).
# ----------------------------------------------------------------------

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file


# Doit correspondre EXACTEMENT a model_config() de 21_train_rodin.py.
PRESETS = {
    "pretest": dict(dim=1024, n_layers=28, n_heads=16, n_kv_heads=16,
                    ffn_hidden=2730, vocab=64000, ctx=2048,
                    rope_theta=10000.0, rmsnorm_eps=1e-5),
    "prod":    dict(dim=2048, n_layers=22, n_heads=16, n_kv_heads=16,
                    ffn_hidden=5461, vocab=64000, ctx=2048,
                    rope_theta=10000.0, rmsnorm_eps=1e-5),
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Tokens speciaux (ids SentencePiece RODIN, verifies via sp.id_to_piece) :
# 0=<pad>, 1=<unk>, 2=<s>, 3=</s>.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def load_state_dict(path):
    """Charge le .pt et renvoie (state_dict, cfg_embarquee_ou_None)."""
    print(f"[load] {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg = None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
        cfg = obj.get("cfg")
        if "step" in obj:
            print(f"[load] checkpoint step={obj['step']}")
    elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
    else:
        sys.exit("[ERREUR] format .pt non reconnu (ni {'model':sd} ni state_dict brut).")
    # nettoie un eventuel prefixe de torch.compile
    cleaned = {}
    for k, v in sd.items():
        cleaned[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
    return cleaned, cfg


def _align_up(n, mult):
    return ((n + mult - 1) // mult) * mult


def _pad_rows(t, target_rows):
    """Pad un tenseur (rows, cols) avec des lignes de zeros -> (target_rows, cols)."""
    import torch as _t
    r, c = t.shape
    if r == target_rows:
        return t
    pad = _t.zeros(target_rows - r, c, dtype=t.dtype)
    return _t.cat([t, pad], dim=0)


def _pad_cols(t, target_cols):
    """Pad un tenseur (rows, cols) avec des colonnes de zeros -> (rows, target_cols)."""
    import torch as _t
    r, c = t.shape
    if c == target_cols:
        return t
    pad = _t.zeros(r, target_cols - c, dtype=t.dtype)
    return _t.cat([t, pad], dim=1)


def map_rodin_to_llama(sd, cfg, ffn_padded):
    """Renomme les cles RodinLM -> HF Llama. Aucune permutation de poids.

    Si ffn_padded > ffn_hidden, on agrandit gate/up (lignes) et down (colonnes)
    avec des ZEROS. Equivalence stricte en inference :
      - gate/up gagnent des neurones dont la sortie vaut 0 (silu(0)*0 = 0 pour
        les colonnes ajoutees du produit gate*up),
      - down a les colonnes correspondantes a 0 -> contribution nulle.
    Necessaire car llama.cpp quantize par blocs de 32 : ffn doit etre aligne.
    """
    n_layers = cfg["n_layers"]
    ffn = cfg["ffn_hidden"]
    out = {}

    def take(name):
        if name not in sd:
            raise KeyError(f"tenseur manquant dans le checkpoint : {name}")
        return sd[name]

    # embeddings
    embed = take("tok_emb.weight")
    out["model.embed_tokens.weight"] = embed

    # couches
    for i in range(n_layers):
        p = f"blocks.{i}"
        h = f"model.layers.{i}"
        out[f"{h}.input_layernorm.weight"]          = take(f"{p}.attn_norm.weight")
        out[f"{h}.self_attn.q_proj.weight"]         = take(f"{p}.attn.wq.weight")
        out[f"{h}.self_attn.k_proj.weight"]         = take(f"{p}.attn.wk.weight")
        out[f"{h}.self_attn.v_proj.weight"]         = take(f"{p}.attn.wv.weight")
        out[f"{h}.self_attn.o_proj.weight"]         = take(f"{p}.attn.wo.weight")
        out[f"{h}.post_attention_layernorm.weight"] = take(f"{p}.ffn_norm.weight")
        # SwiGLU : gate(w1) et up(w3) ont la sortie en lignes -> pad lignes ;
        # down(w2) prend l'intermediate en entree -> pad colonnes.
        gate = take(f"{p}.ffn.w1.weight")   # (ffn, dim)
        up   = take(f"{p}.ffn.w3.weight")   # (ffn, dim)
        down = take(f"{p}.ffn.w2.weight")   # (dim, ffn)
        if ffn_padded > ffn:
            gate = _pad_rows(gate, ffn_padded)
            up   = _pad_rows(up,   ffn_padded)
            down = _pad_cols(down, ffn_padded)
        out[f"{h}.mlp.gate_proj.weight"] = gate
        out[f"{h}.mlp.up_proj.weight"]   = up
        out[f"{h}.mlp.down_proj.weight"] = down

    # norme finale
    out["model.norm.weight"] = take("norm.weight")

    # lm_head : tie. Si lm_head.weight existe et differe, on le prend ;
    # sinon on reutilise l'embedding (weight tying RodinLM).
    if "lm_head.weight" in sd and sd["lm_head.weight"].data_ptr() != embed.data_ptr():
        out["lm_head.weight"] = sd["lm_head.weight"]
    else:
        out["lm_head.weight"] = embed.clone()  # safetensors interdit le partage

    return out


def sanity_check(sd, cfg):
    """Verifie les shapes attendues avant export. Guard-fail explicite."""
    dim, vocab = cfg["dim"], cfg["vocab"]
    hd = dim // cfg["n_heads"]
    ffn = cfg["ffn_hidden"]
    nkv = cfg["n_kv_heads"]
    checks = [
        ("model.embed_tokens.weight", (vocab, dim)),
        ("lm_head.weight", (vocab, dim)),
        ("model.norm.weight", (dim,)),
        ("model.layers.0.self_attn.q_proj.weight", (cfg["n_heads"] * hd, dim)),
        ("model.layers.0.self_attn.k_proj.weight", (nkv * hd, dim)),
        ("model.layers.0.self_attn.v_proj.weight", (nkv * hd, dim)),
        ("model.layers.0.self_attn.o_proj.weight", (dim, cfg["n_heads"] * hd)),
        ("model.layers.0.mlp.gate_proj.weight", (ffn, dim)),
        ("model.layers.0.mlp.up_proj.weight", (ffn, dim)),
        ("model.layers.0.mlp.down_proj.weight", (dim, ffn)),
        ("model.layers.0.input_layernorm.weight", (dim,)),
        ("model.layers.0.post_attention_layernorm.weight", (dim,)),
    ]
    for name, shape in checks:
        got = tuple(sd[name].shape)
        if got != shape:
            sys.exit(f"[ERREUR] shape {name} = {got}, attendu {shape}")
    print(f"[check] shapes OK ({len(sd)} tenseurs)")


def write_config(out_dir, cfg, dtype_str):
    hf_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype_str]
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg["dim"],
        "intermediate_size": cfg["ffn_hidden"],
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["n_heads"],
        "num_key_value_heads": cfg["n_kv_heads"],
        "head_dim": cfg["dim"] // cfg["n_heads"],
        "max_position_embeddings": cfg["ctx"],
        "vocab_size": cfg["vocab"],
        "rms_norm_eps": cfg["rmsnorm_eps"],
        "rope_theta": cfg["rope_theta"],
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "torch_dtype": hf_dtype,
        "transformers_version": "4.44.0",
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    # generation_config minimal (cohabite avec les PARAMETER du Modelfile)
    gen = {"bos_token_id": BOS_ID, "eos_token_id": EOS_ID,
           "pad_token_id": PAD_ID, "max_length": cfg["ctx"]}
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen, f, indent=2)
    print("[write] config.json + generation_config.json")


def write_tokenizer(out_dir, tok_path, variant):
    # 1) tokenizer.model brut (SentencePiece) -> lu par convert_hf_to_gguf.py
    shutil.copyfile(tok_path, os.path.join(out_dir, "tokenizer.model"))

    # 2) tokenizer_config.json : type Llama (SentencePiece), tokens speciaux.
    #    IMPORTANT : ne PAS forcer pad=<unk>. Le vocab a <pad> a l'id 0 et
    #    <unk> a l'id 1, distincts. Forcer pad=<unk> faisait renommer l'id 0
    #    en <unk> -> deux tokens de texte identique -> llama.cpp plante
    #    (id_to_token.size() != token_to_id.size()). On respecte les vrais
    #    textes du SentencePiece : 0=<pad>, 1=<unk>, 2=<s>, 3=</s>.
    pad_tok = {"content": "<pad>", "lstrip": False, "normalized": False,
               "rstrip": False, "single_word": False, "special": True}
    unk_tok = {"content": "<unk>", "lstrip": False, "normalized": False,
               "rstrip": False, "single_word": False, "special": True}
    bos_tok = {"content": "<s>", "lstrip": False, "normalized": False,
               "rstrip": False, "single_word": False, "special": True}
    eos_tok = {"content": "</s>", "lstrip": False, "normalized": False,
               "rstrip": False, "single_word": False, "special": True}
    tok_cfg = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 2048,
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "added_tokens_decoder": {
            str(PAD_ID): pad_tok,   # 0 = <pad>
            str(UNK_ID): unk_tok,   # 1 = <unk>
            str(BOS_ID): bos_tok,   # 2 = <s>
            str(EOS_ID): eos_tok,   # 3 = </s>
        },
    }
    if variant == "instruct":
        tok_cfg["chat_template"] = CHATML_TEMPLATE
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token": "<s>", "eos_token": "</s>",
                   "unk_token": "<unk>", "pad_token": "<pad>"}, f, indent=2)
    print(f"[write] tokenizer.model + tokenizer_config.json (variant={variant})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True, help="rodin.model (SentencePiece)")
    ap.add_argument("--preset", default="prod", choices=list(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--variant", default="instruct", choices=["base", "instruct"])
    args = ap.parse_args()

    if not os.path.isfile(args.tokenizer):
        sys.exit(f"[ERREUR] tokenizer introuvable : {args.tokenizer}")
    os.makedirs(args.out, exist_ok=True)

    sd_raw, cfg_ck = load_state_dict(args.weights)
    cfg = dict(PRESETS[args.preset])
    if cfg_ck:
        # le cfg embarque prime (source de verite), mais on garde ctx/vocab figes
        for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden",
                  "rope_theta", "rmsnorm_eps"):
            if k in cfg_ck:
                cfg[k] = cfg_ck[k]
        print(f"[cfg] cfg embarquee utilisee : {cfg}")
    else:
        print(f"[cfg] preset {args.preset} : {cfg}")

    # FFN doit etre aligne pour la quantization GGUF (blocs de 32 ; K-quants
    # aiment 256). 5461 n'est pas aligne -> on pad a un multiple de 256.
    ffn_orig = cfg["ffn_hidden"]
    ffn_padded = _align_up(ffn_orig, 256)
    if ffn_padded != ffn_orig:
        print(f"[pad] ffn_hidden {ffn_orig} -> {ffn_padded} "
              f"(alignement 256, padding zero, equivalence stricte)")

    sd = map_rodin_to_llama(sd_raw, cfg, ffn_padded)

    # le config.json et le sanity check doivent refleter la taille paddee
    cfg = dict(cfg, ffn_hidden=ffn_padded)

    target = DTYPES[args.dtype]
    sd = {k: v.to(target).contiguous() for k, v in sd.items()}
    sanity_check(sd, cfg)

    save_file(sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    print(f"[write] model.safetensors ({args.dtype})")

    write_config(args.out, cfg, args.dtype)
    write_tokenizer(args.out, args.tokenizer, args.variant)

    print(f"\n[OK] export HF -> {args.out}")
    print("     Etape suivante OBLIGATOIRE : verify_hf_export.py avant GGUF.")


if __name__ == "__main__":
    main()

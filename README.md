<!--
  NOTE DE MAINTENANCE / MAINTENANCE NOTE
  Document bilingue (EN + FR). Editer une section = repercuter dans les DEUX langues.
  Bilingual document (EN + FR). Editing a section = update BOTH language blocks.
-->

# RODIN

**A French large language model, built from scratch — solo, on consumer-grade hardware.**
**Un grand modèle de langage français, construit de zéro — en solo, sur du matériel grand public.**

**[🇬🇧 English](#-english)** · **[🇫🇷 Français](#-français)**

This repository contains the **full source code** of the RODIN project: the complete pipeline used to build [RODIN-1B](https://huggingface.co/rodin-llm/rodin-1b), from raw data collection to a quantized GGUF model — every stage, reproducible.

---

<a name="-english"></a>
## 🇬🇧 English

### What is RODIN?

RODIN (**R**esearch **O**pen **D**eep **I**ntelligence **N**atively-french) is a French-only large language model trained **entirely from scratch** by a single person, as an open and reproducible research project. No fine-tune, no derivative: a custom tokenizer, a custom architecture, a hand-built data pipeline, and pretraining from random weights.

The first release, **RODIN-1B** (1.24 B parameters, 32 B training tokens), is available on Hugging Face:

- 🤗 [`rodin-llm/rodin-1b`](https://huggingface.co/rodin-llm/rodin-1b) — base model
- 🤗 [`rodin-llm/rodin-1b-instruct`](https://huggingface.co/rodin-llm/rodin-1b-instruct) — conversational model (+ GGUF)

### Why this project

The point was never to beat large, well-funded French models on raw scores. Comparable French open-source efforts ran on **3,000 billion tokens** across hundreds of H100 GPUs on national supercomputers. RODIN ran on **32 billion tokens**, one person, a rented spot B200 for pretraining and a single RTX 3090 for local iteration and SFT.

The value is **pedagogical and reproducible**: showing, end to end and honestly, what one motivated individual can build — data, tokenizer, architecture, training, evaluation, deployment, *and the cloud orchestration that makes spot-GPU pretraining survivable* — and documenting every decision, including the limitations.

### Repository structure

The code follows the actual pipeline order. Scripts are numbered by execution stage.

```
data/        # Stage 1 — data pipeline (download → clean → dedup → tokenize → blend)
training/    # Stage 2 — model architecture (RodinLM) + pretraining loop
sft/         # Stage 3 — supervised fine-tuning (ChatML) + SFT dataset generation
export/      # Stage 4 — RodinLM → HuggingFace Llama → GGUF conversion
infra/       # Spot-GPU orchestration (Verda): golden volume, startup, watchdog, bootstrap
inference/   # Sampling / probing + Ollama Modelfiles
docs/        # Runbooks
```

#### Pipeline overview

| Stage | Scripts | What it does |
|---|---|---|
| **Data** | `01`–`07` | Download sources, clean per source, MinHash dedup + quality filtering |
| | `08`–`09` | Inspect sources, stratified sampling for tokenizer training |
| | `10`–`11` | Train the custom 64K BPE tokenizer, validate fertility |
| | `12`–`13` | Tokenize the full corpus to `uint16` `.bin` shards |
| | `14`–`19` | Index docs ↔ sources ↔ tokens, hunt "ghost docs", measure OCR quality |
| | `20` | Build the final train/val blend by per-source token budget |
| **Training** | `21` | RodinLM architecture (RoPE, RMSNorm, SwiGLU) + pretraining loop |
| **SFT** | `22` | Full fine-tune on ChatML, loss masked on assistant responses |
| **Export** | `export_rodin_to_hf.py` | Map custom `RodinLM` tensors → HuggingFace `LlamaForCausalLM` |
| | `convert_and_quantize.sh` | HF → GGUF (F16, Q8_0, Q4_K_M) via llama.cpp |
| **Infra** | `01`–`02` | Build a persistent "golden volume", register the cloud startup script |
| | `03_watchdog.py` | Long-running watchdog: balance guard, spot-eviction detection, auto-resume, Telegram alerts |
| | `04` + `bootstrap` | GPU pretest, instance bootstrap, checkpoint push to cold storage |

> **Why an `infra/` layer?** Pretraining ran on **interruptible spot B200 instances**, which can be reclaimed at any moment. The orchestration layer makes that survivable: a persistent volume holds the code and checkpoints, a watchdog watches the account balance and the instance health, redeploys when capacity returns, and refuses to redeploy below a safety threshold so a zero balance never wipes the golden volume. This is the part that turns "rent a cheap GPU" into "actually finish a 60-day run on cheap GPUs."

### Data sources

Pretrained exclusively on open or public-domain French data: **HPLT** (CC0 packaging), **CC100**, **Wikipedia** & **Wikisource** (CC BY-SA), **Pleias** books & news (open / public domain), **Légifrance** (open license). For web-crawl sources, the open license covers the dataset packaging, not each underlying document.

### Model architecture

LLaMA-style, 1.238 B parameters: hidden size 2048, 22 layers, 16 attention heads (no GQA), FFN 5461, vocabulary 64,000, context 2048, RoPE θ=10,000, RMSNorm, SwiGLU, tied embeddings, bfloat16. Full details on the [model card](https://huggingface.co/rodin-llm/rodin-1b).

### Reproducing

The scripts use local paths and assume a working PyTorch + SentencePiece environment. They are provided as **reference and documentation** of the real pipeline, not as a turnkey one-click trainer. Large artifacts (weights, `.bin` shards, GGUF) are **not** in this repo — the models live on Hugging Face, and the tokenized corpus is regenerable from the data scripts. Set `HF_TOKEN` (and, for `infra/`, `VERDA_CLIENT_ID` / `VERDA_CLIENT_SECRET`, `RODIN_TG_TOKEN`) in your environment; never hard-code secrets.

### The RODIN family — and how you can help

RODIN-1B is the **first** model, not the last. The same pipeline scales to larger models — the architecture, data pipeline, and spot-GPU orchestration are size-agnostic. The repository is named `rodin` (not `rodin-1b`) precisely so the whole family can live here.

The blocker is not capability — it is **compute budget**. Below are honest order-of-magnitude estimates for training each size from scratch, anchored on RODIN-1B's *real* run (1× B200 spot, ~2.2 $/GPU-hour, Chinchilla-style ≈20 tokens/parameter):

| Model | Params | Train tokens | B200 GPU-hours (est.) | Spot cost (~2.2 $/h) |
|---|---|---|---|---|
| RODIN-1B | 1.24 B | 32 B *(actual run)* | ~140 h | **~285 €** *(done)* |
| RODIN-3B | 3 B | ~60 B | ~600 h | ~1,200 € |
| RODIN-7B | 7 B | ~140 B | ~2,600 h | ~5,200 € |
| RODIN-13B | 13 B | ~260 B | ~7,500 h | ~15,000 € |
| RODIN-17B | 17 B | ~340 B | ~12,000 h | ~24,000 € |

*Rough estimates: single-GPU spot pricing, no multi-GPU parallelism. Costs are total GPU-hours — running several B200s in parallel lowers wall-clock time but not the total. Storage, egress, and failed-run overhead not included.*

I have the pipeline, the method, and the track record (RODIN-1B, end to end, solo) to build any of these. What I don't have is the budget to rent the GPUs. **If you'd like to sponsor a larger RODIN — or simply want to see the family exist — the door is open.** Reach out: **rodin.lab@proton.me**

### License

**Apache 2.0** — see [LICENSE](./LICENSE). Covers the code and the released weights; data sources keep their own licenses.

### Transparency

Carried out by one person, with **AI assistance openly acknowledged** throughout. Thanks to EleutherAI (lm-evaluation-harness), the HPLT and Pleias teams, Wikimedia, and the llama.cpp / Ollama / LM Studio projects.

---

<a name="-français"></a>
## 🇫🇷 Français

### Qu'est-ce que RODIN ?

RODIN (**R**esearch **O**pen **D**eep **I**ntelligence **N**atively-french) est un grand modèle de langage uniquement francophone, entraîné **entièrement de zéro** par une seule personne, dans le cadre d'un projet de recherche ouvert et reproductible. Pas de fine-tune, pas de dérivé : un tokenizer maison, une architecture maison, un pipeline de données construit à la main, et un pré-entraînement depuis des poids aléatoires.

La première version, **RODIN-1B** (1,24 milliard de paramètres, 32 milliards de tokens d'entraînement), est disponible sur Hugging Face :

- 🤗 [`rodin-llm/rodin-1b`](https://huggingface.co/rodin-llm/rodin-1b) — modèle de base
- 🤗 [`rodin-llm/rodin-1b-instruct`](https://huggingface.co/rodin-llm/rodin-1b-instruct) — modèle conversationnel (+ GGUF)

### Pourquoi ce projet

Le but n'a jamais été de battre en score brut les gros modèles français bien financés. Des projets français open source comparables ont tourné sur **3 000 milliards de tokens** avec des centaines de GPU H100 sur des supercalculateurs nationaux. RODIN a tourné sur **32 milliards de tokens**, une seule personne, une instance B200 spot louée pour le pré-entraînement et une seule RTX 3090 pour l'itération locale et le SFT.

La valeur est **pédagogique et reproductible** : montrer, de bout en bout et honnêtement, ce qu'une personne motivée peut construire — données, tokenizer, architecture, entraînement, évaluation, déploiement, *et l'orchestration cloud qui rend le pré-entraînement sur GPU spot survivable* — en documentant chaque décision, limites comprises.

### Structure du dépôt

Le code suit l'ordre réel du pipeline. Les scripts sont numérotés par étape d'exécution.

```
data/        # Étape 1 — pipeline données (download → clean → dédup → tokenisation → blend)
training/    # Étape 2 — architecture du modèle (RodinLM) + boucle de pré-entraînement
sft/         # Étape 3 — fine-tuning supervisé (ChatML) + génération du dataset SFT
export/      # Étape 4 — RodinLM → HuggingFace Llama → conversion GGUF
infra/       # Orchestration GPU spot (Verda) : golden volume, startup, watchdog, bootstrap
inference/   # Échantillonnage / probe + Modelfiles Ollama
docs/        # Runbooks
```

#### Vue d'ensemble du pipeline

| Étape | Scripts | Rôle |
|---|---|---|
| **Données** | `01`–`07` | Téléchargement des sources, nettoyage par source, dédup MinHash + filtrage qualité |
| | `08`–`09` | Inspection des sources, échantillonnage stratifié pour le tokenizer |
| | `10`–`11` | Entraînement du tokenizer BPE 64K maison, validation de la fertilité |
| | `12`–`13` | Tokenisation du corpus complet en shards `.bin` `uint16` |
| | `14`–`19` | Indexation docs ↔ sources ↔ tokens, chasse aux « ghost docs », qualité OCR |
| | `20` | Construction du blend train/val final par budget-tokens par source |
| **Entraînement** | `21` | Architecture RodinLM (RoPE, RMSNorm, SwiGLU) + boucle de pré-entraînement |
| **SFT** | `22` | Full fine-tune ChatML, loss masquée sur les réponses de l'assistant |
| **Export** | `export_rodin_to_hf.py` | Mapping des tenseurs `RodinLM` → `LlamaForCausalLM` HuggingFace |
| | `convert_and_quantize.sh` | HF → GGUF (F16, Q8_0, Q4_K_M) via llama.cpp |
| **Infra** | `01`–`02` | Construction d'un « golden volume » persistant, enregistrement du startup script cloud |
| | `03_watchdog.py` | Watchdog au long cours : garde anti-solde-zéro, détection d'éviction spot, reprise auto, alertes Telegram |
| | `04` + `bootstrap` | Pretest GPU, bootstrap de l'instance, push des checkpoints vers le stockage froid |

> **Pourquoi une couche `infra/` ?** Le pré-entraînement a tourné sur des **instances spot B200 interruptibles**, qui peuvent être reprises à tout moment. La couche d'orchestration rend ça survivable : un volume persistant garde le code et les checkpoints, un watchdog surveille le solde du compte et la santé de l'instance, redéploie quand de la capacité revient, et refuse de redéployer sous un seuil de sécurité pour qu'un solde à zéro ne détruise jamais le golden volume. C'est la partie qui transforme « louer un GPU pas cher » en « réussir vraiment un run de 60 jours sur des GPU pas chers ».

### Sources de données

Pré-entraîné exclusivement sur des données françaises ouvertes ou du domaine public : **HPLT** (packaging CC0), **CC100**, **Wikipédia** & **Wikisource** (CC BY-SA), **Pleias** livres & presse (libre / domaine public), **Légifrance** (licence ouverte). Pour les sources issues de web crawl, la licence ouverte couvre le packaging du dataset, pas chaque document sous-jacent.

### Architecture du modèle

Style LLaMA, 1,238 milliard de paramètres : dimension cachée 2048, 22 couches, 16 têtes d'attention (pas de GQA), FFN 5461, vocabulaire 64 000, contexte 2048, RoPE θ=10 000, RMSNorm, SwiGLU, embeddings liés, bfloat16. Détails complets sur la [model card](https://huggingface.co/rodin-llm/rodin-1b).

### Reproduire

Les scripts utilisent des chemins locaux et supposent un environnement PyTorch + SentencePiece fonctionnel. Ils sont fournis comme **référence et documentation** du pipeline réel, pas comme un entraîneur clé en main. Les gros artefacts (poids, shards `.bin`, GGUF) ne sont **pas** dans ce dépôt — les modèles vivent sur Hugging Face, et le corpus tokenisé est régénérable depuis les scripts data. Définis `HF_TOKEN` (et, pour `infra/`, `VERDA_CLIENT_ID` / `VERDA_CLIENT_SECRET`, `RODIN_TG_TOKEN`) dans ton environnement ; ne jamais coder de secret en dur.

### La famille RODIN — et comment aider

RODIN-1B est le **premier** modèle, pas le dernier. Le même pipeline passe à l'échelle sur des modèles plus gros — l'architecture, le pipeline de données et l'orchestration GPU spot sont indépendants de la taille. Le dépôt s'appelle `rodin` (et pas `rodin-1b`) justement pour que toute la famille puisse y vivre.

Le blocage n'est pas la capacité technique — c'est le **budget compute**. Voici des estimations honnêtes, en ordre de grandeur, pour entraîner chaque taille de zéro, ancrées sur le run *réel* de RODIN-1B (1× B200 spot, ~2,2 $/GPU-heure, ratio Chinchilla ≈20 tokens/paramètre) :

| Modèle | Paramètres | Tokens d'entraînement | GPU-heures B200 (est.) | Coût spot (~2,2 $/h) |
|---|---|---|---|---|
| RODIN-1B | 1,24 B | 32 B *(run réel)* | ~140 h | **~285 €** *(fait)* |
| RODIN-3B | 3 B | ~60 B | ~600 h | ~1 200 € |
| RODIN-7B | 7 B | ~140 B | ~2 600 h | ~5 200 € |
| RODIN-13B | 13 B | ~260 B | ~7 500 h | ~15 000 € |
| RODIN-17B | 17 B | ~340 B | ~12 000 h | ~24 000 € |

*Estimations grossières : tarif spot mono-GPU, sans parallélisme multi-GPU. Les coûts sont en GPU-heures totales — faire tourner plusieurs B200 en parallèle réduit le temps réel mais pas le total. Stockage, egress et marge pour runs ratés non inclus.*

J'ai le pipeline, la méthode et la preuve par l'exemple (RODIN-1B, de bout en bout, en solo) pour construire n'importe lequel de ces modèles. Ce que je n'ai pas, c'est le budget pour louer les GPU. **Si vous souhaitez sponsoriser un RODIN plus grand — ou simplement voir la famille exister — la porte est ouverte.** Contact : **rodin.lab@proton.me**

### Licence

**Apache 2.0** — voir [LICENSE](./LICENSE). Couvre le code et les poids publiés ; les sources de données conservent leurs propres licences.

### Transparence

Mené par une seule personne, avec une **assistance IA assumée et transparente** tout du long. Merci à EleutherAI (lm-evaluation-harness), aux équipes HPLT et Pleias, à Wikimedia, et aux projets llama.cpp / Ollama / LM Studio.

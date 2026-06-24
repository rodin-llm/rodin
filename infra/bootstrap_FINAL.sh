#!/usr/bin/env bash
# ============================================================================
# RODIN — bootstrap.sh
# ----------------------------------------------------------------------------
# Le "cerveau d'amorcage" de l'instance B200. Vit sur le GOLDEN VOLUME
# (/data/scripts/bootstrap.sh) donc modifiable a chaud sans repasser par
# l'image Verda. Appele par le startup script Verda minimal (qui, lui, monte
# /data puis exec ce fichier).
#
# RESPONSABILITES :
#   1. S'assurer que le golden volume est monte sur /data (idempotent : si le
#      startup script l'a deja monte, on ne refait rien ; sinon on le monte).
#   2. Verifier l'integrite minimale du contenu (blend + scripts + venv).
#   3. Activer le venv (/data/.venv, torch cu128, deja installe -> survit aux kicks).
#   4. Lancer le trainer 21_train_rodin.py.
#        - resume AUTOMATIQUE : le trainer detecte seul le dernier ckpt_*.pt
#          dans --out et reprend a l'offset exact (window_offset). On relance
#          DONC TOUJOURS LA MEME COMMANDE, kick ou premier boot, peu importe.
#   5. (hook) push checkpoints async vers storage box -> CABLE PLUS TARD (3d).
#
# IDEMPOTENT & SUR : peut etre relance autant de fois que l'instance reboot.
# Tout echec fatal -> exit non-zero + log clair (le watchdog VPS verra que
# l'instance ne devient pas "saine" et redeploiera si besoin).
#
# Conventions trainer (verifiees dans 21_train_rodin.py) :
#   --train --val --out --preset prod --max-steps --batch-size --grad-accum
#   --ckpt-every --keep --compile-mode  (resume = relancer la meme cmd)
# ============================================================================

set -Eeuo pipefail

# --- repere temporel + log a la fois console ET fichier sur le golden volume -
TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(TS)] [bootstrap] $*"; }
die() { echo "[$(TS)] [bootstrap][FATAL] $*" >&2; exit 1; }

# ============================================================================
# CONFIG (toutes les valeurs figees du run vivent ICI, en haut, modifiables)
# ============================================================================
MOUNT_POINT="/data"
GOLDEN_DEV_CANDIDATES=("/dev/vdb" "/dev/sdb" "/dev/vdc" "/dev/nvme1n1")  # 1er device "data" plausible
GOLDEN_FS_LABEL=""            # optionnel : si tu labellises le FS, ex "rodin-golden"

VENV="${MOUNT_POINT}/.venv"
SCRIPTS_DIR="${MOUNT_POINT}/scripts"
BLEND_DIR="${MOUNT_POINT}/blend"
RUNS_DIR="${MOUNT_POINT}/runs"

TRAIN_BIN="${BLEND_DIR}/train.bin"
VAL_BIN="${BLEND_DIR}/val.bin"
TRAINER="${SCRIPTS_DIR}/21_train_rodin.py"

RUN_NAME="rodin1b_32b"                 # dossier checkpoints : /data/runs/<RUN_NAME>
OUT_DIR="${RUNS_DIR}/${RUN_NAME}"

# --- hyper-params du RUN cloud (preset prod = ~1B) -------------------------
PRESET="prod"
# 32B tokens cibles, 65536 tok/step -> ~488k steps. Marge large pour absorber
# les reprises (le trainer s'arrete au max_steps ; on dimensionne genereux).
# --- valeurs FIGEES au pretest B200 (16/06/2026) ---------------------------
# 130k tok/s mesure, 133 Go VRAM (50 Go de marge), ~67h / ~123 EUR pour 32B.
# tokens/step = 32*1*2048 = 65536 (cible du handoff, pile).
# reduce-overhead ECARTE : incompatible weight tying (CUDA graphs overwrite) +
#   glouton VRAM. compile-mode default = le bon choix (gain batch>32 marginal).
MAX_STEPS=478000                       # ~1 epoch du blend 32B (31.28e9 / 65536)
BATCH_SIZE=32                          # pretest : 130k tok/s, 133 Go (marge 50 Go)
GRAD_ACCUM=1                           # 32*1*2048 = 65536 tok/step
LR=3e-4
WARMUP=2000
CKPT_EVERY=1000                        # ckpt local /1000 steps (DECIDE)
KEEP=3                                 # rotation locale : garder 3 derniers (DECIDE)
EVAL_EVERY=2000
LOG_EVERY=50
NUM_WORKERS=12                         # pretest : GPU 100%, dataloader suit
PREFETCH_FACTOR=4
COMPILE_MODE="default"                 # reduce-overhead ECARTE (cf. ci-dessus)
SEED=1234

# Marqueur de "sante" : le watchdog peut SSH/poll ce fichier pour confirmer que
# le trainer tourne bien (au-dela du simple statut RUNNING de l'instance).
HEALTH_FILE="${MOUNT_POINT}/bootstrap_health.json"
BOOT_LOG="${MOUNT_POINT}/bootstrap.log"

# ============================================================================
# 0. Rediriger toute la sortie vers la console ET un log persistant sur /data
#    (mais d'abord, /data doit etre monte -> on tee apres le montage).
# ============================================================================
log "=== RODIN bootstrap demarre (PID $$) ==="

# ============================================================================
# 1. MONTAGE DU GOLDEN VOLUME (idempotent)
# ============================================================================
mount_golden() {
    if mountpoint -q "${MOUNT_POINT}"; then
        log "Golden volume deja monte sur ${MOUNT_POINT} (rien a faire)."
        return 0
    fi

    mkdir -p "${MOUNT_POINT}"

    # Strategie de detection du device, par ordre de fiabilite :
    #   a) par label de FS si renseigne
    #   b) premier candidat de la liste qui existe ET qui n'est pas le disque OS
    local dev=""

    if [[ -n "${GOLDEN_FS_LABEL}" ]] && blkid -L "${GOLDEN_FS_LABEL}" >/dev/null 2>&1; then
        dev="$(blkid -L "${GOLDEN_FS_LABEL}")"
        log "Device golden trouve par label '${GOLDEN_FS_LABEL}' : ${dev}"
    else
        # Disque racine (a NE PAS monter) : celui qui porte /
        local root_src
        root_src="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
        for cand in "${GOLDEN_DEV_CANDIDATES[@]}"; do
            if [[ -b "${cand}" ]]; then
                # eviter de prendre une partition du disque OS
                if [[ "${cand}" == "${root_src}"* ]]; then
                    continue
                fi
                dev="${cand}"
                break
            fi
        done
        [[ -n "${dev}" ]] && log "Device golden detecte : ${dev}"
    fi

    [[ -z "${dev}" ]] && die "Aucun device golden detecte (candidats: ${GOLDEN_DEV_CANDIDATES[*]}). lsblk:
$(lsblk -f 2>/dev/null || true)"

    # Le golden volume est deja formate (build phase) : on monte tel quel.
    # NE JAMAIS mkfs ici (ca detruirait le blend + les checkpoints).
    log "Montage ${dev} -> ${MOUNT_POINT}"
    mount "${dev}" "${MOUNT_POINT}" \
        || die "Echec montage ${dev} sur ${MOUNT_POINT}. blkid:
$(blkid "${dev}" 2>/dev/null || true)"

    mountpoint -q "${MOUNT_POINT}" || die "Apres mount, ${MOUNT_POINT} n'est pas un point de montage."
    log "Golden volume monte. df:"
    df -h "${MOUNT_POINT}" | sed 's/^/    /'
}

mount_golden

# A partir d'ici /data existe : on duplique la sortie vers un log persistant.
exec > >(tee -a "${BOOT_LOG}") 2>&1
log "Sortie dupliquee vers ${BOOT_LOG}"

# ============================================================================
# 2. VERIFICATIONS D'INTEGRITE (fail fast et clair)
# ============================================================================
log "Verification du contenu du golden volume…"
[[ -f "${TRAIN_BIN}" ]] || die "train.bin absent : ${TRAIN_BIN}"
[[ -f "${TRAINER}"   ]] || die "trainer absent : ${TRAINER}"
[[ -d "${VENV}"      ]] || die "venv absent : ${VENV}"
[[ -x "${VENV}/bin/python" ]] || die "python du venv introuvable/non exécutable : ${VENV}/bin/python"

# val.bin est optionnel cote trainer, mais on le signale s'il manque.
if [[ ! -f "${VAL_BIN}" ]]; then
    log "AVERTISSEMENT : val.bin absent (${VAL_BIN}). L'eval sera desactivee."
    VAL_ARG=()
else
    VAL_ARG=(--val "${VAL_BIN}")
fi

mkdir -p "${OUT_DIR}"
log "OUT_DIR = ${OUT_DIR}"

# Taille du train.bin (sanity, le cable SSD du PC a deja foire une fois) :
log "train.bin : $(ls -lh "${TRAIN_BIN}" | awk '{print $5}')  ($(stat -c%s "${TRAIN_BIN}") octets)"

# ============================================================================
# 3. ACTIVATION DU VENV
# ============================================================================
log "Activation du venv ${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate" || die "Echec activation venv."
cd "${SCRIPTS_DIR}" || die "cd ${SCRIPTS_DIR} impossible."

# ----------------------------------------------------------------------------
# DEPENDANCES SYSTEME pour torch.compile (Inductor).
# torch.compile genere du code C compile a la volee -> il lui faut Python.h
# (paquet -dev) + un compilo. Le volume OS est JETABLE (recree a chaque kick),
# donc on RE-INSTALLE a CHAQUE boot. Idempotent (apt ne refait rien si deja la,
# rapide) et best-effort (si apt echoue, on log et on continue : sans ca, le
# trainer planterait des le 1er step en compile, comme observe au pretest).
# ----------------------------------------------------------------------------
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.12)"
if ! python -c "import sysconfig, os; h=os.path.join(sysconfig.get_path('include'),'Python.h'); exit(0 if os.path.exists(h) else 1)" 2>/dev/null; then
    log "Installation des headers Python (python${PYVER}-dev + build-essential) pour torch.compile…"
    export DEBIAN_FRONTEND=noninteractive
    if apt-get update -qq && apt-get install -y -qq "python${PYVER}-dev" build-essential >/dev/null 2>&1; then
        log "  headers installes."
    else
        log "  AVERTISSEMENT : install headers echouee. torch.compile risque de planter."
        log "  -> repli possible : relancer le trainer en --compile-mode none (a la main)."
    fi
else
    log "Headers Python deja presents (Python.h trouve). Rien a installer."
fi

# Sanity GPU + torch (non bloquant : on log, on continue ; le trainer re-checkera)
log "Verif torch/CUDA :"
python - <<'PY' || log "AVERTISSEMENT : verif torch a echoue (le trainer tranchera)."
import torch
print(f"    torch {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"cuda_dispo={torch.cuda.is_available()}  "
      f"gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"    GPU0 = {torch.cuda.get_device_name(0)}")
PY

# ============================================================================
# 4. DETECTION DU CHECKPOINT (purement informatif : le trainer reprend SEUL)
# ============================================================================
LATEST_CKPT="$(ls -1t "${OUT_DIR}"/ckpt_*.pt 2>/dev/null | head -n1 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
    log "Checkpoint detecte -> REPRISE : ${LATEST_CKPT}"
    MODE="resume"
else
    log "Aucun checkpoint -> PREMIER DEMARRAGE (from scratch)."
    MODE="fresh"
fi

# ============================================================================
# 5. ECRITURE DU MARQUEUR DE SANTE (le watchdog peut le lire)
# ============================================================================
write_health() {
    cat > "${HEALTH_FILE}" <<JSON
{
  "ts": "$(TS)",
  "run_name": "${RUN_NAME}",
  "mode": "${MODE}",
  "latest_ckpt": "${LATEST_CKPT:-null}",
  "out_dir": "${OUT_DIR}",
  "pid": ${BASHPID},
  "hostname": "$(hostname)"
}
JSON
}
write_health
log "Marqueur de sante ecrit : ${HEALTH_FILE}"

# ============================================================================
# 6. DÉMON STORAGE BOX (backup froid checkpoints) — lancé en arrière-plan
# ----------------------------------------------------------------------------
# Le pousseur vit dans push_checkpoints.sh (script séparé, sur le golden
# volume). On le lance DÉTACHÉ : il surveille OUT_DIR et pousse 1 ckpt sur 3
# vers le box, best-effort. Il ne bloque JAMAIS le GPU (process indépendant).
# Si la clé box est absente, le démon s'arrête seul en loggant — le training
# continue quoi qu'il arrive (chemin non critique).
# ============================================================================
PUSHER="${SCRIPTS_DIR}/push_checkpoints.sh"
PUSH_LOG="${MOUNT_POINT}/push_checkpoints.log"

start_cold_backup_daemon() {
    if [[ ! -f "${PUSHER}" ]]; then
        log "AVERTISSEMENT : ${PUSHER} absent -> pas de backup froid (non bloquant)."
        return 0
    fi
    if [[ ! -f "<SSH_KEY_BOX>" ]]; then
        log "AVERTISSEMENT : clé box <SSH_KEY_BOX> absente -> backup froid désactivé (non bloquant)."
        return 0
    fi
    log "Lancement du démon de backup froid (détaché) -> ${PUSH_LOG}"
    # nohup + & + disown : survit à la fin du bootstrap, indépendant du trainer.
    nohup bash "${PUSHER}" >> "${PUSH_LOG}" 2>&1 &
    disown || true
    log "Démon backup froid lancé (PID $!)."
}
start_cold_backup_daemon

# ============================================================================
# 7. LANCEMENT DU TRAINER
# ----------------------------------------------------------------------------
# MEME commande au premier boot ET a la reprise : le trainer detecte le dernier
# ckpt dans --out et reprend a l'offset exact. C'est tout l'interet de l'archi.
# 'exec' : le trainer DEVIENT le process principal -> propre pour systemd /
# pour la terminaison a l'eviction (signaux transmis directement au trainer).
# ============================================================================
log "Lancement trainer (${MODE}) preset=${PRESET} out=${OUT_DIR}"
log "Commande :"
set -x
exec python -u "${TRAINER}" \
    --train "${TRAIN_BIN}" \
    "${VAL_ARG[@]}" \
    --out "${OUT_DIR}" \
    --preset "${PRESET}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --warmup "${WARMUP}" \
    --ckpt-every "${CKPT_EVERY}" \
    --keep "${KEEP}" \
    --eval-every "${EVAL_EVERY}" \
    --log-every "${LOG_EVERY}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --compile-mode "${COMPILE_MODE}" \
    --seed "${SEED}"
# (exec : aucune ligne apres ne sera atteinte ; le trainer est le PID principal)

#!/usr/bin/env bash
# ============================================================================
# RODIN — push_checkpoints.sh
# ----------------------------------------------------------------------------
# DÉMON DE BACKUP FROID des checkpoints vers le STORAGE BOX Hetzner.
# Lancé en arrière-plan par bootstrap.sh (étape 3d). DÉCOUPLÉ du trainer :
# il SURVEILLE le dossier de run et pousse, sans jamais bloquer le GPU.
#
# RÔLE EXACT (à ne pas surinterpréter) :
#   - Le chemin de reprise NORMAL (kick spot) utilise le ckpt LOCAL sur le
#     golden volume (qui survit aux kicks). Le box ne joue AUCUN rôle là.
#   - Le box = ASSURANCE catastrophe : "si Verda disparaît / golden volume
#     perdu". On y pousse donc 1 ckpt sur N (pas tous : inutile, le local
#     couvre le chemin normal), avec rotation pour ne pas saturer le box.
#
# MÉCANIQUE :
#   - boucle toutes les POLL_SECONDS
#   - liste les ckpt_*.pt COMPLETS (le trainer écrit en .tmp puis rename, donc
#     un .pt sans .tmp est toujours intègre)
#   - ne pousse QUE ceux dont le step est multiple de PUSH_EVERY_STEPS
#   - saute ceux déjà poussés (journal local des steps poussés)
#   - après push : purge le box pour ne garder que KEEP_ON_BOX derniers
#   - best-effort : tout échec réseau est loggé, JAMAIS fatal (on retente au
#     prochain tour). Le training continue quoi qu'il arrive.
#
# AUTH : clé dédiée sur le golden volume (<SSH_KEY_BOX>), sous-compte
#   Hetzner isolé (<STORAGE_BOX_USER>, jail /rodin). Validé bout-en-bout en test.
#
# Lancement (par bootstrap.sh) :
#   nohup bash /data/scripts/push_checkpoints.sh >> /data/push_checkpoints.log 2>&1 &
# Lancement manuel (debug) :
#   bash push_checkpoints.sh
# ============================================================================

set -uo pipefail   # PAS -e : un échec réseau ne doit pas tuer le démon.

# ============================================================================
# CONFIG (tout est ici, modifiable en une ligne)
# ============================================================================
RUN_NAME="rodin1b_32b"
OUT_DIR="/data/runs/${RUN_NAME}"        # où le trainer écrit les ckpt_*.pt

# Cadence : pousser 1 ckpt sur 3. Le trainer checkpointe tous les 1000 steps,
# donc PUSH_EVERY_STEPS=3000 => 1 sur 3. (Mettre 2000 pour 1 sur 2, etc.)
PUSH_EVERY_STEPS=3000

# Rétention sur le box : garder les N ckpt les plus récents (rotation).
# 10 × ~13 Go = ~130 Go, le box (1 To, ~974 Go libres) tient large.
KEEP_ON_BOX=10

# Storage box (sous-compte isolé, clé sur le golden volume).
BOX_USER="<STORAGE_BOX_USER>"
BOX_HOST="<STORAGE_BOX_HOST>"
BOX_PORT=23
BOX_KEY="<SSH_KEY_BOX>"
BOX_DEST_DIR="checkpoints"               # sous-dossier dans le jail du sous-compte

# Fréquence de surveillance du dossier (s). Le push lui-même prend ~90s pour
# ~13 Go ; inutile de poller trop vite.
POLL_SECONDS=60

# Journal local des steps déjà poussés (sur le golden volume, survit aux kicks).
PUSHED_LOG="/data/runs/${RUN_NAME}/.pushed_to_box.log"

# Commande SSH commune (port + clé + accept-new pour instance fraîche).
# /!\ Le storage box Hetzner est un shell RESTREINT : il tolère `ls` et `df`
# mais N'EXÉCUTE PAS `rm`, ni un enchaînement `cd ...; rm ...; rm ...`.
# Les SUPPRESSIONS doivent passer par SFTP (protocole supporté), pas par ssh.
SSH_CMD="ssh -p ${BOX_PORT} -i ${BOX_KEY} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
SFTP_CMD="sftp -P ${BOX_PORT} -i ${BOX_KEY} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

# ============================================================================
TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(TS)] [push] $*"; }

log "=== Démon backup froid démarré (PID $$) ==="
log "OUT_DIR=${OUT_DIR}  push 1/$(( PUSH_EVERY_STEPS / 1000 ))  keep=${KEEP_ON_BOX}  box=${BOX_USER}@${BOX_HOST}:${BOX_DEST_DIR}/"

# --- prérequis -------------------------------------------------------------
if [[ ! -f "${BOX_KEY}" ]]; then
    log "FATAL : clé box absente (${BOX_KEY}). Le backup froid ne peut pas démarrer."
    log "        (le training continue ; corriger la clé puis relancer ce démon.)"
    exit 1
fi
chmod 600 "${BOX_KEY}" 2>/dev/null || true
mkdir -p "$(dirname "${PUSHED_LOG}")"
touch "${PUSHED_LOG}"

# --- helpers ---------------------------------------------------------------

# step (entier) extrait d'un nom ckpt_00003000.pt -> 3000 (sans zéros de tête)
step_of() {
    local base; base="$(basename "$1")"          # ckpt_00003000.pt
    local num; num="${base#ckpt_}"; num="${num%.pt}"   # 00003000
    echo $((10#${num}))                           # 10# = force base-10 (évite octal sur zéros)
}

already_pushed() {
    grep -qxF "$1" "${PUSHED_LOG}" 2>/dev/null
}

mark_pushed() {
    echo "$1" >> "${PUSHED_LOG}"
}

# crée le dossier de destination sur le box (idempotent, best-effort)
ensure_box_dir() {
    ${SSH_CMD} "${BOX_USER}@${BOX_HOST}" "mkdir -p ${BOX_DEST_DIR}" 2>/dev/null
}

# pousse un fichier ckpt vers le box. Renvoie 0 si OK.
push_one() {
    local path="$1"
    local name; name="$(basename "${path}")"
    log "Push ${name} ($(du -h "${path}" 2>/dev/null | cut -f1))…"
    # rsync : reprend un transfert interrompu (--partial), barre de progression
    # off (log propre), --inplace pour ne pas doubler l'espace sur le box.
    if rsync -a --partial --inplace \
            -e "${SSH_CMD}" \
            "${path}" \
            "${BOX_USER}@${BOX_HOST}:${BOX_DEST_DIR}/"; then
        log "  OK ${name}"
        return 0
    else
        log "  ÉCHEC push ${name} (réseau ?). On retentera au prochain tour."
        return 1
    fi
}

# purge le box : ne garder que les KEEP_ON_BOX ckpt_*.pt les plus récents.
# (tri par nom = tri par step, car format zero-paddé sur 8 chiffres.)
# SUPPRESSION via SFTP (batch) : le shell restreint Hetzner refuse `ssh rm`.
purge_box() {
    # liste triée des ckpt sur le box (ls toléré par le box ; le pipe sort
    # s'exécute côté box, validé en prod).
    local listing
    listing="$(${SSH_CMD} "${BOX_USER}@${BOX_HOST}" \
        "ls -1 ${BOX_DEST_DIR}/ckpt_*.pt 2>/dev/null | sort")" || {
        log "Purge box : listing impossible (réseau ?), on saute cette fois."
        return 0
    }
    [[ -z "${listing}" ]] && return 0

    local total; total="$(echo "${listing}" | wc -l)"
    if (( total <= KEEP_ON_BOX )); then
        return 0
    fi

    local to_delete; to_delete="$(echo "${listing}" | head -n $(( total - KEEP_ON_BOX )))"
    local ndel; ndel="$(echo "${to_delete}" | wc -l)"
    log "Purge box : ${total} ckpt présents, suppression de ${ndel} ancien(s) via SFTP."

    # construire un batch SFTP : une ligne "rm <path>" par fichier, puis "bye".
    local batch; batch="$(mktemp /tmp/rodin_purge.XXXXXX)"
    while IFS= read -r f; do
        [[ -z "${f}" ]] && continue
        echo "rm ${f}"                     # chemin RELATIF tel que listé (checkpoints/ckpt_...)
    done <<< "${to_delete}" > "${batch}"
    echo "bye" >> "${batch}"

    # exécuter le batch. -b lit les commandes ; les options AVANT la destination.
    # sftp continue sur erreur d'une ligne (et logge), il ne s'arrête pas net.
    if ${SFTP_CMD} -b "${batch}" "${BOX_USER}@${BOX_HOST}" >/dev/null 2>&1; then
        log "  purge OK (${ndel} supprimé(s))"
    else
        # rc != 0 = au moins une ligne a échoué (ou réseau). On logge mais on ne
        # bloque pas : on retentera au prochain tour, le ménage est idempotent.
        log "  purge SFTP : au moins une suppression a échoué (réseau ?), on retentera."
    fi
    rm -f "${batch}"
}

# --- boucle principale -----------------------------------------------------
ensure_box_dir

while true; do
    # 1) PURGE D'ABORD, à CHAQUE tour, INDÉPENDAMMENT de tout push.
    #    C'est la correction du deadlock : avant, la purge ne tournait que si un
    #    push avait réussi -> box plein => push échoue => purge jamais appelée
    #    => box reste plein. En purgeant en tête de boucle, on libère la place
    #    AVANT d'essayer d'écrire. purge_box est idempotente et best-effort.
    purge_box

    # 2) lister les ckpt COMPLETS (sans .tmp) présents localement
    shopt -s nullglob
    candidates=( "${OUT_DIR}"/ckpt_*.pt )
    shopt -u nullglob

    for ckpt in "${candidates[@]}"; do
        # ignorer un éventuel .tmp (ne devrait pas matcher *.pt, sécurité)
        [[ "${ckpt}" == *.tmp ]] && continue

        step="$(step_of "${ckpt}")"
        # ne pousser que les multiples de PUSH_EVERY_STEPS
        if (( step % PUSH_EVERY_STEPS != 0 )); then
            continue
        fi
        # déjà poussé ?
        if already_pushed "$(basename "${ckpt}")"; then
            continue
        fi
        # pousser
        if push_one "${ckpt}"; then
            mark_pushed "$(basename "${ckpt}")"
        fi
    done

    sleep "${POLL_SECONDS}"
done

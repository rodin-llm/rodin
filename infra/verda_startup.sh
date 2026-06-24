#!/usr/bin/env bash
# ============================================================================
# RODIN — startup script VERDA (minimal)
# ----------------------------------------------------------------------------
# Ce script est enregistre cote VERDA (startup_script_id) et execute par
# l'instance A CHAQUE BOOT, AVANT toute intervention. Il vit dans l'image, donc
# IL EST FIGE : on ne peut pas le modifier sans repasser par Verda. C'est
# pourquoi il doit rester BETE et MINIMAL — toute l'intelligence (reprise,
# checkpoints, backups) vit dans /data/scripts/bootstrap.sh sur le golden
# volume, modifiable a chaud.
#
# Son unique role : monter le golden volume sur /data, puis donner la main a
# bootstrap.sh. Le montage est aussi fait (idempotent) par bootstrap.sh lui-meme
# -> si la detection de device echoue ici, bootstrap.sh re-tentera proprement.
#
# IMPORTANT : ce script tourne en root au boot. Sortie -> journal systemd de
# l'instance (consultable via la console Verda / SSH : journalctl).
# ============================================================================
set -uo pipefail   # PAS -e : un echec de montage ici ne doit pas tuer le boot,
                   # bootstrap.sh sait re-monter. On veut TOUJOURS l'atteindre.

MOUNT_POINT="/data"
BOOTSTRAP="${MOUNT_POINT}/scripts/bootstrap.sh"
DEV_CANDIDATES=("/dev/vdb" "/dev/sdb" "/dev/vdc" "/dev/nvme1n1")

echo "[verda-startup] $(date -u +%FT%TZ) demarrage"

# --- 1. Monter le golden volume (best-effort ; bootstrap re-tentera) --------
if ! mountpoint -q "${MOUNT_POINT}"; then
    mkdir -p "${MOUNT_POINT}"
    root_src="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
    for dev in "${DEV_CANDIDATES[@]}"; do
        if [[ -b "${dev}" && "${dev}" != "${root_src}"* ]]; then
            echo "[verda-startup] montage ${dev} -> ${MOUNT_POINT}"
            mount "${dev}" "${MOUNT_POINT}" && break || \
                echo "[verda-startup] echec mount ${dev}, candidat suivant"
        fi
    done
else
    echo "[verda-startup] ${MOUNT_POINT} deja monte"
fi

# --- 2. Donner la main au cerveau d'amorcage --------------------------------
if [[ -x "${BOOTSTRAP}" ]]; then
    echo "[verda-startup] exec ${BOOTSTRAP}"
    exec bash "${BOOTSTRAP}"
elif [[ -f "${BOOTSTRAP}" ]]; then
    echo "[verda-startup] ${BOOTSTRAP} present mais non exécutable -> bash direct"
    exec bash "${BOOTSTRAP}"
else
    echo "[verda-startup][FATAL] ${BOOTSTRAP} introuvable. Le golden volume est-il monte ?"
    echo "[verda-startup] lsblk :"
    lsblk -f || true
    exit 1
fi

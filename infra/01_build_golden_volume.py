#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_golden_volume.py — RODIN cloud, étape 3a.

OBJET
  Construire le "golden volume" : un volume NVMe DÉTACHÉ, persistant, qui
  contiendra le blend 32B (lecture seule) + le venv torch cu128 + les scripts,
  et qui survivra à toutes les évictions spot. C'est la pièce qui rend la
  reprise quasi-instantanée (le blend ne se re-télécharge jamais).

  Ce script NE FAIT PAS le transfert du blend lui-même : les 58 Go partent de
  ton PC Windows (source de vérité) via rclone, étape MANUELLE supervisée.
  Le script orchestre tout AUTOUR : création volume, instance jetable bon
  marché (V100 spot ~0.059/h), attente RUNNING, puis — après TON transfert —
  détache le volume et supprime l'instance.

DÉROULÉ (machine à états, reprenable)
  Le script fonctionne en PHASES explicites, pilotées par --phase :

    --phase create   : crée le golden volume détaché. Affiche son ID.
                       >>> NOTE L'ID, c'est la clé stable de TOUTE la reprise.
                       Sauvegardé aussi dans golden_volume.json.

    --phase deploy   : déploie l'instance jetable V100 avec le golden volume
                       attaché, attend RUNNING, affiche l'IP + le mémo des
                       commandes (format disque, mount, venv, rclone).
                       >>> Le script s'ARRÊTE ici. À toi de jouer (SSH + rclone).

    --phase teardown : APRÈS que tu aies rempli et vérifié le blend, détache
                       le golden volume et SUPPRIME l'instance jetable.
                       Le golden volume survit, prêt pour le run B200.

    --phase status   : affiche l'état courant (volume, instance) sans rien
                       modifier. Pour s'y retrouver entre deux sessions.

  Chaque phase est idempotente et lit/écrit golden_volume.json (état local).

SÉCURITÉ
  - Aucune ressource chère : V100 spot ~0.059/h. Le build coûte des centimes.
  - teardown ne touche JAMAIS au golden volume (seulement detach + delete
    instance). Le volume n'est supprimable que manuellement, exprès.

PRÉREQUIS
  - verda_lib.py dans le même dossier.
  - .env chargé (VERDA_CLIENT_ID / VERDA_CLIENT_SECRET).
  - Solde Verda > 0.
  - Clé SSH PC Windows enregistrée sur Verda (pour le rclone depuis le PC).

Auteur : RODIN — projet RODIN-1B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import verda_lib as vl

LOG = vl.get_logger("build_golden")

# ──────────────────────────────────────────────────────────────────────────
# Config (DECIDE)
# ──────────────────────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).resolve().parent / "golden_volume.json"

GOLDEN_VOLUME_NAME = "rodin-golden"
GOLDEN_VOLUME_SIZE_GB = 120          # blend 58 + venv ~6 + ckpts rotation ~39 + marge

BUILD_HOSTNAME = "rodin-build"
BUILD_OS_VOLUME_GB = 50              # volume OS jetable de l'instance de build

# ── Build sur instance CPU ──
# Pour remplir le golden volume (formater + copier le blend), aucun GPU n'est
# nécessaire. Une instance CPU est moins chère, plus souvent dispo, et son image
# générique 'ubuntu-24.04' évite l'erreur "OS not valid for this instance type"
# (chaque GPU n'accepte qu'un sous-ensemble d'images via son champ supported_os).
# torch cu128 s'installera quand même via pip sur le golden volume.
BUILD_GPU_HINTS = ("CPU",)           # types CPU.xV.yG
# Image générique CPU : PAS de suffixe CUDA (incompatible sinon). Largement supportée.
BUILD_IMAGE = "ubuntu-24.04"
# Plafond de prix : les CPU spot sont à quelques centimes/h. 0.30 large.
BUILD_MAX_PRICE = 0.30

# Chemin de montage du golden volume dans l'instance.
MOUNT_POINT = "/data"

# Le blend, côté PC Windows (source de vérité).
BLEND_SRC_WINDOWS = r"G:\data\rodin\blend"


# ──────────────────────────────────────────────────────────────────────────
# État local (golden_volume.json)
# ──────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("État sauvegardé dans %s", STATE_FILE.name)


# ──────────────────────────────────────────────────────────────────────────
# Phases
# ──────────────────────────────────────────────────────────────────────────

def phase_create(client) -> None:
    """Crée le golden volume détaché (si pas déjà fait)."""
    state = load_state()
    vol_id = state.get("golden_volume_id")

    if vol_id and vl.volume_exists(client, vol_id):
        LOG.info("Golden volume déjà existant : %s — rien à faire.", vol_id)
        return

    LOG.info("Création du golden volume NVMe détaché (%d Go)…", GOLDEN_VOLUME_SIZE_GB)
    vol_id = vl.create_detached_nvme_volume(
        client, name=GOLDEN_VOLUME_NAME, size_gb=GOLDEN_VOLUME_SIZE_GB
    )
    state["golden_volume_id"] = vol_id
    state["golden_volume_name"] = GOLDEN_VOLUME_NAME
    state["golden_volume_size_gb"] = GOLDEN_VOLUME_SIZE_GB
    save_state(state)

    print()
    print("=" * 70)
    print("  GOLDEN VOLUME CRÉÉ")
    print("=" * 70)
    print(f"  ID    : {vol_id}")
    print(f"  Nom   : {GOLDEN_VOLUME_NAME}")
    print(f"  Taille: {GOLDEN_VOLUME_SIZE_GB} Go (NVMe, détaché)")
    print()
    print("  >>> CET ID EST LA CLÉ STABLE DE TOUTE LA REPRISE.")
    print("  >>> Il est sauvegardé dans golden_volume.json. Ne le perds pas.")
    print("=" * 70)


def phase_deploy(client) -> None:
    """Déploie l'instance jetable V100 avec le golden volume attaché."""
    state = load_state()
    vol_id = state.get("golden_volume_id")
    if not vol_id:
        raise vl.RodinInfraError(
            "Pas de golden volume. Lance d'abord : --phase create"
        )
    if not vl.volume_exists(client, vol_id):
        raise vl.RodinInfraError(
            f"Le golden volume {vol_id} n'existe plus. Relance --phase create."
        )

    # Une instance de build déjà active ?
    existing_iid = state.get("build_instance_id")
    if existing_iid and vl.instance_is_alive(client, existing_iid):
        ip = vl._instance_ip(client, existing_iid)
        LOG.info("Instance de build déjà active : %s (IP %s)", existing_iid, ip)
        _print_fill_memo(ip, vol_id)
        return

    # Déployer le GPU le MOINS CHER réellement DISPONIBLE (fallback auto si la
    # capacité manque sur le 1er choix). Le modèle de GPU n'importe pas pour le
    # build : on ne fait que formater un disque et copier des fichiers.
    ssh_ids = vl.get_ssh_key_ids(client)

    # IMPORTANT : un volume et son instance DOIVENT être dans la même région.
    # On lit la région réelle du golden volume et on déploie l'instance LÀ.
    vol = vl.get_volume(client, vol_id)
    vol_region = (
        getattr(vol, "location", None)
        or getattr(vol, "location_code", None)
        or vl.DEFAULT_LOCATION
    )
    LOG.info("Golden volume en région %s → l'instance de build ira là.", vol_region)

    LOG.info("Déploiement instance de build (moins cher dispo, spot)…")
    try:
        res = vl.deploy_cheapest_available(
            client,
            model_hints=BUILD_GPU_HINTS,
            hostname=BUILD_HOSTNAME,
            ssh_key_ids=ssh_ids,
            image=BUILD_IMAGE,
            spot=True,
            os_volume_size_gb=BUILD_OS_VOLUME_GB,
            existing_volume_ids=[vol_id],     # ← golden volume attaché au boot
            description="rodin golden volume build (jetable)",
            max_price=BUILD_MAX_PRICE,
            locations=(vol_region,),          # ← même région que le volume
            wait_running=True,
        )
    except vl.CapacityUnavailable as exc:
        LOG.error(
            "Aucune instance CPU disponible en %s actuellement : %s\n"
            "Réessaie dans quelques minutes (la dispo spot fluctue).",
            vol_region, exc,
        )
        return

    state["build_instance_id"] = res.instance_id
    state["build_instance_ip"] = res.ip
    state["build_instance_type"] = res.instance_type
    save_state(state)

    _print_fill_memo(res.ip, vol_id)


def _print_fill_memo(ip: str, vol_id: str) -> None:
    """Affiche le mémo des commandes manuelles de remplissage."""
    memo = f"""
{"=" * 72}
  INSTANCE DE BUILD PRÊTE — À TOI DE JOUER (étape MANUELLE)
{"=" * 72}
  IP instance     : {ip}
  Golden volume   : {vol_id} (attaché)
  Point de montage: {MOUNT_POINT}

  ── 1. SSH dans l'instance (depuis le VPS ou le PC) ──
     ssh root@{ip}

  ── 2. Repérer le golden volume (le disque NON-OS) ──
     lsblk
     # cherche un disque ~{GOLDEN_VOLUME_SIZE_GB}G sans point de montage,
     # typiquement /dev/vdb (vda = OS). Adapte si besoin.

  ── 3. Formater (UNE SEULE FOIS — détruit tout sur le disque) + monter ──
     mkfs.ext4 -F /dev/vdb
     mkdir -p {MOUNT_POINT}
     mount /dev/vdb {MOUNT_POINT}
     mkdir -p {MOUNT_POINT}/blend {MOUNT_POINT}/scripts {MOUNT_POINT}/runs
     df -h {MOUNT_POINT}

  ── 4. Préparer le venv torch cu128 (Blackwell-ready) sur le volume ──
     apt-get update && apt-get install -y python3-venv rclone
     python3 -m venv {MOUNT_POINT}/.venv
     {MOUNT_POINT}/.venv/bin/pip install --upgrade pip
     {MOUNT_POINT}/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
     {MOUNT_POINT}/.venv/bin/pip install numpy
     {MOUNT_POINT}/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
     # NB: instance CPU -> torch.cuda.is_available() sera False, c'est NORMAL.
     # On valide ici que torch cu128 s'INSTALLE et s'IMPORTE. Le test GPU réel
     # se fera sur le B200 (benchmark, étape 4). 'cuda build' doit afficher 12.8.

  ── 5. Transférer le blend 58 Go DEPUIS TON PC WINDOWS (PowerShell) ──
     # sur le PC, pas sur l'instance :
     rclone copy "{BLEND_SRC_WINDOWS}" :sftp:{MOUNT_POINT}/blend `
       --sftp-host {ip} `
       --sftp-user root `
       --sftp-key-file "$env:USERPROFILE\\.ssh\\rodin_verda_pc" `
       --progress --transfers 4 --checkers 8

  ── 6. Vérifier le blend sur l'instance ──
     ls -lh {MOUNT_POINT}/blend/
     # doit montrer train.bin (~58 Go) + val.bin (~19 Mo) + blend_manifest.json

  ── 7. (optionnel) déposer rodin_data.py + 21_train_rodin.py ──
     #    dans {MOUNT_POINT}/scripts/ (via rclone aussi)

  ── 8. Démonter proprement AVANT teardown ──
     sync
     umount {MOUNT_POINT}

  Quand TOUT est vérifié, reviens sur le VPS et lance :
     python 01_build_golden_volume.py --phase teardown
{"=" * 72}
"""
    print(memo)


def phase_teardown(client) -> None:
    """Détache le golden volume + supprime l'instance jetable. Volume préservé."""
    state = load_state()
    iid = state.get("build_instance_id")
    vol_id = state.get("golden_volume_id")

    if not iid:
        LOG.info("Pas d'instance de build connue. Rien à supprimer.")
        return

    # Détacher le golden volume AVANT de supprimer l'instance (sécurité).
    if vol_id:
        try:
            vl.detach_volumes(client, [vol_id])
        except vl.APIException as exc:
            LOG.warning("Détachement volume : %s (peut être déjà détaché)", exc)

    LOG.info("Suppression de l'instance de build %s…", iid)
    vl.delete_instance(client, iid)

    state.pop("build_instance_id", None)
    state.pop("build_instance_ip", None)
    state.pop("build_instance_type", None)
    save_state(state)

    print()
    print("=" * 70)
    print("  TEARDOWN TERMINÉ")
    print("=" * 70)
    print(f"  Instance jetable supprimée.")
    print(f"  Golden volume PRÉSERVÉ : {vol_id}")
    print(f"  → prêt à être rattaché au run B200 (existing_volumes).")
    print("=" * 70)


def phase_status(client) -> None:
    """Affiche l'état courant sans rien modifier."""
    state = load_state()
    print()
    print("=" * 70)
    print("  ÉTAT GOLDEN VOLUME / BUILD")
    print("=" * 70)
    vol_id = state.get("golden_volume_id")
    if vol_id:
        alive = vl.volume_exists(client, vol_id)
        print(f"  Golden volume : {vol_id}  [{'EXISTE' if alive else 'ABSENT'}]")
        print(f"    nom={state.get('golden_volume_name')} "
              f"taille={state.get('golden_volume_size_gb')}Go")
    else:
        print("  Golden volume : (pas encore créé)")

    iid = state.get("build_instance_id")
    if iid:
        alive = vl.instance_is_alive(client, iid)
        ip = state.get("build_instance_ip", "?")
        print(f"  Instance build: {iid}  [{'RUNNING' if alive else 'absente/arrêtée'}]")
        print(f"    ip={ip} type={state.get('build_instance_type')}")
    else:
        print("  Instance build: (aucune)")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construction du golden volume RODIN (étape 3a).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Séquence type :
              python 01_build_golden_volume.py --phase create
              python 01_build_golden_volume.py --phase deploy
              # ... (SSH + rclone manuel, voir mémo affiché) ...
              python 01_build_golden_volume.py --phase teardown
        """),
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["create", "deploy", "teardown", "status"],
        help="Phase à exécuter.",
    )
    args = parser.parse_args()

    try:
        client = vl.build_client()
    except vl.RodinInfraError as exc:
        LOG.error("%s", exc)
        return 2

    # Garde-fou solde (sauf pour status, lecture seule).
    if args.phase in ("create", "deploy"):
        bal = vl.get_balance(client)
        if bal <= 0:
            LOG.error(
                "Solde Verda à %.2f. Crédite le compte avant de déployer "
                "(le build V100 coûte ~0.06/h, mais il faut un solde > 0).", bal
            )
            return 3

    dispatch = {
        "create": phase_create,
        "deploy": phase_deploy,
        "teardown": phase_teardown,
        "status": phase_status,
    }
    try:
        dispatch[args.phase](client)
    except vl.RodinInfraError as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

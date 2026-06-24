#!/usr/bin/env python3
# ============================================================================
# RODIN — 04_pretest_b200.py
# ----------------------------------------------------------------------------
# Déploie une B200 spot pour le PRETEST MANUEL (mesure MFU), PUIS teardown.
# Différent du watchdog : ICI on NE branche PAS le startup script -> l'instance
# boote et ATTEND. Tu te connectes en SSH, tu montes /data toi-même, tu lances
# le trainer À LA MAIN avec un petit --max-steps, tu regardes les chiffres, tu
# tues, tu détruis. Contrôle total, aucun automatisme qui lancerait le vrai run.
#
# Le golden volume est rattaché (existing_volumes) donc blend + venv + scripts
# sont déjà là. Région = celle du golden volume (lue depuis l'API).
#
# PHASES :
#   --phase deploy    : déploie la B200 (sans startup script), affiche IP + memo.
#   --phase teardown  : détache golden volume + supprime l'instance de pretest.
#   --phase status    : état courant (lecture seule).
#
# Usage (sur le VPS) :
#   set -a; source .env; set +a
#   python 04_pretest_b200.py --phase deploy
#   # ... pretest manuel via SSH (voir memo affiché) ...
#   python 04_pretest_b200.py --phase teardown
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import verda_lib as vl

HERE = Path(__file__).resolve().parent
GOLDEN_STATE = HERE / "golden_volume.json"
STATE_FILE = HERE / "pretest_b200.json"

HOSTNAME = "rodin-pretest"
GPU_HINTS = "B200"
OS_VOLUME_SIZE_GB = 50
SSH_KEY_VPS = "<SSH_KEY_VERDA>"

LOG = vl.get_logger("pretest_b200")


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_golden(client) -> tuple[str, str]:
    g = _load(GOLDEN_STATE)
    gid = g.get("golden_volume_id")
    if not gid:
        raise vl.RodinInfraError(f"golden_volume_id absent de {GOLDEN_STATE}.")
    if not vl.volume_exists(client, gid):
        raise vl.RodinInfraError(f"Golden volume {gid} n'existe plus côté Verda.")
    vol = vl.get_volume(client, gid)
    loc = (getattr(vol, "location", None)
           or getattr(vol, "location_code", None)
           or getattr(vol, "region", None))
    if not loc:
        raise vl.RodinInfraError("Région du golden volume illisible.")
    return gid, loc


def phase_deploy(client) -> int:
    golden_id, location = read_golden(client)
    ssh_ids = vl.get_ssh_key_ids(client)

    bal = vl.get_balance(client)
    LOG.info("Solde : %.2f€. Déploiement B200 pretest en %s (golden=%s)…",
             bal, location, golden_id)

    res = vl.deploy_cheapest_available(
        client,
        model_hints=GPU_HINTS,
        hostname=HOSTNAME,
        ssh_key_ids=ssh_ids,
        image=vl.DEFAULT_IMAGE,                 # ubuntu-24.04-cuda-12.8-open-docker
        spot=True,
        os_volume_size_gb=OS_VOLUME_SIZE_GB,
        existing_volume_ids=[golden_id],
        description="rodin-pretest",
        startup_script_id=None,                 # ← PAS de startup : l'instance attend
        locations=(location,),
        wait_running=True,
    )
    _save(STATE_FILE, {
        "instance_id": res.instance_id,
        "ip": res.ip,
        "instance_type": res.instance_type,
        "location": location,
        "golden_volume_id": golden_id,
    })

    ip = res.ip
    print("\n" + "=" * 70)
    print(f"  B200 PRETEST RUNNING — {res.instance_type} en {location}")
    print(f"  IP : {ip}")
    print("=" * 70)
    print(f"""
MÉMO PRETEST (copie-colle, dans l'ordre) :

1) Se connecter :
   ssh -i {SSH_KEY_VPS} root@{ip}

2) Sur l'instance — monter le golden volume :
   mkdir -p /data && mount /dev/vdb /data && df -h /data
   ls -lh /data/scripts/        # bootstrap.sh, push_checkpoints.sh, 21_train_rodin.py

3) Activer le venv + vérifier le GPU :
   source /data/.venv/bin/activate
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

4) LANCER LE TRAINER EN PRETEST (court, ~200 steps, preset prod) :
   cd /data/scripts
   python -u 21_train_rodin.py \\
       --train /data/blend/train.bin \\
       --val   /data/blend/val.bin \\
       --out   /data/runs/PRETEST \\
       --preset prod \\
       --max-steps 200 \\
       --batch-size 24 --grad-accum 2 \\
       --ckpt-every 100 --keep 2 \\
       --eval-every 100 --log-every 10 \\
       --num-workers 8 --prefetch-factor 4 \\
       --compile-mode default

5) DEUXIÈME SSH (autre terminal) pour voir le GPU en live :
   ssh -i {SSH_KEY_VPS} root@{ip}
   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv'

   -> regarde : tok/s dans le log du trainer, memory.used (marge avant 180Go ?),
      utilization.gpu (99% = saturé, <90% = dataloader à la traîne).

6) Quand tu as les chiffres : Ctrl-C le trainer, puis sur le VPS :
   python 04_pretest_b200.py --phase teardown

NB : le dossier /data/runs/PRETEST est jetable (ckpt de test). Le vrai run
     utilisera /data/runs/rodin1b_32b. Tu peux rm -rf /data/runs/PRETEST après.
""")
    return 0


def phase_teardown(client) -> int:
    st = _load(STATE_FILE)
    iid = st.get("instance_id")
    if not iid:
        print("Aucune instance de pretest suivie.")
        return 0
    LOG.info("Teardown instance pretest %s…", iid)
    try:
        # SHUTDOWN propre d'abord (évite le warning au détache), puis DELETE.
        try:
            client.instances.action(iid, getattr(client.actions, "SHUTDOWN", "shutdown"))
            LOG.info("Shutdown demandé.")
        except Exception as exc:
            LOG.warning("Shutdown ignoré (%s), on passe au delete.", exc)
        vl.delete_instance(client, iid)
    except vl.RodinInfraError as exc:
        LOG.error("Échec teardown : %s", exc)
        return 1
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("✅ Instance pretest supprimée. Golden volume préservé (detached).")
    return 0


def phase_status(client) -> int:
    st = _load(STATE_FILE)
    if not st:
        print("Aucune instance de pretest suivie.")
    else:
        print(json.dumps(st, indent=2, ensure_ascii=False))
        iid = st.get("instance_id")
        if iid:
            alive = vl.instance_is_alive(client, iid)
            print(f"Vivante : {'OUI' if alive else 'NON'}")
    print(f"\nSolde : {vl.get_balance(client):.2f}€")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretest B200 manuel (MFU).")
    ap.add_argument("--phase", required=True, choices=["deploy", "teardown", "status"])
    args = ap.parse_args()
    try:
        client = vl.build_client()
    except vl.RodinInfraError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 2
    if args.phase == "deploy":
        return phase_deploy(client)
    if args.phase == "teardown":
        return phase_teardown(client)
    if args.phase == "status":
        return phase_status(client)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

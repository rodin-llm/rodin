#!/usr/bin/env python3
# ============================================================================
# RODIN — 03_watchdog.py
# ----------------------------------------------------------------------------
# LE CERVEAU DE DÉCISION. Vit sur le VPS (immortel). Boucle de polling qui :
#
#   1. Vérifie que l'instance B200 du run est VIVANTE (instance_is_alive).
#        - vivante  -> ne rien faire, dormir.
#        - morte    -> kick spot : REDÉPLOYER une B200 (deploy_cheapest_available
#                      épinglé B200 + région du golden volume + golden volume
#                      rattaché + startup_script_id). Retry patient si la capa
#                      B200 est indisponible (elle va et vient).
#
#   2. Surveille le SOLDE (garde anti-zéro — piège Verda : solde=0 => golden
#      volume DÉTRUIT). Politique (décidée) :
#        - solde < SOFT (20€) ET instance vivante (run en cours) -> ALERTE logs,
#          on ne touche à RIEN (couper un run qui tourne serait absurde).
#        - solde < HARD (10€) -> plancher dur : on stoppe proprement (on ne
#          redéploie plus, on alerte fort). Anti-destruction du golden volume.
#        - avant un REDÉPLOIEMENT : si solde < coût d'amorçage raisonnable,
#          on n'allume pas une B200 à 1.8€/h — on attend une recharge.
#
# CE QU'IL NE FAIT PAS (volontairement, pour rester increvable) :
#   - pas de SSH fragile dans l'instance. API + polling uniquement.
#   - pas de décision sur les checkpoints (c'est l'instance qui gère).
#   - le resume est AUTOMATIQUE côté trainer : redéployer = relancer la même
#     instance (startup script -> bootstrap.sh -> trainer détecte le dernier
#     ckpt et reprend à l'offset exact). Le watchdog ne fait QUE ressusciter.
#
# ÉTAT PERSISTANT : current_instance.json (id + ip de l'instance courante).
#   -> permet de se reconnecter (ssh) à l'instance même après un redeploy,
#      et de reprendre le suivi après un reboot du VPS.
#
# COMMANDES :
#   python 03_watchdog.py run         # la boucle (Ctrl-C pour arrêter)
#   python 03_watchdog.py status      # IP courante + commande ssh prête
#   python 03_watchdog.py once        # un seul tour de boucle (debug)
#   python 03_watchdog.py deploy      # force un (re)déploiement maintenant
#   python 03_watchdog.py forget      # oublie l'instance suivie (n'en supprime
#                                      # AUCUNE côté Verda ; nettoie l'état local)
#
# NB : le watchdog NE LANCE PAS le run depuis zéro tout seul. Le PREMIER
#      déploiement se fait explicitement (commande 'deploy') une fois que tu es
#      prêt. Ensuite 'run' surveille et ressuscite en cas de kick.
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import verda_lib as vl


# ============================================================================
# CONFIG
# ============================================================================
HERE = Path(__file__).resolve().parent

# Fichiers d'état produits par les étapes précédentes.
GOLDEN_STATE = HERE / "golden_volume.json"        # produit par 01_build (id + region)
STARTUP_STATE = HERE / "startup_script.json"      # produit par 02_setup (id du startup)
INSTANCE_STATE = HERE / "current_instance.json"   # produit par CE watchdog

# Run.
RUN_HOSTNAME = "rodin-run"
GPU_HINTS = "B200"                 # épinglé B200 (décidé)
OS_VOLUME_SIZE_GB = 50             # volume OS jetable (le blend est sur le golden)

# Garde solde (€). Politique décidée pour le projet.
SOFT_BALANCE_EUR = 20.0            # alerte logs, on ne touche à rien si run en cours
HARD_BALANCE_EUR = 10.0           # plancher dur : on stoppe proprement
# Avant un redéploiement, refuser d'allumer une B200 si le solde ne couvre pas
# au moins quelques heures (sécurité : ne pas cramer le golden volume en run sec).
MIN_BALANCE_TO_DEPLOY_EUR = 15.0

# Polling.
POLL_INTERVAL = 45                 # secondes entre deux vérifications
CAPACITY_RETRY_INTERVAL = 120      # si B200 indispo, attendre avant de réessayer
DEPLOY_MAX_WAIT = 1800             # max d'attente qu'une instance passe RUNNING
# Auth Verda : si l'API est en rideau au démarrage (incident provider, 522,
# réponse vide -> JSONDecodeError dans le SDK), on NE meurt PAS. On retente
# patiemment jusqu'à ce que l'API revienne. Sans ça, un incident provider au
# moment du boot tue le watchdog en boucle (crash-loop systemd).
AUTH_RETRY_INTERVAL = 60           # secondes entre deux tentatives d'auth au boot

# SSH (pour la commande affichée par 'status').
SSH_KEY_VPS = "<SSH_KEY_VERDA>"
INSTANCE_LOG = "/data/bootstrap.log"   # le trainer tee son loss ici (persistant)

# --- Garde-fou GPU + détection fin de run (SSH non-interactif VPS -> instance) ---
RUN_OUT_DIR = "/data/runs/rodin1b_32b"        # --out du trainer (doit matcher bootstrap.sh)
DONE_MARKER = f"{RUN_OUT_DIR}/DONE"           # écrit par le trainer à max_steps
GPU_IDLE_THRESHOLD = 10                        # % util en-dessous duquel on considère "inactif"
IDLE_TICKS_BEFORE_REDEPLOY = 4                 # tours idle CONSÉCUTIFS (sans DONE) avant de juger "planté"
                                               #   -> 4 * 45s = ~3 min de GPU à plat = bootstrap mort
SSH_TIMEOUT = 15                               # s : un SSH plus long = réseau/instance HS, on ne conclut RIEN
SSH_OPTS = [
    "-i", SSH_KEY_VPS,
    "-o", "BatchMode=yes",                     # jamais de prompt interactif
    # Instance JETABLE : l'IP change à chaque redeploy ET le pool spot recycle
    # les IP -> on retombe sur une IP déjà connue avec une NOUVELLE clé hôte.
    # accept-new NE protège PAS de ce cas (conflit de clé) -> SSH échoue ->
    # watchdog aveugle (sonde GPU/DONE KO). La vérif host key n'a aucune valeur
    # de sécurité sur une cible jetable recyclée : on la neutralise pour CE host.
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",      # ne jamais persister/comparer de clé hôte
    "-o", f"ConnectTimeout={SSH_TIMEOUT}",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2",
]

# --- Notif Telegram (push sortant uniquement, pas de commandes entrantes) ---
# Lit RODIN_TG_TOKEN / RODIN_TG_CHAT dans l'ENV DU SERVICE systemd (voir
# `systemctl edit rodin-watchdog`). Best-effort : si absent -> notif désactivée.
TG_REPORT_EVERY_H = 1.0          # rapport toutes les heures (calé sur la facturation)
MAX_STEPS_HINT = 478000          # pour le % d'avancement / ETA (doit matcher bootstrap.sh)

# --- Storage box Hetzner (pour vérif cohérence golden<->box dans le rapport) ---
# Mêmes paramètres que push_checkpoints.sh. La clé box est sur le VPS.
BOX_USER = "<STORAGE_BOX_USER>"
BOX_HOST = "<STORAGE_BOX_HOST>"
BOX_PORT = "23"
BOX_KEY = "<SSH_KEY_BOX>"          # sur le VPS (pas /data/, ça c'est l'instance)
BOX_DEST = "checkpoints"
BOX_QUOTA_GB = 1024.0                      # 1 To
BOX_SSH_OPTS = [
    "-p", BOX_PORT, "-i", BOX_KEY,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"ConnectTimeout={SSH_TIMEOUT}",
]

# --- Seuils d'alertes critiques (push immédiat, avec anti-rebond) ---
ALERT_BALANCE_EUR = 30.0          # alerte solde bas (au-dessus du soft, pour anticiper)
ALERT_BOX_PCT = 85.0              # alerte si le box dépasse ce % de remplissage
KICK_BURST_WINDOW_S = 1800        # fenêtre (30 min) pour compter les kicks rapprochés
KICK_BURST_COUNT = 3              # nb de kicks dans la fenêtre déclenchant l'alerte "capa tendue"

LOG = vl.get_logger("watchdog")


# ============================================================================
# État persistant
# ============================================================================
def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("%s illisible.", path)
    return {}


def _save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_instance_state() -> dict:
    return _load_json(INSTANCE_STATE)


def save_instance_state(instance_id: str, ip: str, instance_type: str, location: str) -> None:
    _save_json(INSTANCE_STATE, {
        "instance_id": instance_id,
        "ip": ip,
        "instance_type": instance_type,
        "location": location,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    LOG.info("État instance sauvegardé : id=%s ip=%s", instance_id, ip)


def clear_instance_state() -> None:
    if INSTANCE_STATE.exists():
        INSTANCE_STATE.unlink()
        LOG.info("État instance local effacé.")


# ============================================================================
# Lecture des prérequis (golden volume + startup script)
# ============================================================================
def read_golden(client) -> tuple[str, str]:
    """
    Renvoie (golden_volume_id, location).

    L'id vient du fichier d'état (écrit par 01_build). La RÉGION, elle, est lue
    DYNAMIQUEMENT depuis l'API Verda (comme le fait 01_build) : c'est la seule
    source de vérité, pas de risque de désynchro avec un champ figé dans le JSON.
    existing_volumes exige que l'instance soit dans la MÊME région que le volume.
    """
    g = _load_json(GOLDEN_STATE)
    gid = g.get("golden_volume_id")
    if not gid:
        raise vl.RodinInfraError(
            f"golden_volume_id introuvable dans {GOLDEN_STATE}. "
            "Lance d'abord 01_build_golden_volume.py."
        )

    if not vl.volume_exists(client, gid):
        raise vl.RodinInfraError(
            f"Le golden volume {gid} n'existe PLUS côté Verda. "
            "A-t-il été supprimé (solde tombé à 0 ?). Restauration sous 96h "
            "possible, ou rebuild depuis le PC."
        )

    vol = vl.get_volume(client, gid)
    loc = (
        getattr(vol, "location", None)
        or getattr(vol, "location_code", None)
        or getattr(vol, "region", None)
    )
    if not loc:
        raise vl.RodinInfraError(
            f"Région du golden volume {gid} illisible depuis l'API. "
            "Le golden volume et l'instance DOIVENT être dans la même région."
        )
    return gid, loc


def read_startup_id() -> str | None:
    s = _load_json(STARTUP_STATE)
    sid = s.get("id")
    if not sid:
        LOG.warning(
            "Aucun startup_script_id dans %s. L'instance bootera SANS startup "
            "script (le trainer ne démarrera pas tout seul). Lance "
            "02_setup_startup_script.py --phase ensure.", STARTUP_STATE,
        )
    return sid


# ============================================================================
# Solde
# ============================================================================
def check_balance(client) -> tuple[float, str]:
    """
    Renvoie (solde, niveau) où niveau ∈ {"ok", "soft", "hard"}.
    """
    bal = vl.get_balance(client)
    if bal < HARD_BALANCE_EUR:
        return bal, "hard"
    if bal < SOFT_BALANCE_EUR:
        return bal, "soft"
    return bal, "ok"


# ============================================================================
# Déploiement / redéploiement
# ============================================================================
def deploy_run_instance(client, *, reason: str) -> vl.DeployResult:
    """
    Déploie l'instance de run B200 : golden volume rattaché, startup script
    branché, épinglé à la région du golden volume (existing_volumes exige la
    même région). Retry géré par deploy_cheapest_available (capa fluctuante).
    """
    golden_id, location = read_golden(client)
    startup_id = read_startup_id()
    ssh_ids = vl.get_ssh_key_ids(client)

    LOG.info("=== DÉPLOIEMENT (%s) — B200 en %s, golden=%s ===",
             reason, location, golden_id)

    res = vl.deploy_cheapest_available(
        client,
        model_hints=GPU_HINTS,
        hostname=RUN_HOSTNAME,
        ssh_key_ids=ssh_ids,
        image=vl.DEFAULT_IMAGE,           # ubuntu-24.04-cuda-12.8-open-docker (Blackwell)
        spot=True,
        os_volume_size_gb=OS_VOLUME_SIZE_GB,
        existing_volume_ids=[golden_id],
        description="rodin-run",
        startup_script_id=startup_id,
        locations=(location,),            # MÊME région que le golden volume (obligatoire)
        wait_running=True,
        max_wait=DEPLOY_MAX_WAIT,
    )
    save_instance_state(res.instance_id, res.ip, res.instance_type, location)
    LOG.info("Instance de run RUNNING : %s (%s) ip=%s",
             res.instance_id, res.instance_type, res.ip)
    _print_ssh_hint(res.ip)
    return res


def _print_ssh_hint(ip: str) -> None:
    if not ip:
        return
    LOG.info("Pour suivre le loss en direct :")
    LOG.info("    ssh -i %s root@%s", SSH_KEY_VPS, ip)
    LOG.info("    tail -f %s", INSTANCE_LOG)


# ============================================================================
# Sonde instance via SSH (best-effort) : GPU util + présence marqueur DONE
# ----------------------------------------------------------------------------
# PRINCIPE DE SÛRETÉ : tout échec SSH (réseau, instance en reboot, kick en
# cours) renvoie None / "unknown". On ne conclut JAMAIS "GPU à 0%" sur un SSH
# raté. Un SSH raté retombe sur le check API instance_is_alive du tick.
# ============================================================================
def _ssh(ip: str, remote_cmd: str) -> tuple[bool, str]:
    """
    Exécute une commande sur l'instance via SSH non-interactif.
    Renvoie (ok, sortie_strippée). ok=False = échec transport/timeout/commande.
    Ne lève jamais : un watchdog ne meurt pas sur un SSH capricieux.
    """
    if not ip:
        return False, ""
    cmd = ["ssh", *SSH_OPTS, f"root@{ip}", remote_cmd]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=SSH_TIMEOUT + 5,    # garde-fou au-dessus du ConnectTimeout SSH
        )
    except subprocess.TimeoutExpired:
        LOG.debug("SSH timeout vers %s", ip)
        return False, ""
    except Exception as exc:               # FileNotFoundError (ssh absent), etc.
        LOG.debug("SSH erreur vers %s : %s", ip, exc)
        return False, ""
    if out.returncode != 0:
        LOG.debug("SSH rc=%s vers %s : %s", out.returncode, ip, out.stderr.strip())
        return False, out.stdout.strip()
    return True, out.stdout.strip()


def probe_gpu_util(ip: str) -> int | None:
    """
    Util GPU en % via nvidia-smi. None si on n'a pas pu lire (on ne conclut rien).
    Sur multi-GPU on prendrait le max ; ici 1 seule B200 -> 1 ligne.
    """
    ok, out = _ssh(ip, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    if not ok or not out:
        return None
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            vals.append(int(line))
    if not vals:
        return None
    return max(vals)


def probe_done(ip: str) -> bool | None:
    """
    True si le marqueur DONE existe sur l'instance, False s'il est absent,
    None si on n'a pas pu vérifier (SSH KO -> on ne conclut rien).
    """
    ok, out = _ssh(ip, f"test -f {DONE_MARKER} && echo DONE || echo RUNNING")
    if not ok:
        return None
    if out == "DONE":
        return True
    if out == "RUNNING":
        return False
    return None


def probe_train_progress(ip: str) -> dict | None:
    """
    Lit la DERNIÈRE ligne [step ...] du bootstrap.log de l'instance pour extraire
    step / loss / tok_s. Renvoie un dict ou None (SSH KO / pas de ligne). Format
    attendu (écrit par le trainer) :
      [step 121900/478000] loss 3.2730 | lr 2.60e-04 | 123.9k tok/s | off 3,900,800
    Best-effort, parsing tolérant : si un champ manque, il vaut None.
    """
    ok, out = _ssh(ip, f"grep '\\[step ' {INSTANCE_LOG} 2>/dev/null | tail -n 1")
    if not ok or not out:
        return None
    res = {"step": None, "max_steps": None, "loss": None, "tok_s": None}
    try:
        # step X/Y
        if "[step " in out:
            seg = out.split("[step ", 1)[1].split("]", 1)[0].strip()  # "121900/478000"
            if "/" in seg:
                a, b = seg.split("/", 1)
                res["step"] = int(a.strip())
                res["max_steps"] = int(b.strip())
        # loss
        if "loss " in out:
            res["loss"] = float(out.split("loss ", 1)[1].split("|", 1)[0].strip())
        # tok/s (ex "123.9k tok/s")
        if "tok/s" in out:
            tok = out.split("tok/s", 1)[0].strip().split()[-1]   # "123.9k"
            mult = 1000.0 if tok.lower().endswith("k") else 1.0
            res["tok_s"] = float(tok.lower().rstrip("k")) * mult
    except (ValueError, IndexError):
        pass
    return res


def probe_local_last_ckpt(ip: str) -> int | None:
    """Step du dernier ckpt LOCAL sur le golden volume de l'instance. None si SSH KO."""
    ok, out = _ssh(ip, f"ls -1 {RUN_OUT_DIR}/ckpt_*.pt 2>/dev/null | sort | tail -n1")
    if not ok or not out:
        return None
    return _step_from_ckpt_name(out)


def _step_from_ckpt_name(path: str) -> int | None:
    """checkpoints/ckpt_00216000.pt -> 216000. None si format inattendu."""
    base = path.rsplit("/", 1)[-1]            # ckpt_00216000.pt
    if not base.startswith("ckpt_") or not base.endswith(".pt"):
        return None
    num = base[len("ckpt_"):-len(".pt")]
    try:
        return int(num)
    except ValueError:
        return None


def _box_ssh(remote_cmd: str) -> tuple[bool, str]:
    """SSH best-effort vers le storage box (depuis le VPS, clé box). Ne lève jamais."""
    cmd = ["ssh", *BOX_SSH_OPTS, f"{BOX_USER}@{BOX_HOST}", remote_cmd]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT + 10)
    except Exception as exc:
        LOG.debug("SSH box échec : %s", exc)
        return False, ""
    if out.returncode != 0:
        return False, out.stdout.strip()
    return True, out.stdout.strip()


def probe_box_last_ckpt() -> int | None:
    """Step du dernier ckpt sur le BOX. None si box injoignable."""
    ok, out = _box_ssh(f"ls -1 {BOX_DEST}/ckpt_*.pt 2>/dev/null | sort | tail -n1")
    if not ok or not out:
        return None
    # le box peut renvoyer plusieurs lignes si le tail passe mal : prendre la dernière
    last = out.splitlines()[-1].strip() if out.splitlines() else ""
    return _step_from_ckpt_name(last)


def probe_box_usage_gb() -> float | None:
    """
    Taille occupée par le dossier checkpoints sur le box, en Go. None si KO.
    `du -sb` (octets) puis conversion. Le box tolère du/ls (validé en prod).
    """
    ok, out = _box_ssh(f"du -sb {BOX_DEST} 2>/dev/null")
    if not ok or not out:
        return None
    try:
        first = out.split()[0]
        return int(first) / 1e9
    except (ValueError, IndexError):
        return None


# ============================================================================
# Notif Telegram (push sortant best-effort, jamais bloquant)
# ============================================================================
def notify_telegram(text: str) -> bool:
    token = os.environ.get("RODIN_TG_TOKEN", "").strip()
    chat = os.environ.get("RODIN_TG_CHAT", "").strip()
    if not token or not chat:
        LOG.debug("Telegram non configuré (RODIN_TG_TOKEN/RODIN_TG_CHAT) : notif sautée.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    cmd = [
        "curl", "-sS", "--max-time", "30", "-X", "POST", url,
        "--data-urlencode", f"chat_id={chat}",
        "--data-urlencode", f"text={text}",
        "--data-urlencode", "parse_mode=HTML",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if r.returncode == 0 and '"ok":true' in r.stdout:
            return True
        LOG.warning("Telegram échec : rc=%s %s", r.returncode, r.stdout.strip()[:200])
        return False
    except Exception as exc:
        LOG.warning("Telegram exception (non bloquant) : %s", exc)
        return False


def build_periodic_report(client, instance_id, ip, bal, gpu) -> str:
    """Compose le texte du rapport périodique à partir de l'état courant."""
    prog = probe_train_progress(ip)
    lines = ["<b>RODIN-1B — rapport</b>"]
    if prog and prog.get("step") is not None:
        step = prog["step"]
        mx = prog.get("max_steps") or MAX_STEPS_HINT
        pct = 100.0 * step / mx if mx else 0.0
        lines.append(f"Step : {step:,}/{mx:,} ({pct:.1f}%)")
        if prog.get("loss") is not None:
            lines.append(f"Loss : {prog['loss']:.4f}")
        toks = prog.get("tok_s")
        if toks:
            lines.append(f"Débit : {toks/1e3:.1f}k tok/s")
            remaining = mx - step
            # steps restants * tokens/step / débit. tokens/step = 65536 (handoff).
            eta_s = remaining * 65536 / toks if toks > 0 else 0
            eta_h = eta_s / 3600
            lines.append(f"ETA : ~{eta_h:.1f}h ({remaining:,} steps restants)")
    else:
        lines.append("Progression : non lisible (SSH KO ou pas encore de step).")
    lines.append(f"GPU : {gpu}%" if gpu is not None else "GPU : ? (SSH KO)")
    lines.append(f"Solde Verda : {bal:.2f}€")

    # --- cohérence checkpoints golden <-> box ---
    local_step = probe_local_last_ckpt(ip)
    box_step = probe_box_last_ckpt()
    box_gb = probe_box_usage_gb()
    if local_step is not None and box_step is not None:
        lag = local_step - box_step
        # le box pousse 1 ckpt sur 3 (tous les 3000 steps) -> retard NORMAL jusqu'à
        # ~quelques milliers de steps. Au-delà de ~10000 steps de retard = anormal.
        if lag <= 10000:
            lines.append(f"Ckpt local {local_step:,} | box {box_step:,} "
                         f"(retard {lag:,}, normal)")
        else:
            lines.append(f"⚠️ Ckpt local {local_step:,} | box {box_step:,} "
                         f"(retard {lag:,} ANORMAL — push bloqué ?)")
    elif local_step is not None:
        lines.append(f"Ckpt local {local_step:,} | box : ? (injoignable)")
    if box_gb is not None:
        pct_box = 100.0 * box_gb / BOX_QUOTA_GB
        flag = " ⚠️" if pct_box >= ALERT_BOX_PCT else ""
        lines.append(f"Box : {box_gb:.0f}/{BOX_QUOTA_GB:.0f} Go ({pct_box:.0f}%){flag}")

    lines.append(f"Instance : {instance_id}")
    return "\n".join(lines)


# ============================================================================
# Un tour de boucle
# ============================================================================
def one_tick(client) -> str:
    """
    Exécute une vérification. Renvoie un statut court pour la boucle :
      "alive"          : instance vivante ET GPU actif, rien à faire
      "done"           : run TERMINÉ (marqueur DONE + GPU idle) -> stop redeploy, alerte
      "idle-redeployed": instance vivante mais GPU à plat sans DONE depuis N tours
                         (bootstrap planté) -> redéployée
      "redeployed"     : kick détecté, instance redéployée
      "capacity"       : kick/plantage détecté mais pas de capa B200 (réessayer plus tard)
      "halt-balance"   : solde sous le plancher dur -> arrêt propre demandé
      "no-instance"    : aucune instance suivie (premier déploiement pas fait)
    """
    bal, level = check_balance(client)

    state = load_instance_state()
    instance_id = state.get("instance_id")

    # --- garde solde : plancher dur, indépendant de l'état instance ---
    if level == "hard":
        LOG.error("⛔ SOLDE %.2f€ < plancher dur %.0f€. ARRÊT de la surveillance "
                  "active. NE redéploie plus (anti-destruction golden volume). "
                  "Recharge le compte AVANT que ça touche 0 (golden volume détruit "
                  "à 0€, restaurable 96h seulement).", bal, HARD_BALANCE_EUR)
        return "halt-balance"

    if instance_id is None:
        LOG.info("Aucune instance suivie (solde %.2f€). Lance 'deploy' pour "
                 "démarrer le run quand tu es prêt.", bal)
        return "no-instance"

    # --- instance vivante ? ---
    alive = vl.instance_is_alive(client, instance_id)

    if alive:
        if level == "soft":
            LOG.warning("⚠ Solde %.2f€ < %.0f€ MAIS run en cours : on continue "
                        "(couper serait absurde). Surveille / recharge bientôt.",
                        bal, SOFT_BALANCE_EUR)

        # --- 3e état : vivante MAIS calcule-t-elle vraiment ? ---
        # Sonde best-effort. Tout SSH raté -> on ne conclut rien, on traite comme
        # "vivante, RAS" et on retentera au tour suivant (l'API la voit vivante).
        ip = state.get("ip")
        gpu = probe_gpu_util(ip)            # int | None
        done = probe_done(ip)              # bool | None

        # (a) RUN TERMINÉ : marqueur DONE présent ET GPU retombé.
        #     -> on ARRÊTE de surveiller activement (plus de redeploy) + alerte.
        #     PAS de teardown auto (décision §4.8) : humain sécurise les livrables.
        if done is True and (gpu is None or gpu <= GPU_IDLE_THRESHOLD):
            _reset_idle()
            LOG.warning("🏁 RUN TERMINÉ : marqueur DONE détecté sur %s, GPU=%s%%. "
                        "Instance %s laissée VIVANTE (teardown MANUEL après "
                        "sécurisation des checkpoints/poids). Le watchdog NE "
                        "redéploiera plus.", ip, gpu if gpu is not None else "?",
                        instance_id)
            LOG.warning("   -> Récupère le dernier ckpt complet (réserve 130B) + "
                        "extrais les poids, PUIS teardown manuel de l'instance.")
            return "done"

        # (b) GPU à plat SANS marqueur DONE : bootstrap probablement planté.
        #     On exige N tours CONSÉCUTIFS (anti-faux-positif : ckpt en cours,
        #     eval, hiccup réseau). done is None (SSH KO) ne compte PAS comme idle.
        if gpu is not None and gpu <= GPU_IDLE_THRESHOLD and done is False:
            n = _bump_idle()
            LOG.warning("⚠ Instance %s vivante mais GPU=%s%% (≤%d%%) SANS DONE "
                        "[%d/%d tours idle consécutifs].", instance_id, gpu,
                        GPU_IDLE_THRESHOLD, n, IDLE_TICKS_BEFORE_REDEPLOY)
            if n >= IDLE_TICKS_BEFORE_REDEPLOY:
                LOG.error("💀 GPU à plat depuis %d tours sans fin de run : "
                          "bootstrap/trainer planté. REDÉPLOIEMENT forcé.", n)
                if bal < MIN_BALANCE_TO_DEPLOY_EUR:
                    LOG.error("Solde %.2f€ < %.0f€ : pas de redeploy (run sec). "
                              "Recharge.", bal, MIN_BALANCE_TO_DEPLOY_EUR)
                    return "halt-balance"
                # on oublie l'instance plantée et on en relance une (resume auto
                # depuis le dernier ckpt). On ne tue PAS l'ancienne : si elle est
                # zombie elle sera kick/facturée à part ; priorité = relancer le calcul.
                # Note : delete_instance dispo dans verda_lib si on veut nettoyer.
                try:
                    vl.delete_instance(client, instance_id)
                    LOG.info("Instance plantée %s supprimée avant redeploy.",
                             instance_id)
                except Exception as exc:
                    LOG.warning("Suppression de l'instance plantée %s impossible "
                                "(%s) — on redéploie quand même.", instance_id, exc)
                _reset_idle()
                try:
                    deploy_run_instance(client, reason="redeploy GPU à plat (bootstrap planté)")
                    return "idle-redeployed"
                except vl.CapacityUnavailable as exc:
                    LOG.warning("Pas de capa B200 pour relancer : %s", exc)
                    return "capacity"
            return "alive"

        # (c) GPU actif (ou sonde indisponible) : tout va bien.
        _reset_idle()
        if gpu is not None:
            LOG.info("Instance %s vivante, GPU=%s%%. Solde %.2f€. RAS.",
                     instance_id, gpu, bal)
        else:
            LOG.info("Instance %s vivante (sonde GPU indispo, SSH KO). "
                     "Solde %.2f€. RAS.", instance_id, bal)
        return "alive"

    # --- instance morte : kick spot ---
    LOG.warning("💥 Instance %s NON vivante (kick spot probable). Solde %.2f€.",
                instance_id, bal)
    # notif IMMÉDIATE (avant le redeploy, qui peut prendre du temps ou échouer).
    notify_telegram("💥 <b>B200 kické</b> (spot). Redéploiement en cours…")

    # avant de rallumer une B200 (chère), vérifier qu'on a de quoi
    if bal < MIN_BALANCE_TO_DEPLOY_EUR:
        LOG.error("Solde %.2f€ < %.0f€ : on NE redéploie PAS une B200 maintenant "
                  "(éviter de cramer le solde en run sec). Recharge puis relance.",
                  bal, MIN_BALANCE_TO_DEPLOY_EUR)
        return "halt-balance"

    try:
        deploy_run_instance(client, reason="redeploy après kick")
        return "redeployed"
    except vl.CapacityUnavailable as exc:
        LOG.warning("Pas de capacité B200 pour l'instant : %s\n"
                    "On réessaiera au prochain tour (capa spot fluctuante).", exc)
        return "capacity"
    except vl.RodinInfraError as exc:
        LOG.error("Échec redéploiement (erreur non liée à la capacité) : %s", exc)
        return "capacity"   # on retentera, mais c'est à surveiller


# ============================================================================
# Boucle principale
# ============================================================================
_STOP = False
_IDLE_TICKS = 0          # tours consécutifs "GPU à plat sans DONE" (anti-faux-positif)
_LAST_REPORT_TS = 0.0    # dernier rapport périodique Telegram (epoch s)
_DONE_NOTIFIED = False   # pour n'envoyer la notif "run terminé" qu'UNE fois

# --- état anti-rebond des alertes critiques ---
# Chaque alerte "latch" : envoyée une fois quand la condition devient vraie,
# re-armée seulement quand la condition repasse fausse. Évite le spam.
_ALERT_LATCH = {          # nom_alerte -> bool (True = déjà alerté, en attente de retour normal)
    "balance": False,
    "box": False,
    "loss": False,
}
_KICK_TIMES: list[float] = []     # horodatages epoch des kicks récents (pour burst)
_KICK_BURST_LATCH = False         # anti-rebond de l'alerte "kicks répétés"


def _bump_idle() -> int:
    global _IDLE_TICKS
    _IDLE_TICKS += 1
    return _IDLE_TICKS


def _reset_idle() -> None:
    global _IDLE_TICKS
    _IDLE_TICKS = 0


def _alert_once(key: str, condition: bool, message: str) -> None:
    """
    Alerte 'latch' : envoie le message UNE fois quand condition devient vraie,
    re-arme quand elle redevient fausse. Anti-spam.
    """
    global _ALERT_LATCH
    if condition and not _ALERT_LATCH.get(key, False):
        notify_telegram(message)
        _ALERT_LATCH[key] = True
    elif not condition and _ALERT_LATCH.get(key, False):
        _ALERT_LATCH[key] = False     # condition revenue normale -> ré-armer


def _record_kick() -> None:
    """Enregistre un kick et déclenche l'alerte burst si trop de kicks rapprochés."""
    global _KICK_TIMES, _KICK_BURST_LATCH
    now = time.time()
    _KICK_TIMES.append(now)
    # purge des kicks hors fenêtre
    _KICK_TIMES = [t for t in _KICK_TIMES if now - t <= KICK_BURST_WINDOW_S]
    if len(_KICK_TIMES) >= KICK_BURST_COUNT and not _KICK_BURST_LATCH:
        notify_telegram(
            f"🔁 <b>Capacité B200 tendue</b> : {len(_KICK_TIMES)} kicks en "
            f"{KICK_BURST_WINDOW_S // 60} min. Le run avance par à-coups. "
            "Rien à faire (le watchdog encaisse), mais surveille le solde.")
        _KICK_BURST_LATCH = True
    elif len(_KICK_TIMES) < KICK_BURST_COUNT:
        _KICK_BURST_LATCH = False     # accalmie -> ré-armer


def check_critical_alerts(client, status: str, ip: str | None, bal: float) -> None:
    """
    Évalue les conditions d'alerte critique et notifie (avec anti-rebond).
    Best-effort : toute exception est avalée (ne casse jamais la boucle).
    Appelée à chaque tick.
    """
    try:
        # 1. solde bas (anticipe le soft à 20€ en alertant dès 30€)
        _alert_once(
            "balance", bal < ALERT_BALANCE_EUR,
            f"⚠️ <b>Solde bas</b> : {bal:.2f}€ (< {ALERT_BALANCE_EUR:.0f}€). "
            f"Recharge pour ne pas risquer un blocage (soft {SOFT_BALANCE_EUR:.0f}€, "
            f"hard {HARD_BALANCE_EUR:.0f}€, golden détruit à 0€).")

        # Les sondes box/loss n'ont de sens que si l'instance tourne.
        if status != "alive" or not ip:
            return

        # 2. box qui se remplit (le drame du 1 To)
        box_gb = probe_box_usage_gb()
        if box_gb is not None:
            pct = 100.0 * box_gb / BOX_QUOTA_GB
            _alert_once(
                "box", pct >= ALERT_BOX_PCT,
                f"🗄️ <b>Box {pct:.0f}% plein</b> ({box_gb:.0f}/{BOX_QUOTA_GB:.0f} Go). "
                "Vérifie que la purge SFTP tourne (push_checkpoints.sh).")

        # 3. loss anormale : UNIQUEMENT NaN/inf (vraie divergence dure).
        #    On NE compare PLUS à un minimum absolu : le loss instantané oscille
        #    normalement (bruit de batch 2.4<->2.8), comparer au min vu (un batch
        #    facile isolé) déclenchait des dizaines de faux positifs. Seul NaN/inf
        #    est un signal de divergence fiable et sans faux positif.
        prog = probe_train_progress(ip)
        if prog and prog.get("loss") is not None:
            loss = prog["loss"]
            is_nan = (loss != loss) or (loss == float("inf"))    # NaN: x!=x
            _alert_once(
                "loss", is_nan,
                f"🔥 <b>Loss NaN/inf</b> : {loss}. DIVERGENCE DURE — le training "
                "a explosé. Vérifie le bootstrap.log d'urgence.")
    except Exception as exc:
        LOG.warning("check_critical_alerts (non bloquant) : %s", exc)


def _handle_sigterm(signum, frame):
    global _STOP
    LOG.info("Signal %s reçu : arrêt propre du watchdog après ce tour.", signum)
    _STOP = True


def run_loop(client) -> int:
    global _LAST_REPORT_TS, _DONE_NOTIFIED
    signal.signal(signal.SIGINT, _handle_sigterm)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    LOG.info("=== WATCHDOG RODIN démarré (poll %ss) ===", POLL_INTERVAL)
    LOG.info("Garde solde : soft=%.0f€ (alerte), hard=%.0f€ (stop), "
             "min_deploy=%.0f€.", SOFT_BALANCE_EUR, HARD_BALANCE_EUR,
             MIN_BALANCE_TO_DEPLOY_EUR)
    if os.environ.get("RODIN_TG_TOKEN") and os.environ.get("RODIN_TG_CHAT"):
        notify_telegram("👁️ Watchdog RODIN démarré. Rapport périodique toutes "
                        f"les {TG_REPORT_EVERY_H:.0f}h.")
    else:
        LOG.info("Telegram non configuré : pas de notif (RODIN_TG_TOKEN/CHAT "
                 "absents de l'env du service).")

    while not _STOP:
        try:
            status = one_tick(client)
        except vl.RodinInfraError as exc:
            LOG.error("Erreur pendant le tick : %s", exc)
            status = "capacity"
        except Exception as exc:  # robustesse : un watchdog ne meurt pas sur une exception API
            LOG.exception("Exception inattendue (on continue) : %s", exc)
            status = "capacity"

        # --- notifs d'événements (push sortant, best-effort) ---
        try:
            if status == "redeployed":
                _record_kick()
                notify_telegram("✅ <b>B200 redéployée</b> après kick. "
                                "Reprise auto depuis le dernier checkpoint.")
            elif status == "idle-redeployed":
                notify_telegram("⚠️ GPU à plat sans fin de run (bootstrap planté) "
                                "→ instance redéployée. Reprise auto.")
            elif status == "capacity":
                # kick détecté mais pas de capa pour relancer : on compte le kick
                # (la notif "💥 en cours" est déjà partie depuis one_tick).
                _record_kick()
                notify_telegram("⏳ Kické mais <b>pas de capacité B200</b> dispo. "
                                "Le watchdog réessaie en boucle. Run gelé (0€) "
                                "en attendant qu'une B200 se libère.")
            elif status == "done" and not _DONE_NOTIFIED:
                _DONE_NOTIFIED = True
                st = load_instance_state()
                notify_telegram(
                    "🏁 <b>RUN TERMINÉ</b> (vu par le watchdog).\n"
                    f"Instance {st.get('instance_id')} laissée VIVANTE. "
                    "Vérifie le mail de fin (push box) puis fais le teardown "
                    "MANUEL. Le watchdog ne redéploiera plus.")
            elif status == "halt-balance":
                notify_telegram("⛔ Watchdog : solde sous le plancher. "
                                "Redéploiement suspendu. RECHARGE avant 0€ "
                                "(golden volume détruit à 0€).")
        except Exception as exc:
            LOG.warning("Notif événement échouée (non bloquant) : %s", exc)

        # --- alertes critiques (anti-rebond) : solde / box / loss, à chaque tour ---
        _st = load_instance_state()
        try:
            _bal_now = vl.get_balance(client)
        except Exception:
            _bal_now = SOFT_BALANCE_EUR + 1   # valeur neutre si l'API solde échoue
        check_critical_alerts(client, status, _st.get("ip"), _bal_now)

        # --- rapport périodique (toutes les TG_REPORT_EVERY_H heures) ---
        # Seulement si un run est en cours (pas en veille post-DONE ni sans instance).
        if status == "alive":
            now = time.time()
            if now - _LAST_REPORT_TS >= TG_REPORT_EVERY_H * 3600:
                try:
                    st = load_instance_state()
                    iid = st.get("instance_id")
                    ip = st.get("ip")
                    bal = vl.get_balance(client)
                    gpu = probe_gpu_util(ip)
                    notify_telegram(build_periodic_report(client, iid, ip, bal, gpu))
                    _LAST_REPORT_TS = now
                except Exception as exc:
                    LOG.warning("Rapport périodique échoué (non bloquant) : %s", exc)

        if status == "done":
            LOG.warning("=== RUN TERMINÉ. Le watchdog passe en VEILLE : il "
                        "continue de tourner mais NE redéploiera plus. "
                        "Sécurise les livrables puis teardown manuel + Ctrl-C. ===")
            # veille longue : on ne tue rien, on ne redéploie rien. On reste
            # juste en vie pour que 'status' reste consultable et qu'un éventuel
            # kick de l'instance terminée ne déclenche aucune action.
            _sleep(POLL_INTERVAL * 8)
            continue

        if status == "halt-balance":
            LOG.error("Surveillance active suspendue (solde). Le watchdog "
                      "continue de tourner mais ne redéploie plus. Recharge.")
            # on continue de polit pour reprendre dès que le solde remonte,
            # mais on espace pour ne pas spammer l'API.
            _sleep(POLL_INTERVAL * 4)
            continue

        if status == "capacity":
            _sleep(CAPACITY_RETRY_INTERVAL)
            continue

        _sleep(POLL_INTERVAL)

    LOG.info("Watchdog arrêté proprement.")
    return 0


def _sleep(seconds: int) -> None:
    """Sleep interruptible (réagit vite au SIGINT/SIGTERM)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and not _STOP:
        time.sleep(min(2, end - time.monotonic()))


# ============================================================================
# Commandes one-shot
# ============================================================================
def cmd_status(client) -> int:
    state = load_instance_state()
    print("── État instance suivie ──")
    if not state:
        print("  (aucune instance suivie)")
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))

    bal, level = check_balance(client)
    print(f"\nSolde : {bal:.2f}€  (niveau: {level})")

    instance_id = state.get("instance_id")
    if instance_id:
        alive = vl.instance_is_alive(client, instance_id)
        print(f"Instance {instance_id} vivante : {'OUI' if alive else 'NON'}")
        ip = state.get("ip")
        if ip and alive:
            gpu = probe_gpu_util(ip)
            done = probe_done(ip)
            gpu_s = f"{gpu}%" if gpu is not None else "? (SSH KO)"
            if done is True:
                done_s = "OUI -> RUN TERMINÉ"
            elif done is False:
                done_s = "non (run en cours)"
            else:
                done_s = "? (SSH KO)"
            print(f"  GPU util : {gpu_s}")
            print(f"  Marqueur DONE : {done_s}")
        if ip:
            print("\nSuivre le loss en direct :")
            print(f"  ssh -i {SSH_KEY_VPS} root@{ip}")
            print(f"  tail -f {INSTANCE_LOG}")
    return 0


def cmd_deploy(client) -> int:
    bal, level = check_balance(client)
    if bal < MIN_BALANCE_TO_DEPLOY_EUR:
        LOG.error("Solde %.2f€ < %.0f€ : refus de déployer. Recharge d'abord.",
                  bal, MIN_BALANCE_TO_DEPLOY_EUR)
        return 1
    # avertir si une instance est déjà suivie et vivante
    state = load_instance_state()
    if state.get("instance_id") and vl.instance_is_alive(client, state["instance_id"]):
        LOG.warning("Une instance suivie (%s) est déjà vivante. Déploiement "
                    "annulé (évite les doublons). Utilise 'forget' si tu veux "
                    "repartir.", state["instance_id"])
        return 1
    try:
        deploy_run_instance(client, reason="déploiement manuel")
        return 0
    except vl.CapacityUnavailable as exc:
        LOG.error("Pas de capacité B200 : %s", exc)
        return 2


def cmd_once(client) -> int:
    status = one_tick(client)
    LOG.info("Tick unique -> %s", status)
    return 0


def cmd_forget() -> int:
    clear_instance_state()
    print("Instance oubliée localement (AUCUNE suppression côté Verda).")
    return 0


# ============================================================================
# main
# ============================================================================
def build_client_resilient():
    """
    Construit le client Verda en RÉSISTANT à un incident provider au démarrage.

    Le SDK Verda authentifie dès la construction du client. Si l'API est en
    rideau (incident provider, Cloudflare 522, réponse d'auth vide), le SDK lève
    une exception — souvent un json.JSONDecodeError BRUT (réponse vide ->
    json.loads("")), PAS un vl.RodinInfraError. Si on laisse cette exception
    remonter, le process meurt, systemd relance, ça re-crashe : crash-loop qui
    martèle l'API d'auth et entretient le problème.

    Ici on attrape TOUTE exception, on logge, on notifie Telegram UNE fois (le
    réseau Telegram est indépendant de Verda, donc la notif passe même si Verda
    est down), puis on retente toutes les AUTH_RETRY_INTERVAL secondes jusqu'à
    ce que l'API revienne. Le watchdog démarre alors normalement.

    Respecte _STOP : si on reçoit SIGTERM/SIGINT pendant l'attente, on abandonne
    proprement (retourne None).
    """
    signal.signal(signal.SIGINT, _handle_sigterm)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    notified = False
    attempt = 0
    while not _STOP:
        attempt += 1
        try:
            client = vl.build_client()
            if attempt > 1:
                LOG.info("Auth Verda rétablie après %d tentative(s).", attempt)
                notify_telegram("✅ Auth Verda rétablie. Watchdog opérationnel.")
            return client
        except Exception as exc:
            # JSONDecodeError, RodinInfraError, timeouts réseau, 5xx... : tout.
            LOG.error("Auth Verda KO (tentative %d) : %s : %s — "
                      "nouvelle tentative dans %ds.",
                      attempt, type(exc).__name__, exc, AUTH_RETRY_INTERVAL)
            if not notified:
                notify_telegram(
                    "⚠️ <b>API Verda injoignable</b> au démarrage du watchdog "
                    "(incident provider probable). Le watchdog ATTEND et "
                    "réessaie l'auth en boucle. Le run en cours n'est PAS "
                    "affecté (le calcul ne dépend pas de l'API).")
                notified = True
            _sleep(AUTH_RETRY_INTERVAL)
    LOG.info("Arrêt demandé pendant l'attente d'auth Verda : on abandonne.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Watchdog VPS RODIN (surveille+redéploie B200).")
    ap.add_argument("command", choices=["run", "status", "once", "deploy", "forget"])
    args = ap.parse_args()

    if args.command == "forget":
        return cmd_forget()

    # Pour la commande 'run' (service au long cours), on tolère une API Verda
    # down au boot : retry patient au lieu de mourir (anti crash-loop). Pour les
    # commandes ponctuelles/interactives, on garde l'échec immédiat (inutile de
    # boucler sur un appel manuel).
    if args.command == "run":
        client = build_client_resilient()
        if client is None:        # arrêt demandé pendant l'attente d'auth
            return 0
        return run_loop(client)

    try:
        client = vl.build_client()
    except vl.RodinInfraError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Même en interactif, donner un message clair plutôt qu'une stack brute
        # de JSONDecodeError quand l'API est en rideau.
        print(f"[ERREUR] API Verda injoignable ({type(exc).__name__}: {exc}). "
              f"Réessaie quand l'incident provider est résolu.", file=sys.stderr)
        return 2

    if args.command == "status":
        return cmd_status(client)
    if args.command == "once":
        return cmd_once(client)
    if args.command == "deploy":
        return cmd_deploy(client)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

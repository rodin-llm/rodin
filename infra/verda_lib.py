#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verda_lib.py — Module commun RODIN cloud (étape 3, infra SISR).

Rôle : couche d'abstraction au-dessus du SDK officiel `verda` (ex-datacrunch).
Tout le pilotage de l'infra (build golden volume, watchdog, reprise) s'appuie
sur ce module. Écrit UNE fois, importé partout.

Responsabilités :
  - authentification (client_id / client_secret via variables d'environnement)
  - découverte : SSH keys, instance types (avec prix spot), images, locations
  - cycle de vie instance : deploy (spot, avec golden volume rattaché),
    attente RUNNING, récupération IP, suppression
  - cycle de vie volume : create/get/attach/detach/delete
  - helpers robustes : retries, timeouts, logging propre

Dépendances : verda>=1.24  (pip install verda)

Conventions du projet :
  - "DECIDE" = figé. Ici : B200 spot, golden volume NVMe rattaché via
    existing_volumes, on_spot_discontinue='keep_detached' pour ne JAMAIS
    perdre le volume à l'éviction.
  - Aucun secret en dur. CLIENT_ID / CLIENT_SECRET viennent de l'environnement.

Auteur : RODIN — projet RODIN-1B
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# APIException est ré-exporté volontairement : les scripts appelants peuvent
# l'attraper via vl.APIException sans réimporter le SDK directement.
try:
    from verda import VerdaClient
    from verda.exceptions import APIException
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "[verda_lib] SDK 'verda' introuvable. Installe-le dans le venv :\n"
        "    pip install verda\n"
    )
    raise


# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "rodin", level: int = logging.INFO) -> logging.Logger:
    """Logger console propre, format horodaté, idempotent (pas de double handler)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOG = get_logger("verda_lib")


# ──────────────────────────────────────────────────────────────────────────
# Configuration / constantes projet (DECIDE)
# ──────────────────────────────────────────────────────────────────────────

# Image Verda : Ubuntu 24.04 + CUDA 12.8 (support Blackwell B200, cf. décision torch cu128).
DEFAULT_IMAGE = "ubuntu-24.04-cuda-12.8-open-docker"

# Type d'instance B200 spot (à confirmer via list_instance_types() — le code SDK
# attend une string type, ex. '1B200.xx'. On la résout dynamiquement plutôt que
# de la coder en dur, car la nomenclature peut varier selon la dispo/région).
B200_GPU_MODEL_HINT = "B200"          # sous-chaîne recherchée dans instance_type / description
CHEAP_GPU_MODEL_HINTS = ("A100", "L40S", "V100", "RTX")  # pour l'instance jetable de build

# Politique de conservation du volume OS à l'éviction spot.
# 'keep_detached' = le volume survit, détaché, réutilisable. NE JAMAIS changer.
SPOT_OS_VOLUME_POLICY = "keep_detached"

# Localisation : on reste en Finlande (Helsinki) pour la proximité storage box / VPS.
# location_code est DÉSORMAIS OBLIGATOIRE à la création (API publique) : une requête
# sans location_code reçoit HTTP 400. On essaie les régions FIN dans l'ordre.
PREFERRED_LOCATIONS = ("FIN-01", "FIN-02", "FIN-03")
DEFAULT_LOCATION = "FIN-01"

# Timeouts (secondes)
DEPLOY_POLL_INTERVAL = 15      # fréquence de polling du statut instance
DEPLOY_MAX_WAIT = 1800         # 30 min max d'attente qu'une instance passe RUNNING
ACTION_RETRY_INTERVAL = 10
ACTION_MAX_RETRIES = 6

# Codes/messages d'erreur API qui signalent un manque de CAPACITÉ (≠ erreur de
# config). Quand on les rencontre, on réessaie ou on change de GPU/région, on
# n'abandonne pas. Liste large à dessein : les libellés varient selon l'API.
CAPACITY_ERROR_HINTS = (
    "capacity", "availab", "no_availability", "not available",
    "no capacity", "out of capacity", "insufficient", "sold out",
)


# ──────────────────────────────────────────────────────────────────────────
# Exceptions projet
# ──────────────────────────────────────────────────────────────────────────

class RodinInfraError(RuntimeError):
    """Erreur d'infra non récupérable côté RODIN (config, capacité, etc.)."""


class CapacityUnavailable(RodinInfraError):
    """
    La capacité demandée (GPU spot) n'est pas disponible MAINTENANT.
    Distincte d'une erreur de config : on peut réessayer plus tard, ou changer
    de GPU/région, plutôt que d'abandonner. Le watchdog s'appuie là-dessus.
    """


def _is_capacity_error(exc: Exception) -> bool:
    """
    Heuristique : l'exception API traduit-elle un manque de capacité
    (vs une vraie erreur de config / auth) ? On regarde code + message.
    """
    text = ""
    code = getattr(exc, "code", "") or ""
    msg = getattr(exc, "message", "") or ""
    text = f"{code} {msg} {exc}".lower()
    return any(h in text for h in CAPACITY_ERROR_HINTS)


class DeployTimeout(RodinInfraError):
    """L'instance n'a pas atteint l'état RUNNING dans le délai imparti."""


# ──────────────────────────────────────────────────────────────────────────
# Authentification
# ──────────────────────────────────────────────────────────────────────────

def build_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> VerdaClient:
    """
    Crée un client Verda authentifié.

    Les credentials sont lus dans cet ordre de priorité :
      1. arguments explicites client_id / client_secret
      2. variables d'environnement VERDA_CLIENT_ID / VERDA_CLIENT_SECRET

    Lève RodinInfraError si les credentials sont absents.
    """
    cid = client_id or os.environ.get("VERDA_CLIENT_ID")
    csec = client_secret or os.environ.get("VERDA_CLIENT_SECRET")
    if not cid or not csec:
        raise RodinInfraError(
            "Credentials Verda manquants. Exporte VERDA_CLIENT_ID et "
            "VERDA_CLIENT_SECRET, ou passe-les à build_client()."
        )
    try:
        client = VerdaClient(cid, csec)
    except APIException as exc:
        raise RodinInfraError(f"Échec authentification Verda : {exc}") from exc
    LOG.info("Client Verda authentifié.")
    return client


# ──────────────────────────────────────────────────────────────────────────
# Découverte : SSH keys, instance types, images, balance
# ──────────────────────────────────────────────────────────────────────────

def get_ssh_key_ids(client: VerdaClient) -> list[str]:
    """Retourne la liste des IDs de toutes les clés SSH du compte."""
    keys = client.ssh_keys.get()
    ids = [k.id for k in keys]
    if not ids:
        raise RodinInfraError(
            "Aucune clé SSH enregistrée sur le compte Verda. Ajoute ta clé "
            "publique dans la console Verda (ou via client.ssh_keys.create()) "
            "avant de déployer — sinon tu ne pourras pas SSH dans l'instance."
        )
    LOG.info("%d clé(s) SSH trouvée(s).", len(ids))
    return ids


def get_balance(client: VerdaClient) -> float:
    """Solde du compte (devise du compte). Sert au garde-fou budget."""
    bal = client.balance.get()
    return float(bal.amount)


@dataclass
class InstanceTypeInfo:
    """Vue simplifiée d'un type d'instance, avec prix spot."""
    instance_type: str
    description: str
    price_per_hour: float
    spot_price_per_hour: float
    gpu_model: str = ""
    gpu_count: int = 0
    gpu_vram_gb: int = 0

    @property
    def has_spot(self) -> bool:
        return self.spot_price_per_hour is not None and self.spot_price_per_hour > 0


def list_instance_types(client: VerdaClient) -> list[InstanceTypeInfo]:
    """
    Liste tous les types d'instance disponibles, normalisés en InstanceTypeInfo.
    Le SDK expose gpu/gpu_memory comme dicts ; on les aplatit prudemment.
    """
    raw = client.instance_types.get()
    out: list[InstanceTypeInfo] = []
    for it in raw:
        gpu = getattr(it, "gpu", {}) or {}
        gpu_mem = getattr(it, "gpu_memory", {}) or {}
        # Les clés exactes peuvent varier ; on tente plusieurs noms usuels.
        gpu_model = (
            gpu.get("description")
            or gpu.get("name")
            or gpu.get("model")
            or ""
        )
        gpu_count = int(gpu.get("number_of_gpus") or gpu.get("count") or 0)
        vram = 0
        # gpu_memory peut être {'size_in_gigabytes': X} ou {'description': '180GB'}
        if isinstance(gpu_mem, dict):
            vram = int(
                gpu_mem.get("size_in_gigabytes")
                or gpu_mem.get("size")
                or 0
            )
        out.append(
            InstanceTypeInfo(
                instance_type=it.instance_type,
                description=getattr(it, "description", "") or "",
                price_per_hour=float(getattr(it, "price_per_hour", 0) or 0),
                spot_price_per_hour=float(getattr(it, "spot_price_per_hour", 0) or 0),
                gpu_model=gpu_model,
                gpu_count=gpu_count,
                gpu_vram_gb=vram,
            )
        )
    return out


def find_instance_type(
    client: VerdaClient,
    model_hints: tuple[str, ...] | str,
    *,
    require_spot: bool = True,
    single_gpu: bool = True,
    cheapest: bool = True,
) -> InstanceTypeInfo:
    """
    Trouve un type d'instance dont le GPU matche un des `model_hints`
    (recherche insensible à la casse dans instance_type + description + gpu_model).

    - require_spot : ne garde que les types ayant un prix spot.
    - single_gpu   : privilégie 1 GPU (pour un run 1B mono-GPU).
    - cheapest     : trie par prix spot croissant et renvoie le moins cher.

    Lève RodinInfraError si rien ne matche.
    """
    if isinstance(model_hints, str):
        model_hints = (model_hints,)
    hints = tuple(h.lower() for h in model_hints)

    candidates: list[InstanceTypeInfo] = []
    for it in list_instance_types(client):
        haystack = f"{it.instance_type} {it.description} {it.gpu_model}".lower()
        if not any(h in haystack for h in hints):
            continue
        if require_spot and not it.has_spot:
            continue
        if single_gpu and it.gpu_count and it.gpu_count != 1:
            # on tolère gpu_count==0 (info absente), mais on écarte le multi-GPU explicite
            continue
        candidates.append(it)

    if not candidates:
        raise RodinInfraError(
            f"Aucun type d'instance ne matche {model_hints!r} "
            f"(require_spot={require_spot}, single_gpu={single_gpu}). "
            "Vérifie la disponibilité dans la console Verda."
        )

    key = (lambda it: it.spot_price_per_hour) if require_spot else (lambda it: it.price_per_hour)
    candidates.sort(key=key)
    chosen = candidates[0] if cheapest else candidates[-1]
    LOG.info(
        "Type d'instance retenu : %s (%s, %d GPU, %d Go VRAM) — spot %.3f/h",
        chosen.instance_type, chosen.gpu_model or "?", chosen.gpu_count,
        chosen.gpu_vram_gb, chosen.spot_price_per_hour,
    )
    return chosen


def find_candidates(
    client: VerdaClient,
    model_hints: tuple[str, ...] | str,
    *,
    require_spot: bool = True,
    single_gpu: bool = True,
    max_price: Optional[float] = None,
) -> list[InstanceTypeInfo]:
    """
    Comme find_instance_type, mais renvoie TOUS les candidats triés par prix
    spot croissant (le moins cher d'abord). Sert au déploiement avec fallback :
    on tente le moins cher, et si la capacité manque, on passe au suivant.

    max_price : plafond optionnel sur le prix spot/h (filtre les trop chers).
    """
    if isinstance(model_hints, str):
        model_hints = (model_hints,)
    hints = tuple(h.lower() for h in model_hints)

    candidates: list[InstanceTypeInfo] = []
    for it in list_instance_types(client):
        haystack = f"{it.instance_type} {it.description} {it.gpu_model}".lower()
        if not any(h in haystack for h in hints):
            continue
        if require_spot and not it.has_spot:
            continue
        if single_gpu and it.gpu_count and it.gpu_count != 1:
            continue
        price = it.spot_price_per_hour if require_spot else it.price_per_hour
        if max_price is not None and price > max_price:
            continue
        candidates.append(it)

    candidates.sort(
        key=(lambda it: it.spot_price_per_hour) if require_spot
        else (lambda it: it.price_per_hour)
    )
    return candidates


# ──────────────────────────────────────────────────────────────────────────
# Volumes
# ──────────────────────────────────────────────────────────────────────────

def create_detached_nvme_volume(
    client: VerdaClient,
    name: str,
    size_gb: int,
    location_code: str = DEFAULT_LOCATION,
) -> str:
    """
    Crée un volume NVMe DÉTACHÉ (sans instance_id) et renvoie son ID.
    C'est ainsi qu'on fabrique le golden volume : il existe seul, on l'attache
    ensuite à chaque instance via existing_volumes au déploiement.

    location_code est OBLIGATOIRE (API publique récente). On le passe ; selon la
    version du SDK le kwarg s'appelle 'location_code' ou 'location'.
    """
    NVMe = client.constants.volume_types.NVMe
    vol = None
    last_exc: Optional[Exception] = None
    for loc_kw in ("location_code", "location"):
        try:
            vol = client.volumes.create(
                type=NVMe, name=name, size=size_gb, **{loc_kw: location_code}
            )
            break
        except TypeError as exc:
            last_exc = exc
            continue
        except APIException as exc:
            raise RodinInfraError(f"Échec création volume : {exc}") from exc
    if vol is None:
        # dernier essai sans location (vieilles versions SDK qui défaultent FIN-01)
        try:
            vol = client.volumes.create(type=NVMe, name=name, size=size_gb)
        except APIException as exc:
            raise RodinInfraError(
                f"Échec création volume (location_code requis ?) : {exc} / {last_exc}"
            ) from exc
    LOG.info("Volume NVMe détaché créé : %s (%d Go) id=%s loc=%s",
             name, size_gb, vol.id, location_code)
    return vol.id


def get_volume(client: VerdaClient, volume_id: str):
    """Récupère un volume par ID (objet SDK)."""
    return client.volumes.get_by_id(volume_id)


def volume_exists(client: VerdaClient, volume_id: str) -> bool:
    """True si le volume existe encore (n'a pas été supprimé)."""
    try:
        v = client.volumes.get_by_id(volume_id)
        return v is not None
    except APIException:
        return False


def detach_volumes(client: VerdaClient, volume_ids: list[str]) -> None:
    """Détache un ou plusieurs volumes de leur instance (les volumes survivent)."""
    if not volume_ids:
        return
    client.volumes.detach(volume_ids)
    LOG.info("Volume(s) détaché(s) : %s", ", ".join(volume_ids))


def delete_volumes(client: VerdaClient, volume_ids: list[str]) -> None:
    """SUPPRIME définitivement un ou plusieurs volumes. À utiliser avec prudence."""
    if not volume_ids:
        return
    client.volumes.delete(volume_ids)
    LOG.warning("Volume(s) SUPPRIMÉ(S) définitivement : %s", ", ".join(volume_ids))


# ──────────────────────────────────────────────────────────────────────────
# Instances : déploiement, attente, IP, suppression
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DeployResult:
    instance_id: str
    ip: str
    instance_type: str
    spot: bool
    attached_volumes: list[str] = field(default_factory=list)


def _instance_status(client: VerdaClient, instance_id: str) -> str:
    """Statut courant de l'instance (string), robuste aux erreurs transitoires."""
    inst = client.instances.get_by_id(instance_id)
    return getattr(inst, "status", "") or ""


def _instance_ip(client: VerdaClient, instance_id: str) -> str:
    """IP publique de l'instance, '' si pas encore attribuée."""
    inst = client.instances.get_by_id(instance_id)
    return getattr(inst, "ip", "") or ""


def deploy_instance(
    client: VerdaClient,
    *,
    instance_type: str,
    hostname: str,
    ssh_key_ids: list[str],
    image: str = DEFAULT_IMAGE,
    spot: bool = True,
    os_volume_size_gb: int = 50,
    existing_volume_ids: Optional[list[str]] = None,
    description: str = "rodin",
    startup_script_id: Optional[str] = None,
    location_code: str = DEFAULT_LOCATION,
    wait_running: bool = True,
    max_wait: int = DEPLOY_MAX_WAIT,
) -> DeployResult:
    """
    Déploie une instance (spot par défaut) avec le golden volume rattaché.

    Paramètres clés :
      - instance_type      : ex. type B200 résolu via find_instance_type().
      - existing_volume_ids: IDs de volumes détachés à rattacher au boot
                             (= [golden_volume_id]). C'est CE mécanisme qui
                             rend la reprise instantanée : le blend est déjà là.
      - spot               : True = tarif réduit, éviction possible sans préavis.
      - os_volume_size_gb  : taille du volume OS (jetable). Le blend n'est PAS
                             ici, il est sur le golden volume rattaché.
      - startup_script_id  : ID d'un startup script Verda (bootstrap autonome).
                             Optionnel ici ; le bootstrap peut aussi être géré
                             par systemd dans l'image. Voir étape 3d.

    Construit le kwargs de manière défensive : on ne passe que les options
    supportées, et on tente plusieurs noms pour le flag spot et la politique
    de conservation du volume (la nomenclature a évolué selon les versions SDK).
    """
    existing_volume_ids = existing_volume_ids or []

    os_volume = {
        "name": f"{hostname}-os",
        "size": os_volume_size_gb,
        # à l'éviction, on garde le volume OS détaché (cohérent avec golden vol)
        "on_spot_discontinue": SPOT_OS_VOLUME_POLICY,
    }

    create_kwargs: dict[str, Any] = dict(
        instance_type=instance_type,
        image=image,
        ssh_key_ids=ssh_key_ids,
        hostname=hostname,
        description=description,
        os_volume=os_volume,
    )

    # location_code OBLIGATOIRE (API publique récente, sinon HTTP 400).
    # Selon la version SDK le kwarg est 'location_code' ou 'location' ; on règle
    # ça dans la boucle de création ci-dessous.

    # Volumes existants à rattacher (golden volume). Le SDK attend 'existing_volumes'.
    if existing_volume_ids:
        create_kwargs["existing_volumes"] = existing_volume_ids

    # Flag spot : selon la version SDK le kwarg s'appelle 'is_spot' ou 'spot'.
    # On tente is_spot d'abord (API REST l'expose ainsi), fallback sur spot.
    if startup_script_id:
        create_kwargs["startup_script_id"] = startup_script_id

    LOG.info(
        "Déploiement instance type=%s hostname=%s spot=%s loc=%s volumes_rattachés=%s",
        instance_type, hostname, spot, location_code, existing_volume_ids or "(aucun)",
    )

    instance = None
    last_exc: Optional[Exception] = None
    # On combine les variantes de noms de kwargs (spot × location) jusqu'à ce
    # qu'une combinaison soit acceptée par cette version du SDK.
    for spot_kw in ("is_spot", "spot"):
        for loc_kw in ("location_code", "location"):
            try:
                kwargs = dict(create_kwargs)
                kwargs[spot_kw] = spot
                kwargs[loc_kw] = location_code
                instance = client.instances.create(**kwargs)
                LOG.info(
                    "Instance créée (spot via '%s', location via '%s') id=%s",
                    spot_kw, loc_kw, instance.id,
                )
                break
            except TypeError as exc:
                # kwarg non supporté sous ce nom : on essaie une autre combinaison
                last_exc = exc
                continue
            except APIException as exc:
                # Distinguer manque de CAPACITÉ (récupérable) d'une vraie erreur.
                if _is_capacity_error(exc):
                    raise CapacityUnavailable(
                        f"Capacité indisponible pour {instance_type} "
                        f"en {location_code} : {exc}"
                    ) from exc
                raise RodinInfraError(f"Échec création instance : {exc}") from exc
        if instance is not None:
            break
    if instance is None:
        raise RodinInfraError(
            f"Impossible de créer l'instance (combinaisons kwargs épuisées). "
            f"Vérifie la version du SDK. Dernière erreur : {last_exc}"
        )

    result = DeployResult(
        instance_id=instance.id,
        ip="",
        instance_type=instance_type,
        spot=spot,
        attached_volumes=list(existing_volume_ids),
    )

    if not wait_running:
        return result

    ip = wait_until_running(client, instance.id, max_wait=max_wait)
    result.ip = ip
    return result


def deploy_cheapest_available(
    client: VerdaClient,
    *,
    model_hints: tuple[str, ...] | str,
    hostname: str,
    ssh_key_ids: list[str],
    image: str = DEFAULT_IMAGE,
    spot: bool = True,
    os_volume_size_gb: int = 50,
    existing_volume_ids: Optional[list[str]] = None,
    description: str = "rodin",
    startup_script_id: Optional[str] = None,
    max_price: Optional[float] = None,
    locations: tuple[str, ...] = PREFERRED_LOCATIONS,
    wait_running: bool = True,
    max_wait: int = DEPLOY_MAX_WAIT,
) -> DeployResult:
    """
    Déploie le GPU le MOINS CHER réellement DISPONIBLE parmi les candidats
    matchant model_hints. Stratégie robuste à la dispo spot fluctuante :

      pour chaque type (prix croissant) :
        pour chaque région :
          tenter le déploiement
          - succès            -> on renvoie le résultat
          - CapacityUnavailable -> on passe à la combinaison suivante
          - autre erreur      -> on remonte (vraie erreur, pas de la capacité)

    Si AUCUNE combinaison n'a de capacité, lève CapacityUnavailable (le watchdog
    saura réessayer plus tard). C'est exactement le besoin observé : le V100 le
    moins cher peut être indispo, on prend alors le suivant (A6000, A100...).

    Utilisé pour :
      - le build du golden volume (model_hints = GPU jetables bon marché),
      - le run / la reprise B200 (model_hints = "B200").
    """
    candidates = find_candidates(
        client, model_hints, require_spot=spot, max_price=max_price
    )
    if not candidates:
        raise RodinInfraError(
            f"Aucun type ne matche {model_hints!r} (max_price={max_price}). "
            "Rien à déployer."
        )

    LOG.info(
        "Candidats (prix croissant) : %s",
        ", ".join(f"{c.instance_type}@{c.spot_price_per_hour:.3f}" for c in candidates),
    )

    last_capacity_exc: Optional[Exception] = None
    for cand in candidates:
        for loc in locations:
            LOG.info("Tentative : %s en %s (spot %.3f/h)…",
                     cand.instance_type, loc, cand.spot_price_per_hour)
            try:
                return deploy_instance(
                    client,
                    instance_type=cand.instance_type,
                    hostname=hostname,
                    ssh_key_ids=ssh_key_ids,
                    image=image,
                    spot=spot,
                    os_volume_size_gb=os_volume_size_gb,
                    existing_volume_ids=existing_volume_ids,
                    description=description,
                    startup_script_id=startup_script_id,
                    location_code=loc,
                    wait_running=wait_running,
                    max_wait=max_wait,
                )
            except CapacityUnavailable as exc:
                LOG.warning("  → pas de capacité (%s en %s), candidat suivant.",
                            cand.instance_type, loc)
                last_capacity_exc = exc
                continue
            except DeployTimeout as exc:
                # Provisionné mais jamais RUNNING : on nettoie et on passe au suivant.
                LOG.warning("  → timeout RUNNING (%s en %s) : %s", cand.instance_type, loc, exc)
                last_capacity_exc = exc
                continue

    raise CapacityUnavailable(
        f"Aucune capacité disponible pour {model_hints!r} sur {locations} "
        f"(derniers essais épuisés). Réessaie plus tard. Détail : {last_capacity_exc}"
    )


def wait_until_running(
    client: VerdaClient,
    instance_id: str,
    *,
    max_wait: int = DEPLOY_MAX_WAIT,
    poll: int = DEPLOY_POLL_INTERVAL,
) -> str:
    """
    Bloque jusqu'à ce que l'instance soit RUNNING et ait une IP, ou timeout.
    Renvoie l'IP. Lève DeployTimeout si dépassement.

    Tolère les statuts transitoires (provisioning, ordered, etc.) et les
    erreurs API ponctuelles (réseau) sans abandonner.
    """
    try:
        running = client.instance_status.RUNNING
    except AttributeError:
        running = "running"

    deadline = time.monotonic() + max_wait
    last_status = ""
    while time.monotonic() < deadline:
        try:
            status = _instance_status(client, instance_id)
        except APIException as exc:
            LOG.warning("Lecture statut échouée (transitoire) : %s", exc)
            time.sleep(poll)
            continue

        if status != last_status:
            LOG.info("Instance %s : statut=%s", instance_id, status)
            last_status = status

        if str(status).lower() == str(running).lower():
            ip = _instance_ip(client, instance_id)
            if ip:
                LOG.info("Instance %s RUNNING, IP=%s", instance_id, ip)
                return ip
            LOG.info("RUNNING mais IP pas encore attribuée, on patiente…")
        time.sleep(poll)

    raise DeployTimeout(
        f"Instance {instance_id} pas RUNNING après {max_wait}s "
        f"(dernier statut={last_status!r}). Capacité spot indisponible ?"
    )


def delete_instance(
    client: VerdaClient,
    instance_id: str,
    *,
    retries: int = ACTION_MAX_RETRIES,
) -> None:
    """
    Supprime une instance. Le golden volume (rattaché via existing_volumes)
    survit grâce à on_spot_discontinue=keep_detached.
    Retries car une instance en provisioning refuse l'action.
    """
    try:
        delete_action = client.actions.DELETE
    except AttributeError:
        delete_action = "delete"

    for attempt in range(1, retries + 1):
        try:
            client.instances.action(instance_id, delete_action)
            LOG.info("Instance %s : suppression demandée.", instance_id)
            return
        except APIException as exc:
            LOG.warning(
                "Suppression instance %s tentative %d/%d échouée : %s",
                instance_id, attempt, retries, exc,
            )
            time.sleep(ACTION_RETRY_INTERVAL)
    raise RodinInfraError(
        f"Impossible de supprimer l'instance {instance_id} après {retries} tentatives."
    )


def instance_is_alive(client: VerdaClient, instance_id: str) -> bool:
    """
    True si l'instance existe encore et n'est pas dans un état terminal.
    Sert au watchdog : un kick spot fait disparaître l'instance (ou la passe
    dans un état non-running). On considère 'vivante' = statut running.
    """
    try:
        status = _instance_status(client, instance_id)
    except APIException:
        # plus joignable : on considère qu'elle n'est plus vivante
        return False
    if not status:
        return False
    return str(status).lower() == "running"


# ──────────────────────────────────────────────────────────────────────────
# Garde-fou budget
# ──────────────────────────────────────────────────────────────────────────

def assert_budget_ok(
    client: VerdaClient,
    *,
    spot_price_per_hour: float,
    max_total_eur: float,
    est_hours: float,
) -> None:
    """
    Refuse de continuer si le coût estimé dépasse le budget, ou si le solde
    est insuffisant. Sécurité anti-dérapage (contrainte projet : < 200€).
    """
    est_cost = spot_price_per_hour * est_hours
    LOG.info(
        "Estimation : %.3f/h × %.1f h = %.2f (budget max %.2f).",
        spot_price_per_hour, est_hours, est_cost, max_total_eur,
    )
    if est_cost > max_total_eur:
        raise RodinInfraError(
            f"Coût estimé {est_cost:.2f} > budget {max_total_eur:.2f}. "
            "Réduis le run ou ajuste le budget volontairement."
        )
    balance = get_balance(client)
    if balance < est_cost:
        LOG.warning(
            "Solde compte %.2f < coût estimé %.2f. Recharge avant le run.",
            balance, est_cost,
        )


# ──────────────────────────────────────────────────────────────────────────
# Self-test (exécuté en direct : python verda_lib.py)
# ──────────────────────────────────────────────────────────────────────────

def _selftest() -> int:
    """
    Vérifie l'auth, liste les SSH keys, le solde, et repère le type B200 spot
    + une option jetable bon marché. NE déploie RIEN, NE crée RIEN.
    À lancer en premier pour confirmer que les credentials marchent.
    """
    LOG.info("=== SELF-TEST verda_lib (lecture seule, aucun déploiement) ===")
    client = build_client()

    ssh_ids = get_ssh_key_ids(client)
    LOG.info("SSH keys : %d", len(ssh_ids))

    bal = get_balance(client)
    LOG.info("Solde compte : %.2f", bal)

    LOG.info("--- Recherche B200 spot ---")
    try:
        b200 = find_instance_type(client, B200_GPU_MODEL_HINT, require_spot=True)
        LOG.info("OK B200 spot : %s @ %.3f/h", b200.instance_type, b200.spot_price_per_hour)
    except RodinInfraError as exc:
        LOG.warning("B200 spot non trouvé pour l'instant : %s", exc)

    LOG.info("--- Recherche instance jetable bon marché (build golden volume) ---")
    try:
        cheap = find_instance_type(client, CHEAP_GPU_MODEL_HINTS, require_spot=True)
        LOG.info("OK jetable : %s @ %.3f/h", cheap.instance_type, cheap.spot_price_per_hour)
    except RodinInfraError as exc:
        LOG.warning("Pas d'option jetable trouvée : %s", exc)

    LOG.info("=== SELF-TEST terminé OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())

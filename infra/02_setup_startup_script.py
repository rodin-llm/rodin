#!/usr/bin/env python3
# ============================================================================
# RODIN — 02_setup_startup_script.py
# ----------------------------------------------------------------------------
# Gère le STARTUP SCRIPT Verda via l'API (PAS besoin du dashboard).
#
# Confirmé sur la doc SDK Verda 1.24 :
#   verda.startup_scripts.create(name, script) -> StartupScript(id, name, script)
#   verda.startup_scripts.get()                -> liste de StartupScript
#   verda.startup_scripts.get_by_id(id)        -> StartupScript
#   verda.startup_scripts.delete_by_id(id)     -> suppression
#   (PAS de méthode update : "modifier" = delete + recreate, l'id CHANGE)
#
# Le startup script créé ici contient le code de verda_startup.sh (le bootstrap
# minimal qui monte /data puis exec /data/scripts/bootstrap.sh). On récupère son
# .id et on le PERSISTE dans startup_script.json. Cet id se passe ensuite à
# deploy_instance(..., startup_script_id=...) à CHAQUE déploiement (build/run/
# watchdog).
#
# IDEMPOTENT : repérage par NOM. Si un startup du même nom existe déjà :
#   - identique au fichier  -> on réutilise (rien à faire)
#   - différent             -> on RE-CRÉE (delete + create) car pas d'update API
#                              => l'id change, on met à jour le JSON, et il faut
#                                 re-déployer pour que le nouvel id soit utilisé.
#
# PHASES :
#   --phase ensure   : crée si absent / réutilise si identique / recrée si diffère.
#                      (phase par défaut)
#   --phase status   : lecture seule, montre l'état courant + tous les startups.
#   --phase delete   : supprime le startup script suivi (avec confirmation).
#
# Style RODIN : état persistant JSON, dry-run sur le destructif, fail clair.
# Usage :
#   python 02_setup_startup_script.py --phase ensure
#   python 02_setup_startup_script.py --phase status
#   python 02_setup_startup_script.py --phase delete            # demande confirmation
#   python 02_setup_startup_script.py --phase delete --yes      # sans confirmation
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import verda_lib as vl


# ============================================================================
# CONFIG
# ============================================================================
HERE = Path(__file__).resolve().parent

# Nom du startup script côté Verda (sert de clé d'idempotence).
STARTUP_NAME = "rodin-startup"

# Fichier source du code bash (celui généré à la session précédente).
STARTUP_SRC = HERE / "verda_startup.sh"

# État persistant : mémorise l'id du startup script créé.
STATE_FILE = HERE / "startup_script.json"

LOG = vl.get_logger("setup_startup")


# ============================================================================
# État persistant
# ============================================================================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("State file %s illisible, on repart de zéro.", STATE_FILE)
    return {}


def save_state(state: dict) -> None:
    # écriture atomique : .tmp puis rename
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_FILE)
    LOG.info("État sauvegardé -> %s", STATE_FILE)


# ============================================================================
# Helpers startup scripts
# ============================================================================
def read_source_script() -> str:
    if not STARTUP_SRC.exists():
        raise vl.RodinInfraError(
            f"Fichier source introuvable : {STARTUP_SRC}\n"
            f"Place verda_startup.sh à côté de ce script."
        )
    code = STARTUP_SRC.read_text(encoding="utf-8")
    if not code.strip():
        raise vl.RodinInfraError(f"{STARTUP_SRC} est vide.")
    return code


def list_startups(client) -> list:
    """Tous les startup scripts du compte (objets StartupScript)."""
    return list(client.startup_scripts.get())


def find_by_name(client, name: str):
    """Renvoie le StartupScript du nom donné, ou None."""
    for s in list_startups(client):
        if getattr(s, "name", None) == name:
            return s
    return None


def get_script_code(s) -> str:
    """Code bash d'un StartupScript (attribut .script)."""
    return getattr(s, "script", "") or ""


def create_startup(client, name: str, code: str):
    LOG.info("Création startup script '%s' (%d octets)…", name, len(code.encode()))
    s = client.startup_scripts.create(name, code)
    LOG.info("Startup script créé : id=%s name=%s", s.id, s.name)
    return s


def delete_startup(client, script_id: str) -> None:
    LOG.info("Suppression startup script id=%s …", script_id)
    client.startup_scripts.delete_by_id(script_id)
    LOG.info("Supprimé.")


# ============================================================================
# Phases
# ============================================================================
def phase_status(client) -> int:
    state = load_state()
    print("── État local (startup_script.json) ──")
    if state:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print("  (aucun état local)")

    print("\n── Startup scripts présents sur le compte Verda ──")
    scripts = list_startups(client)
    if not scripts:
        print("  (aucun)")
    for s in scripts:
        marque = "  <-- suivi" if state.get("id") == s.id else ""
        n_bytes = len(get_script_code(s).encode())
        print(f"  id={s.id}  name={s.name!r}  ({n_bytes} octets){marque}")

    # Cohérence : l'id suivi existe-t-il encore côté Verda ?
    if state.get("id"):
        still = any(s.id == state["id"] for s in scripts)
        print(f"\nL'id suivi {state['id']} existe sur Verda : {'OUI' if still else 'NON (obsolète)'}")
    return 0


def phase_ensure(client) -> int:
    code = read_source_script()
    state = load_state()
    existing = find_by_name(client, STARTUP_NAME)

    if existing is None:
        # rien de ce nom -> on crée
        s = create_startup(client, STARTUP_NAME, code)
        save_state({
            "id": s.id,
            "name": s.name,
            "source_file": str(STARTUP_SRC),
            "source_bytes": len(code.encode()),
        })
        print(f"\n✅ Startup script créé. id = {s.id}")
        print("   -> passe cet id à deploy_instance(startup_script_id=...).")
        return 0

    # un startup du même nom existe : identique ou différent ?
    remote_code = get_script_code(existing)
    if remote_code.strip() == code.strip():
        LOG.info("Startup '%s' déjà présent et IDENTIQUE (id=%s). Rien à faire.",
                 STARTUP_NAME, existing.id)
        save_state({
            "id": existing.id,
            "name": existing.name,
            "source_file": str(STARTUP_SRC),
            "source_bytes": len(code.encode()),
        })
        print(f"\n✅ Startup script déjà à jour. id = {existing.id}")
        return 0

    # différent -> pas d'update API : delete + recreate (l'id change)
    LOG.warning(
        "Startup '%s' existe (id=%s) mais son contenu DIFFÈRE du fichier local. "
        "L'API n'a pas d'update : on va supprimer puis recréer (nouvel id).",
        STARTUP_NAME, existing.id,
    )
    delete_startup(client, existing.id)
    s = create_startup(client, STARTUP_NAME, code)
    save_state({
        "id": s.id,
        "name": s.name,
        "source_file": str(STARTUP_SRC),
        "source_bytes": len(code.encode()),
        "replaced_id": existing.id,
    })
    print(f"\n✅ Startup script RECRÉÉ. NOUVEL id = {s.id}")
    print(f"   (ancien id {existing.id} supprimé)")
    print("   ⚠ RE-DÉPLOIE pour que les instances utilisent le nouvel id.")
    return 0


def phase_delete(client, assume_yes: bool) -> int:
    state = load_state()
    target_id = state.get("id")
    if not target_id:
        # pas d'état local : on tente par nom
        existing = find_by_name(client, STARTUP_NAME)
        if existing is None:
            print("Rien à supprimer (aucun état local, aucun startup du nom "
                  f"'{STARTUP_NAME}').")
            return 0
        target_id = existing.id

    # vérifier qu'il existe encore
    if not any(s.id == target_id for s in list_startups(client)):
        print(f"L'id suivi {target_id} n'existe déjà plus côté Verda. "
              "Nettoyage de l'état local.")
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        return 0

    if not assume_yes:
        rep = input(f"Supprimer le startup script id={target_id} ? [oui/NON] ").strip().lower()
        if rep not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            return 1

    delete_startup(client, target_id)
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        LOG.info("État local supprimé.")
    print("✅ Supprimé.")
    return 0


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Gère le startup script Verda (RODIN).")
    ap.add_argument("--phase", default="ensure",
                    choices=["ensure", "status", "delete"])
    ap.add_argument("--yes", action="store_true",
                    help="confirme automatiquement (phase delete).")
    args = ap.parse_args()

    try:
        client = vl.build_client()
    except vl.RodinInfraError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 2

    if args.phase == "status":
        return phase_status(client)
    if args.phase == "ensure":
        return phase_ensure(client)
    if args.phase == "delete":
        return phase_delete(client, assume_yes=args.yes)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

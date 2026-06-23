# rodin_data.py
# Dataloader RODIN pour le pretraining 3090 (et plus tard le run cloud).
#
# Entree : un .bin PLAT uint16 produit par 20_build_blend.py (train.bin / val.bin).
#   - flux de tokens contigu, EOS (id=3) inclus comme separateur de doc,
#   - ponderation du blend DEJA faite a la materialisation -> ici, lecture
#     purement sequentielle, aucune logique de melange par source.
#
# Principe (cf. handoff section 5A) :
#   - np.memmap uint16 mode="r" : on ne charge JAMAIS le .bin en RAM,
#   - fenetres SEQUENTIELLES de (ctx+1) tokens : input=[0:ctx], target=[1:ctx+1],
#   - map-style Dataset : __len__ = nb de fenetres, __getitem__ = une fenetre,
#   - le decoupage en fenetres est NON CHEVAUCHANT par defaut (stride=ctx) ->
#     chaque token vu une fois par epoch, comme un pretraining standard,
#   - PAS de masquage cross-doc : l'EOS dans le flux suffit (standard from-scratch),
#   - resume = un simple offset entier (index de fenetre de depart).
#
# Le DataLoader (num_workers=4, prefetch_factor=2, pin_memory=True) est construit
# par make_dataloader(). Chaque worker ouvre SON PROPRE memmap (les memmap ne se
# partagent pas proprement entre process via fork/spawn) -> worker_init_fn.
#
# Aucune dependance cloud. Aucune dependance au reste du repo RODIN.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EOS_ID = 3            # separateur de doc dans le flux (info, non utilise ici)
DTYPE = np.uint16     # format du .bin (vocab 64000 < 65536 -> uint16 OK)


def _open_memmap(path):
    """Ouvre le .bin en lecture seule sans le charger en RAM."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[rodin_data] introuvable : {path}")
    size_bytes = os.path.getsize(path)
    if size_bytes % 2 != 0:
        raise ValueError(f"[rodin_data] taille impaire ({size_bytes} o) : "
                         f"{path} n'est pas un flux uint16 valide.")
    return np.memmap(path, dtype=DTYPE, mode="r")


class BinWindowDataset(Dataset):
    """Dataset map-style sur un .bin uint16 plat, decoupe en fenetres
    sequentielles de (ctx+1) tokens.

    n_windows = (n_tokens - 1) // stride
    fenetre k : tokens [k*stride : k*stride + ctx + 1]
        input  = fenetre[:-1]   (ctx tokens)
        target = fenetre[1:]    (ctx tokens, decales de 1)

    Le memmap est ouvert paresseusement (lazy) pour survivre au fork/spawn des
    workers : le handle reel est cree dans chaque process via _ensure_open().
    """

    def __init__(self, bin_path, ctx, stride=None):
        self.bin_path = bin_path
        self.ctx = int(ctx)
        self.stride = int(stride) if stride is not None else int(ctx)
        if self.stride <= 0:
            raise ValueError("stride doit etre > 0")

        # On lit la TAILLE sans garder le memmap (chaque worker rouvrira le sien).
        size_bytes = os.path.getsize(bin_path)
        self.n_tokens = size_bytes // 2          # uint16
        if self.n_tokens < self.ctx + 1:
            raise ValueError(
                f"[rodin_data] {bin_path} : {self.n_tokens:,} tokens < ctx+1 "
                f"({self.ctx + 1}). Fichier trop petit ou ctx trop grand.")
        # nb de fenetres dont l'indice de fin (k*stride + ctx) reste < n_tokens
        self.n_windows = (self.n_tokens - 1 - self.ctx) // self.stride + 1
        self._data = None                        # memmap, ouvert par worker

    def _ensure_open(self):
        if self._data is None:
            self._data = _open_memmap(self.bin_path)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.n_windows
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(idx)
        self._ensure_open()
        start = idx * self.stride
        end = start + self.ctx + 1
        # copie explicite hors du memmap -> tableau possede, castable sans souci
        window = np.asarray(self._data[start:end], dtype=np.int64)
        x = torch.from_numpy(window[:-1])        # (ctx,)
        y = torch.from_numpy(window[1:])         # (ctx,)
        return x, y


def _worker_init_fn(worker_id):
    """Chaque worker rouvre son propre memmap (pas de partage de handle)."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._data = None                # force reouverture lazy


class OffsetSampler(torch.utils.data.Sampler):
    """Sampler sequentiel DEMARRANT a une fenetre donnee (resume).
    start_index = index de fenetre de depart (0 = debut). Parcourt ensuite
    toutes les fenetres restantes dans l'ordre, une seule fois (1 epoch)."""

    def __init__(self, n_windows, start_index=0):
        self.n_windows = int(n_windows)
        self.start_index = int(start_index) % max(self.n_windows, 1)

    def __iter__(self):
        return iter(range(self.start_index, self.n_windows))

    def __len__(self):
        return self.n_windows - self.start_index


def make_dataloader(bin_path, ctx, batch_size,
                    num_workers=4, prefetch_factor=2, pin_memory=True,
                    shuffle=False, start_index=0, stride=None, drop_last=True,
                    seed=1234):
    """Construit (dataset, dataloader).

    - shuffle=False (defaut, pretraining) : lecture sequentielle, resume via
      start_index (index de fenetre). C'est le mode du pretest et du run.
    - shuffle=True : ordre aleatoire reproductible (utile en validation/debug).
      start_index est alors ignore.
    """
    ds = BinWindowDataset(bin_path, ctx=ctx, stride=stride)

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(ds, generator=g)
    else:
        sampler = OffsetSampler(len(ds), start_index=start_index)

    dl_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = True

    dl = DataLoader(ds, **dl_kwargs)
    return ds, dl


# ----------------------------------------------------------------------
# Self-test : `python rodin_data.py /chemin/train.bin` (ou val.bin)
# Verifie ouverture, comptage de fenetres, et une fenetre exemple.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python rodin_data.py <chemin_vers_bin> [ctx]")
        sys.exit(1)
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    ds, dl = make_dataloader(path, ctx=ctx, batch_size=2,
                             num_workers=0, shuffle=False)
    print(f"[selftest] {path}")
    print(f"  tokens   : {ds.n_tokens:,}")
    print(f"  ctx      : {ds.ctx}  stride : {ds.stride}")
    print(f"  fenetres : {len(ds):,}")
    x, y = next(iter(dl))
    print(f"  batch x  : {tuple(x.shape)} dtype {x.dtype}")
    print(f"  batch y  : {tuple(y.shape)} dtype {y.dtype}")
    # invariant : y[t] == x[t+1] sur la 1re sequence
    shift_ok = bool(torch.equal(x[0, 1:], y[0, :-1]))
    print(f"  shift target == input decale : {'OK' if shift_ok else 'KO'}")
    print(f"  min id {int(x.min())}  max id {int(x.max())} (doit etre < 64000)")

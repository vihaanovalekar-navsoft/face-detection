"""Multi-identity gallery with enrollment, matching and self-improving updates."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

MAX_OBS_PER_USER = 16


class Gallery:
    def __init__(self, path=None):
        self.ids: list[str] = []
        self.templates: list[np.ndarray] = []
        self.counts: list[int] = []
        self.stored: list[list[np.ndarray]] = []  # up to MAX_OBS_PER_USER per user
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            data = None
            for attempt in range(3):
                try:
                    data = np.load(self.path, allow_pickle=True)
                    break
                except Exception:
                    time.sleep(0.05)

            if data is not None:
                self.ids = [str(x) for x in data["ids"]]
                self.templates = [t for t in data["templates"]]
                self.counts = [int(x) for x in data["counts"]]
                if "stored" in data:
                    stored_raw = data["stored"]
                    self.stored = [list(x) for x in stored_raw] if len(stored_raw) else [[] for _ in self.ids]
                else:
                    self.stored = [[] for _ in self.ids]

    def __len__(self):
        return len(self.ids)

    def clear(self):
        """Remove all enrolled identities and persist to disk."""
        self.ids = []
        self.templates = []
        self.counts = []
        self.stored = []
        if self.path:
            self.save(self.path)

    def match(self, emb, threshold=0.35):
        """Match embedding against master templates and individual multi-angle stored embeddings."""
        if not self.templates:
            return None, 0.0

        # Ensure normalized
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        best_id = None
        best_sim = -1.0

        for i, name in enumerate(self.ids):
            # 1. Similarity with master composite template
            t_sim = float(self.templates[i] @ emb_norm)
            cur_max = t_sim

            # 2. Similarity against individual angle embeddings (multi-angle profile)
            if i < len(self.stored) and self.stored[i]:
                stored_sims = [float(s @ emb_norm) for s in self.stored[i]]
                if stored_sims:
                    s_max = max(stored_sims)
                    if s_max > cur_max:
                        cur_max = s_max

            if cur_max > best_sim:
                best_sim = cur_max
                best_id = name

        if best_sim >= threshold:
            return best_id, float(best_sim)
        return None, float(best_sim)

    identify = match  # convenient alias

    def delete(self, name: str) -> bool:
        """Delete an identity from the gallery and save to disk."""
        if name not in self.ids:
            return False
        idx = self.ids.index(name)
        self.ids.pop(idx)
        self.templates.pop(idx)
        self.counts.pop(idx)
        self.stored.pop(idx)
        if self.path:
            self.save(self.path)
        return True

    def enroll(self, emb, name=None):
        """Enroll single embedding (delegates to enroll_profile)."""
        if name is None:
            name = f"user{len(self.ids) + 1}"
        return self.enroll_profile(name, [emb])

    def enroll_profile(self, name: str, embeddings: list[np.ndarray]):
        """Enroll a person with multi-angle embeddings (e.g. 8 frames: frontal + sideways)."""
        if not embeddings:
            raise ValueError("No embeddings provided for enrollment")

        # Normalize all embeddings
        normed = [e.copy() / (np.linalg.norm(e) + 1e-8) for e in embeddings]
        # Master template is the normalized mean embedding
        mean_emb = np.mean(normed, axis=0)
        master_template = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        if name in self.ids:
            idx = self.ids.index(name)
            self.templates[idx] = master_template
            self.stored[idx] = normed
            self.counts[idx] = len(normed)
        else:
            self.ids.append(name)
            self.templates.append(master_template)
            self.stored.append(normed)
            self.counts.append(len(normed))

        if self.path:
            self.save(self.path)
        return name

    def update(self, name, emb, alpha=0.02, store=True):
        if name not in self.ids:
            return
        i = self.ids.index(name)
        norm_emb = emb.copy() / (np.linalg.norm(emb) + 1e-8)
        # store new observation up to the cap (16 per user), only when gated
        if store and len(self.stored[i]) < MAX_OBS_PER_USER:
            self.stored[i].append(norm_emb)
            self.counts[i] = len(self.stored[i])
        # self-improve template via EMA with anchor protection
        t = self.templates[i]
        new = (1 - alpha) * t + alpha * norm_emb
        new = new / (np.linalg.norm(new) + 1e-8)
        # Anchor protection: ensure updated template doesn't drift away from the baseline anchor
        anchor = self.stored[i][0] if len(self.stored[i]) > 0 else t
        if float(new @ anchor) >= 0.45:
            self.templates[i] = new

    def stored_embeddings(self, name):
        i = self.ids.index(name)
        return self.stored[i]

    def rename(self, old, new):
        if old not in self.ids:
            raise KeyError(f"unknown identity: {old}")
        if new in self.ids:
            raise ValueError(f"identity already exists: {new}")
        self.ids[self.ids.index(old)] = new
        self.save(self.path)
        return new

    def save(self, path=None):
        path = Path(path) if path else self.path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        stored_arr = np.array([np.stack(s) if s else np.zeros((0, 512))
                              for s in self.stored], dtype=object)

        # Atomic write to prevent corrupt reads from concurrent processes
        tmp_npz = path.with_name(f"{path.stem}.tmp{path.suffix}")
        np.savez(tmp_npz, ids=np.array(self.ids),
                 templates=np.stack(self.templates) if self.templates else np.zeros((0, 512)),
                 counts=np.array(self.counts),
                 stored=stored_arr)
        try:
            os.replace(tmp_npz, path)
        except OSError:
            # Fallback if atomic replace fails
            if tmp_npz.exists():
                tmp_npz.rename(path)

        json_path = path.with_suffix(".json")
        tmp_json = path.with_name(f"{path.stem}.tmp.json")
        with open(tmp_json, "w") as f:
            json.dump({"ids": [str(x) for x in self.ids],
                       "counts": [int(x) for x in self.counts],
                       "max_obs": MAX_OBS_PER_USER}, f, indent=2)
        try:
            os.replace(tmp_json, json_path)
        except OSError:
            if tmp_json.exists():
                tmp_json.rename(json_path)

    @classmethod
    def load(cls, path):
        g = cls(path=None)
        data = np.load(path, allow_pickle=True)
        g.ids = [str(x) for x in data["ids"]]
        g.templates = [t for t in data["templates"]]
        g.counts = [int(x) for x in data["counts"]]
        if "stored" in data:
            stored_raw = data["stored"]
            g.stored = [list(x) for x in stored_raw] if len(stored_raw) else [[] for _ in g.ids]
        else:
            g.stored = [[] for _ in g.ids]
        g.path = Path(path)
        return g

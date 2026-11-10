"""Profiles: named sets of settings for different games, avatars or people.

A profile remembers the game mode, body size (height, scale, offsets), the
calibration (bone lengths — "how long your legs are"), which trackers are on
and the smoothing settings (see config.PROFILE_KEYS). Switching is instant:
from the drop-down in the main window or Ctrl+1 … Ctrl+9.

Profiles live in the same config.json and go through the same strict
validation as all other settings.
"""
from __future__ import annotations

import copy
from typing import List, Optional

from .config import MAX_PROFILES, PROFILE_KEYS, Config, clean_profile_name

RESTART_KEYS = {"target"}   # changing these needs the tracking restarted


def names(cfg: Config) -> List[str]:
    return list(cfg.profiles.keys())


def save_profile(cfg: Config, name: str) -> str:
    """Store the current settings under `name` (overwrites). Returns the clean name."""
    name = clean_profile_name(name)
    if not name:
        raise ValueError("пустое имя профиля")
    if name not in cfg.profiles and len(cfg.profiles) >= MAX_PROFILES:
        raise ValueError(f"не больше {MAX_PROFILES} профилей")
    prof = {k: copy.deepcopy(getattr(cfg, k)) for k in PROFILE_KEYS}
    prof["user_set"] = [k for k in PROFILE_KEYS if k in cfg.user_set]
    cfg.profiles = dict(cfg.profiles)
    cfg.profiles[name] = prof
    cfg.active_profile = name
    cfg.validate()
    return name


def apply_profile(cfg: Config, name: str) -> List[str]:
    """Switch to a profile. Returns the keys whose value changed."""
    prof = cfg.profiles.get(name)
    if prof is None:
        raise KeyError(name)
    changed = []
    for k in PROFILE_KEYS:
        if k not in prof:
            continue
        v = copy.deepcopy(prof[k])
        if getattr(cfg, k) != v:
            setattr(cfg, k, v)
            changed.append(k)
    us = set(cfg.user_set) - set(PROFILE_KEYS)
    us.update(prof.get("user_set", []))
    cfg.user_set = sorted(us)
    cfg.active_profile = name
    cfg.validate()
    return changed


def delete_profile(cfg: Config, name: str) -> None:
    cfg.profiles = {k: v for k, v in cfg.profiles.items() if k != name}
    if cfg.active_profile == name:
        cfg.active_profile = ""
    cfg.validate()


def rename_profile(cfg: Config, old: str, new: str) -> str:
    new = clean_profile_name(new)
    if not new or old not in cfg.profiles:
        raise ValueError("нет такого профиля")
    if new != old and new in cfg.profiles:
        raise ValueError("профиль с таким именем уже есть")
    cfg.profiles = {(new if k == old else k): v for k, v in cfg.profiles.items()}
    if cfg.active_profile == old:
        cfg.active_profile = new
    cfg.validate()
    return new


def differs(cfg: Config, name: Optional[str] = None) -> bool:
    """The current settings differ from the (active) profile."""
    name = name or cfg.active_profile
    prof = cfg.profiles.get(name)
    if prof is None:
        return False
    return any(k in prof and getattr(cfg, k) != prof[k] for k in PROFILE_KEYS)

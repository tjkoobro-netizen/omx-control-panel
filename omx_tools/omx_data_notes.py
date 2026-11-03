#!/usr/bin/env python3
"""What each dataset in the training tab is: group, source and description.

A name of the team's data rule MMDD_source_keywords_Nep describes itself. What a name cannot tell comes from the
team file <shared_root>/omx_presets/data_notes.json, so the notes need no panel release and stay out of the public repo:
  {"notes": {"<name>": {"use": "...", "how": "...", "action": "..."}},
   "legacy": {"<name>": {"group": "...", "how": "...", "action": "..."}},
   "legacy_note": "shown above the groups of older data"}
Reads only meta/info.json values the caller already has: called for every status poll, it opens no data file.
"""

from __future__ import annotations

import json

import omx_names as names
import omx_protocol as proto

SOURCE = {"real": "실물 로봇을 텔레옵한 녹화", "sim": "실물 시연을 MuJoCo에서 재생, 기준 통과분", "algo": "알고리즘 궤적을 MuJoCo에서 실행, 기준 통과분",
          "realsim": "실물과 시뮬레이터 데이터를 합친 세트"}
SOURCE_ORDER = ["real", "realsim", "sim", "algo"]
WORDS = {"twoarm": "두 팔", "onearm": "한 팔", "box": "상자", "strawberry": "딸기", "shelf": "선반", "shelf1": "선반 1층", "shelf2": "선반 2층", "bowl": "그릇",
         "handover": "건네주기", "symcarry": "대칭 운반", "tee": "T자 손잡이", "arch": "아치 손잡이"}
_CACHE = {"mtime": None, "data": {}}


def load() -> dict:
    f = proto.preset_dir() / "data_notes.json"
    try:
        mtime = f.stat().st_mtime
        if mtime != _CACHE["mtime"]:
            _CACHE.update(mtime=mtime, data=json.loads(f.read_text()))
    except (OSError, ValueError):
        _CACHE.update(mtime=None, data={})
    return _CACHE["data"]


def action_words(keywords: str) -> str:
    return ", ".join(f"마커 {w[2:]}" if w.startswith("mk") and w[2:].isdigit() else WORDS.get(w, w) for w in keywords.split("_"))


def describe(owner: str, name: str, episodes: int, me: str) -> dict:
    """Fields for one dataset: group ("rule", "new", "other" or a legacy group's title), source, how, action, use, warn."""
    data = load()
    note, legacy, rule = (data.get("notes") or {}).get(name), (data.get("legacy") or {}).get(name), names.RULE_NAME.match(name)
    out = {"group": "new", "source": "", "how": "", "action": "", "use": "", "warn": "", "mmdd": ""}
    if rule:
        out.update(source=rule[2], mmdd=rule[1], how=SOURCE[rule[2]], action=action_words(rule[3]), use="미정")
        if int(rule[4]) != episodes:
            out["warn"] = f"이름은 {rule[4]}개인데 실제 {episodes}개"
        out["group"] = "rule"
    if note:
        out.update({k: note[k] for k in ("use", "how", "action") if note.get(k)})
    if legacy and not rule:
        out.update(group=legacy.get("group") or "new", how=legacy.get("how") or "", action=legacy.get("action") or "")
    if owner != me:
        out["group"] = "other"
    return out

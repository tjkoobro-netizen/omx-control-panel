#!/usr/bin/env python3
"""Team naming rule for models and evaluations.

  model:      {task}_{policy}_{data}_{steps}_{MMDD}[-extras]     e.g. straw_diff_sw48_20k_1024
    task      short task word (straw, dutyfree), from the task sentence
    policy    act / diff / smolvla / pi0
    data      original recording code + episodes used, joined when mixed (pl50sw48; d1013x21 when the
              code ends in a digit)
    steps     40k
    MMDD      the day training started
    extras    only when different from the defaults: -noaug (image augmentation off), -ev12 (12 evaluation
              attempts included), -bs16 (batch other than the policy's default)
  evaluation: eval_{model display name}[-at<step>]__{p10 | free}_{run settings}_{MMDD_HHMM}
    (-at12500 when an intermediate save of the model was evaluated; shown as name@12.5k)
    run settings: sync | async05 (queue threshold 0.5), then rtc / smooth / ddim10 when used

Recording codes come from the team file <shared_root>/omx_presets/data_codes.json (kept by the training
guide; written with DEFAULT_CODES the first time). A recording without a code gets a temporary one from
the start of its name, with a warning.
Existing models keep their folder names (eval notes point at them); they get a display name (alias) in
<checkpoints>/aliases.json: {"<run id>": "<display name>"}. A model trained with a rule name needs none.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import omx_config
import omx_protocol as proto

DEFAULT_CODES = {
    "datasets": {
        "taejackooo_omx_strawberry_pilot": "pl", "taejackooo_omx_strawberry_pilot_2": "pl",
        "taejackooo_omx_strawberry_pilot_3": "pl", "taejackooo_omx_strawberry_pilot_4": "pl",
        "taejackooo_omx_strawberry_pilot_5": "pl", "taejackooo_omx_strawberry_sweep": "sw",
        "strawberry_train_1013": "d1013", "1013_2130": "d1013", "1013_2200": "d1013", "1013_2203": "d1013",
        "1013_2211": "d1013",
    },
    "tasks": {"strawberry": "straw", "duty free": "dutyfree", "dutyfree": "dutyfree"},
}
POLICY_CODES = {"act": "act", "diffusion": "diff", "smolvla": "smolvla", "pi0": "pi0", "pi05": "pi05"}
DEFAULT_BATCH = {"act": 8, "diffusion": 16, "smolvla": 16}
NAME_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[a-z0-9]+_\d+k_\d{4}(-[a-z0-9]+)*$")
CKPT_HOME = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints"))))


def codes_path() -> Path:
    return proto.preset_dir() / "data_codes.json"


def load_codes() -> dict:
    f = codes_path()
    if not f.exists():
        try:
            f.write_text(json.dumps(DEFAULT_CODES, ensure_ascii=False, indent=2))
            os.chmod(f, 0o664)
        except OSError:
            return DEFAULT_CODES
    try:
        c = json.loads(f.read_text())
        return {"datasets": c.get("datasets") or {}, "tasks": c.get("tasks") or DEFAULT_CODES["tasks"]}
    except (OSError, ValueError):
        return DEFAULT_CODES


def data_code(repo: str, codes: dict) -> tuple[str, bool]:
    """(code, known) for an original recording ("user/name" or "name")."""
    name = repo.rsplit("/", 1)[-1]
    table = codes["datasets"]
    if repo in table or name in table:
        return table.get(repo) or table[name], True
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w and w not in ("omx", "taejackooo", "train")]
    return (re.sub(r"[^a-z0-9]", "", "".join(words))[:6] or "data"), False


def task_code(tasks: list[str], codes: dict) -> str:
    text = " ".join(tasks).lower()
    for word, code in codes["tasks"].items():
        if word in text:
            return code
    words = [w for w in re.findall(r"[a-z]+", text) if len(w) > 3 and w not in ("pick", "place", "put", "the", "into", "with")]
    return words[0][:8] if words else "task"


def steps_code(steps: int) -> str:
    return f"{round(steps / 1000)}k" if steps >= 1000 else f"{steps}s"


def model_name(policy: str, steps: int, mmdd: str, origin: dict, tasks: list[str], augmentation: bool = True,
               batch_size: int | None = None, eval_attempts: int = 0) -> dict:
    """Rule name from what a model is trained on. origin: {recording repo: [episodes]} (eval repos are counted
    as -ev). Returns {"name", "warnings"}."""
    codes = load_codes()
    warnings, counts = [], {}
    for repo, eps in origin.items():
        if repo.rsplit("/", 1)[-1].startswith("eval_"):
            eval_attempts += len(eps)
            continue
        code, known = data_code(repo, codes)
        if not known:
            warnings.append(f"{repo}의 데이터 약어가 약어표({codes_path()})에 없어 임시 약어 '{code}'를 씁니다")
        counts[code] = counts.get(code, 0) + len(eps)
    # a code ending in a digit gets "x" before the count (d1013x21), so the two numbers stay apart
    data = "".join(f"{c}{'x' if c[-1:].isdigit() else ''}{n}" for c, n in counts.items()) or "data"
    extras = []
    if not augmentation:
        extras.append("noaug")
    if eval_attempts:
        extras.append(f"ev{eval_attempts}")
    if batch_size and DEFAULT_BATCH.get(policy) and batch_size != DEFAULT_BATCH[policy]:
        extras.append(f"bs{batch_size}")
    name = "_".join([task_code(tasks, codes), POLICY_CODES.get(policy, policy), data, steps_code(steps), mmdd])
    return {"name": name + "".join(f"-{e}" for e in extras), "warnings": warnings}


def clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-")[:80]


# ---------------------------------------------------------------- display names of existing models
def aliases_path() -> Path:
    return CKPT_HOME / "aliases.json"


def load_aliases() -> dict:
    try:
        return json.loads(aliases_path().read_text())
    except (OSError, ValueError):
        return {}


def set_aliases(updates: dict) -> dict:
    """Add or change display names ({run id: name}; an empty name removes it)."""
    a = load_aliases()
    for rid, name in updates.items():
        name = clean_name(name or "")
        if name:
            a[rid] = name
        else:
            a.pop(rid, None)
    aliases_path().write_text(json.dumps(a, ensure_ascii=False, indent=2))
    try:
        os.chmod(aliases_path(), 0o664)
    except PermissionError:
        pass
    return a


def display_name(run_id: str | None, aliases: dict | None = None) -> str:
    """The name to show for a model (run id = folder under the checkpoints, e.g. tkoo/act_strawberry_1013)."""
    if not run_id:
        return ""
    a = load_aliases() if aliases is None else aliases
    return a.get(run_id) or run_id.rsplit("/", 1)[-1]


def suggest_for_run(r: dict) -> dict:
    """Rule name for a model in the training index (an index.json run)."""
    started = r.get("started") or r.get("finished") or ""
    m = re.match(r"\d{4}-(\d{2})-(\d{2})", started)
    return model_name(r.get("policy") or "", int(r.get("steps") or 0), (m.group(1) + m.group(2)) if m else "0000",
                      r.get("origin") or {}, r.get("tasks") or [], bool(r.get("augmentation")), r.get("batch_size"))


# ---------------------------------------------------------------- intermediate checkpoints
CKPT_STEP_RE = re.compile(r"/checkpoints/(\d+)/pretrained_model/?$")


def checkpoint_step(model_path: str | None) -> int | None:
    """Training step of an intermediate save (<run>/checkpoints/012500/pretrained_model), else None."""
    m = CKPT_STEP_RE.search(str(model_path or ""))
    return int(m.group(1)) if m else None


def step_text(step: int | None) -> str:
    if not step:
        return ""
    return f"{step / 1000:g}k" if step >= 1000 else str(step)


def run_steps() -> dict:
    """{run id: final training steps} from the team training index."""
    try:
        return {r["id"]: r.get("steps") for r in json.loads((CKPT_HOME / "index.json").read_text()).get("runs", [])}
    except (OSError, ValueError):
        return {}


def train_amount(note: dict, steps: dict | None = None) -> int | None:
    """How far the evaluated model was trained: the intermediate save's step, else the run's steps."""
    return note.get("train_step") or checkpoint_step(note.get("model")) or (steps if steps is not None else run_steps()).get(note.get("run_id"))


def model_label(display: str, step: int | None, final: int | None) -> str:
    """Display name, with @12.5k for an intermediate save."""
    return f"{display}@{step_text(step)}" if step and step != final else display


# ---------------------------------------------------------------- evaluations
def run_settings_code(opts: dict) -> str:
    parts = [f"async{int(round(float(opts.get('chunk_size_threshold', 0.5)) * 10)):02d}" if opts.get("async_mode") else "sync"]
    if opts.get("rtc"):
        parts.append("rtc")
    if opts.get("smooth"):
        parts.append("smooth")
    if opts.get("diffusion_steps"):
        parts.append(f"ddim{opts['diffusion_steps']}")
    return "_".join(parts)


def eval_name(display: str, preset: dict | None, opts: dict, stamp: str, step: int | None = None) -> str:
    """step: an intermediate save's training step, written as -at12500 after the model name."""
    mode = f"p{len(proto.eval_positions(preset))}" if preset else "free"
    model = clean_name(display).lower() + (f"-at{step}" if step else "")
    return f"eval_{model}__{mode}_{run_settings_code(opts)}_{stamp}"

#!/usr/bin/env python3
"""Upload a recorded LeRobot dataset from the shared folder to the Hugging Face Hub.

  python omx_upload.py tkoo/1014_1930 --keywords strawberry,pick_place --description "..."
      -> <hf user>/omx_strawberry_1014_1930 (private), tags + description on the dataset page
  python omx_upload.py tkoo/1014_1930 --name my_set --public

The local folder keeps its date-time name. The Hub name gets the first keyword so it says what
the data is about. Keywords and the note saved at recording time (see notes_path) are reused
when none are given. Requires a one-time `hf auth login` (write token) on this account.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

os.environ.setdefault("HF_LEROBOT_HOME", str(omx_config.path("data_home")))

from huggingface_hub import HfApi  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

DATA_HOME = Path(os.environ["HF_LEROBOT_HOME"])
CAMERA_LABELS = {"front": "위에서 내려다보는 글로벌 카메라", "wrist": "팔로워 그리퍼에 달린 카메라"}


def notes_path(repo_id: str) -> Path:
    """Sidecar with keywords/description, kept outside the dataset folder."""
    owner, name = repo_id.split("/", 1)
    return DATA_HOME / owner / ".omx_notes" / f"{name}.json"


def clean_keywords(raw: str | list | None) -> list[str]:
    # Split on commas only, so "pick place" stays one keyword (pick_place)
    items = raw if isinstance(raw, list) else re.split(r"[,\n]+", raw or "")
    out = []
    for item in items:
        word = re.sub(r"[^a-z0-9_-]+", "_", str(item).lower()).strip("_-")
        if word and word not in out:
            out.append(word)
    return out[:8]


# Keywords are picked automatically from the task sentence (English) and the note (often Korean):
# the main object first (it goes into the Hub name), then the action, then other objects.
KO_OBJECTS = {
    "딸기": "strawberry", "사과": "apple", "바나나": "banana", "오렌지": "orange", "귤": "orange",
    "접시": "plate", "그릇": "bowl", "컵": "cup", "머그": "mug", "병": "bottle", "캔": "can",
    "상자": "box", "박스": "box", "블록": "block", "큐브": "cube", "공": "ball", "펜": "pen",
    "연필": "pencil", "가위": "scissors", "수건": "towel", "천": "cloth", "서랍": "drawer", "뚜껑": "lid",
}
ACTIONS = [  # (pattern in English or Korean, keyword)
    (r"pick|place|put|move|집|옮기|옮겨|놓|넣", "pick_place"),
    (r"stack|쌓", "stack"),
    (r"push|밀", "push"),
    (r"pour|붓", "pour"),
    (r"open|열", "open"),
    (r"close|닫", "close"),
    (r"sort|분류", "sort"),
]
EN_STOPWORDS = {
    "pick", "up", "place", "put", "move", "it", "them", "the", "a", "an", "and", "on", "onto", "in", "into",
    "to", "from", "of", "with", "at", "then", "back", "its", "is", "this", "that", "white", "red", "blue",
    "green", "black", "yellow", "small", "big", "large", "left", "right", "top", "bottom", "table", "robot",
    "arm", "gripper", "stack", "push", "pour", "open", "close", "sort", "grab", "take", "hold", "release",
}


def derive_keywords(tasks: list[str], note: str) -> list[str]:
    note = note or ""
    # Objects named in the description come first (in the order written), then nouns of the task sentence
    found = sorted((note.find(ko), en) for ko, en in KO_OBJECTS.items() if ko in note)
    found += [(i, w) for i, w in enumerate(re.findall(r"[a-z]+", note.lower())) if w not in EN_STOPWORDS and len(w) > 2]
    objects = []
    for _, en in found:
        if en not in objects:
            objects.append(en)
    for word in re.findall(r"[a-z]+", " ".join(tasks).lower()):
        if word not in EN_STOPWORDS and len(word) > 2 and word not in objects:
            objects.append(word)
    text = (note + " " + " ".join(tasks)).lower()
    actions = [kw for pattern, kw in ACTIONS if re.search(pattern, text)]
    return clean_keywords(objects[:1] + actions[:1] + objects[1:] + actions[1:])


def hub_name(local_repo_id: str, keywords: list[str], name: str | None) -> str:
    if not name:
        local = local_repo_id.rsplit("/", 1)[-1]
        name = "_".join(["omx", *keywords[:1], local])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.-")


def build_description(dataset: LeRobotDataset, local_repo_id: str, note: str, keywords: list[str]) -> str:
    meta = dataset.meta
    fps = meta.fps
    tasks = list(meta.tasks.index) if meta.tasks is not None else []
    cameras = [k.rsplit(".", 1)[-1] for k in meta.video_keys]
    seconds = meta.total_frames / fps if fps else 0
    owner, local = local_repo_id.split("/", 1)
    when = re.match(r"(\d{2})(\d{2})_(\d{2})(\d{2})$", local)
    camera_text = ", ".join(f"`{c}` {CAMERA_LABELS.get(c, '')}".strip() for c in cameras) or "-"
    recorded = f"{when.group(1)}월 {when.group(2)}일 {when.group(3)}:{when.group(4)}, " if when else ""
    lines = []
    if note:
        lines += [note.strip(), ""]
    lines += [
        "| 항목 | 내용 |",
        "|---|---|",
        "| 로봇 | ROBOTIS OMX-AI 팔로워 (5관절 + 그리퍼), 리더암 텔레오퍼레이션으로 시연 |",
        f"| 작업 | {' / '.join(tasks) if tasks else '-'} |",
        f"| 에피소드 | {meta.total_episodes}개, 총 {seconds / 60:.1f}분 ({meta.total_frames} 프레임, {fps}fps) |",
        f"| 카메라 | {camera_text} |",
        f"| 키워드 | {', '.join(keywords) if keywords else '-'} |",
        f"| 녹화 | {recorded}계정 {owner} (원본 `{local_repo_id}`) |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="local dataset, e.g. tkoo/1014_1930")
    parser.add_argument("--keywords", help="comma separated; default: picked from the task sentence and the description")
    parser.add_argument("--description", help="free text shown at the top of the dataset page")
    parser.add_argument("--name", help="Hub dataset name (default: omx_<first keyword>_<local name>)")
    parser.add_argument("--public", action="store_true", help="make the Hub dataset public (default: private)")
    args = parser.parse_args()

    root = DATA_HOME / args.repo_id
    if not (root / "meta/info.json").exists():
        print(f"[omx] upload failed: dataset not found: {root}", flush=True)
        return 2

    saved = {}
    if notes_path(args.repo_id).exists():
        saved = json.loads(notes_path(args.repo_id).read_text())
    note = args.description if args.description is not None else saved.get("description", "")

    try:
        user = HfApi().whoami()["name"]
    except Exception as exc:
        print(f"[omx] upload failed: not logged in to Hugging Face ({exc}). Run: hf auth login", flush=True)
        return 3

    dataset = LeRobotDataset(args.repo_id, root=root)
    if dataset.meta.total_episodes == 0:
        print("[omx] upload skipped: the dataset has no episodes", flush=True)
        return 4
    given = args.keywords if args.keywords is not None else saved.get("keywords")
    tasks = list(dataset.meta.tasks.index) if dataset.meta.tasks is not None else []
    keywords = clean_keywords(given) if given else derive_keywords(tasks, note)
    target = f"{user}/{hub_name(args.repo_id, keywords, args.name)}"
    print(f"[omx] uploading {args.repo_id} -> {target} ({'public' if args.public else 'private'}), tags: {keywords}", flush=True)
    description = build_description(dataset, args.repo_id, note, keywords)
    # push_to_hub() publishes under dataset.repo_id; point it at the Hub name without renaming the folder
    dataset.repo_id = target
    dataset.meta.repo_id = target
    dataset.push_to_hub(private=not args.public, tags=["omx", *keywords], dataset_description=description)
    print(f"[omx] uploaded https://huggingface.co/datasets/{target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

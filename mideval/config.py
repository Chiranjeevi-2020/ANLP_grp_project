"""Config loading, run directories and small I/O helpers shared by all stages."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "data" / "mideval" / "spec.json"
SPLITS = ("train", "dev", "test")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg["_path"] = str(path)
    return cfg


def load_spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


def run_dir(cfg) -> Path:
    d = ROOT / "results" / "mideval" / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def sub(cfg, *parts) -> Path:
    p = run_dir(cfg).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(*parts, n=10) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:n]


def config_hash(cfg) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return stable_hash(json.dumps(clean, sort_keys=True), n=12)


def run_id(cfg) -> str:
    """The run id is created once by build-data and stamped on every artifact."""
    p = sub(cfg, "manifest.json")
    if not p.exists():
        raise RuntimeError(f"{p} missing: run `build-data` first")
    return read_json(p)["run_id"]


def stamp(cfg, **extra) -> dict:
    return {"run_id": run_id(cfg), "config_hash": config_hash(cfg),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "created_unix": time.time(), **extra}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path) as f:
        return json.load(f)

"""Command-line entry point.

  python -m mideval.cli <stage> --config configs/full.json
stages: build-data | audit-data | gen-decoder-data | train-decoder | build-bank | smoke | audit-smoke |
        train-rl | evaluate | audit-final | all
"""
import argparse
import shutil
import time

from .config import load_config, load_spec, read_json, run_dir, sub, write_json

STAGES = ["build-data", "audit-data", "gen-decoder-data", "train-decoder", "build-bank", "smoke",
          "audit-smoke", "train-rl", "evaluate", "audit-final"]
NEEDS_LM = {"gen-decoder-data", "build-bank", "smoke", "evaluate"}

_LM = None


def get_lm(cfg):
    global _LM
    if _LM is None:
        from .lm import FrozenLM
        _LM = FrozenLM(cfg["model_name"], cfg["device"], cfg["dtype"])
    return _LM


def run_stage(stage, cfg, args):
    from . import audit, data, evaluate, pipeline
    spec = load_spec()
    lm = get_lm(cfg) if stage in NEEDS_LM else None
    if stage == "build-data":
        d = run_dir(cfg)
        if any(d.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"{d} is not empty; pass --overwrite (old run is archived, never mixed)")
            arch = d.parent / "_archive" / f"{cfg['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
            arch.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(arch))
            print(f"[cli] archived previous run to {arch}")
        data.build(cfg)
    elif stage.startswith("audit-"):
        audit.run(cfg, stage.split("-", 1)[1])
    elif stage == "gen-decoder-data":
        pipeline.gen_decoder_data(cfg, spec, lm)
    elif stage == "train-decoder":
        pipeline.train_decoders(cfg, spec)
    elif stage == "build-bank":
        pipeline.build_bank(cfg, spec, lm)
    elif stage == "smoke":
        pipeline.smoke(cfg, spec, lm)
    elif stage == "train-rl":
        pipeline.train_rl(cfg, spec)
    elif stage == "evaluate":
        evaluate.evaluate(cfg, spec, lm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for st in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        print(f"\n===== {st} ({cfg['run_name']}) =====", flush=True)
        run_stage(st, cfg, args)
        tp = sub(cfg, "timings.json")
        tim = read_json(tp) if tp.exists() else {}
        tim[st] = round(time.time() - t0, 1)
        write_json(tp, tim)
        print(f"===== {st} done in {tim[st]:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()

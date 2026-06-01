#!/usr/bin/env python
"""Pack/unpack experiment data for scp transfer between machines.

Reads the current USER CONFIGURATION from energy_betaM_experiment.py and
collects all associated files (samples, models, configs, results) into
data_dump/ with the same directory structure as the project root.

Pack on the source machine, scp data_dump/ to the destination, unpack.

Usage:
    python scripts/data_dump.py --pack
    python scripts/data_dump.py --pack --dry-run     # preview without copying
    python scripts/data_dump.py --unpack
    python scripts/data_dump.py --unpack --overwrite  # replace existing files
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS / "experiments"))
from energy_betaM_experiment import (  # noqa: E402
    DIMS, BETA_MS, SEEDS,
    _sample_run_name, _model_run_name, _experiment_dir,
    SAMPLE_DIR, MODEL_DIR, MODEL_SAMPLE_DIR,
    SAMPLING_CONFIGS_DIR, TRAINING_CONFIGS_DIR,
    ROOT,
)

DUMP_DIR = ROOT / "data_dump"


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 2


def _relevant_sources() -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []

    def add(p: Path) -> None:
        if p not in seen:
            seen.add(p)
            result.append(p)

    for seed in SEEDS:
        for dim in DIMS:
            for beta_m in BETA_MS:
                sample_run = _sample_run_name(dim, beta_m, seed)
                model_run  = _model_run_name(sample_run, seed)
                add(SAMPLE_DIR           / sample_run)
                add(MODEL_DIR            / model_run)
                add(MODEL_SAMPLE_DIR     / model_run)
                add(SAMPLING_CONFIGS_DIR / f"{sample_run}.yaml")
                add(TRAINING_CONFIGS_DIR / f"{model_run}.yaml")

    add(_experiment_dir())
    return result


def _copy_item(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def pack(dry_run: bool = False) -> None:
    sources = _relevant_sources()
    found   = [s for s in sources if s.exists()]
    missing = [s for s in sources if not s.exists()]

    print(f"Experiment dir : {_experiment_dir().relative_to(ROOT)}")
    print(f"Dump target    : {DUMP_DIR}")
    print(f"Items found: {len(found)}   missing: {len(missing)}\n")

    total_mb = 0.0
    for src in found:
        rel = src.relative_to(ROOT)
        if src.is_dir():
            mb = _dir_size_mb(src)
            total_mb += mb
            print(f"  DIR   {rel}  ({mb:.1f} MB)")
        else:
            mb = src.stat().st_size / 1024 ** 2
            total_mb += mb
            print(f"  FILE  {rel}  ({mb:.3f} MB)")
        if not dry_run:
            _copy_item(src, DUMP_DIR / rel)

    print(f"\nTotal: {total_mb:.1f} MB")

    if missing:
        print(f"\nMissing (not packed):")
        for m in missing:
            print(f"  {m.relative_to(ROOT)}")

    if dry_run:
        print("\n(dry run — nothing written)")
    else:
        print(f"\nDone. Transfer with:")
        print(f"  scp -r {DUMP_DIR} user@host:/path/to/project/data_dump/")
        print(f"  # or rsync for incremental updates:")
        print(f"  rsync -avz {DUMP_DIR}/ user@host:/path/to/project/data_dump/")


def unpack(overwrite: bool = False) -> None:
    if not DUMP_DIR.exists():
        print(f"No data_dump/ found at {DUMP_DIR}")
        return

    copied = skipped = 0
    for src in sorted(DUMP_DIR.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(DUMP_DIR)
        dst = ROOT / rel
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  COPY  {rel}")
        copied += 1

    print(f"\nDone. Copied {copied} files, skipped {skipped} existing"
          + (" (use --overwrite to replace)." if skipped else "."))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack experiment data into data_dump/ or unpack it back into the project."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack",   action="store_true", help="Collect files → data_dump/")
    group.add_argument("--unpack", action="store_true", help="data_dump/ → project directories")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Preview without copying (--pack only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing files (--unpack only)")
    args = parser.parse_args()

    if args.pack:
        pack(dry_run=args.dry_run)
    else:
        unpack(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

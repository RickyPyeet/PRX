from __future__ import annotations

import argparse
import csv
import random
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "ma-xu/fine-t2i"
SUBFOLDER = "curated"

FIRST_SHARD = 0
LAST_SHARD = 68

TRAIN_SIZE = 50_000
REFERENCE_SIZE = 10_000
TOTAL_REQUIRED = TRAIN_SIZE + REFERENCE_SIZE

SEED = 42


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if set(reader.fieldnames or []) != {"sample_id", "shard"}:
            raise RuntimeError(f"Invalid manifest columns: {reader.fieldnames}")

        rows = list(reader)

    ids = [row["sample_id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample IDs in candidate manifest.")

    return rows


def write_manifest_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "shard"],
        )
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(path)


def scan_shard(
    tar_path: Path,
    shard_name: str,
) -> list[dict[str, str]]:
    jpg_ids = set()
    json_ids = set()

    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue

            path = Path(member.name)
            suffix = path.suffix.lower()

            if suffix in {".jpg", ".jpeg"}:
                jpg_ids.add(path.stem)

            elif suffix == ".json":
                json_ids.add(path.stem)

    if jpg_ids != json_ids:
        raise RuntimeError(
            f"{shard_name}: JPEG/JSON IDs do not match "
            f"(JPEG={len(jpg_ids):,}, JSON={len(json_ids):,})"
        )

    return [
        {
            "sample_id": sample_id,
            "shard": shard_name,
        }
        for sample_id in sorted(jpg_ids)
    ]


def collect(
    start_shard: int,
    num_shards: int,
    data_dir: Path,
    manifest_path: Path,
    revision: str | None,
) -> None:
    rows = load_manifest(manifest_path)

    existing_ids = {row["sample_id"] for row in rows}
    existing_shards = {row["shard"] for row in rows}

    before = len(rows)
    new_rows = []

    print(f"Candidates before: {before:,}")

    for idx in range(start_shard, start_shard + num_shards):
        shard_name = f"train-{idx:06d}.tar"

        if shard_name in existing_shards:
            print(f"[{idx:06d}] already present -> skipping")
            continue

        filename = f"{SUBFOLDER}/{shard_name}"

        print(f"[{idx:06d}] downloading...")

        tar_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                local_dir=data_dir,
            )
        )

        shard_rows = scan_shard(tar_path, shard_name)
        shard_ids = {row["sample_id"] for row in shard_rows}

        duplicates = existing_ids & shard_ids

        if duplicates:
            print(
                f"         {len(duplicates):,} duplicate IDs -> skipping"
            )

        shard_rows = [
            row
            for row in shard_rows
            if row["sample_id"] not in existing_ids
        ]

        shard_ids = {row["sample_id"] for row in shard_rows}

        size_gib = tar_path.stat().st_size / (1024**3)

        print(
            f"         {len(shard_rows):,} unique samples | "
            f"{size_gib:.2f} GiB"
        )

        new_rows.extend(shard_rows)
        existing_ids.update(shard_ids)
        existing_shards.add(shard_name)

    final_rows = rows + new_rows

    if len(final_rows) != before + len(new_rows):
        raise RuntimeError("Candidate count mismatch.")

    write_manifest_atomic(manifest_path, final_rows)

    # Reload what was actually written.
    saved_rows = load_manifest(manifest_path)

    if len(saved_rows) != len(final_rows):
        raise RuntimeError("Saved manifest count mismatch.")

    print("\n" + "=" * 50)
    print(f"New candidates:   {len(new_rows):,}")
    print(f"Total candidates: {len(saved_rows):,}")
    print(f"Unique IDs:       {len(existing_ids):,}")
    print("=" * 50)

    if len(saved_rows) >= TOTAL_REQUIRED:
        print("\nEnough candidates for the 50k/10k split.")


def split(
    manifest_path: Path,
    output_dir: Path,
) -> None:
    rows = load_manifest(manifest_path)

    if len(rows) < TOTAL_REQUIRED:
        raise RuntimeError(
            f"Need {TOTAL_REQUIRED:,} candidates; "
            f"found {len(rows):,}."
        )

    # Canonical ordering before seeded randomization.
    rows = sorted(rows, key=lambda row: row["sample_id"])

    rng = random.Random(SEED)
    rng.shuffle(rows)

    train_rows = rows[:TRAIN_SIZE]
    reference_rows = rows[
        TRAIN_SIZE:TRAIN_SIZE + REFERENCE_SIZE
    ]

    train_ids = {row["sample_id"] for row in train_rows}
    reference_ids = {
        row["sample_id"] for row in reference_rows
    }

    assert len(train_rows) == TRAIN_SIZE
    assert len(reference_rows) == REFERENCE_SIZE
    assert not train_ids & reference_ids

    output_dir.mkdir(parents=True, exist_ok=True)

    write_manifest_atomic(
        output_dir / "fine_t2i_train_50k.csv",
        train_rows,
    )

    write_manifest_atomic(
        output_dir / "fine_t2i_reference_10k.csv",
        reference_rows,
    )

    print(f"Train:     {len(train_rows):,}")
    print(f"Reference: {len(reference_rows):,}")
    print(f"Overlap:   {len(train_ids & reference_ids):,}")
    print(f"Seed:      {SEED}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--start-shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, required=True)
    collect_parser.add_argument("--revision", type=str, default=None)
    collect_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/data/fine-t2i"),
    )
    collect_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/content/manifests/fine_t2i_candidates.csv"),
    )

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--manifest", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "collect":
        collect(
            args.start_shard,
            args.num_shards,
            args.data_dir,
            args.manifest,
            args.revision,
        )
    else:
        split(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
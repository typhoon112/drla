"""Build the self-contained one-row packet bundle used by the migration smoke test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "p2c_musique_calibration_2hop__38555_442521"
SOURCE_MANIFEST = REPO_ROOT / (
    "outputs/p2_phase_c_manifests/"
    "musique_calibration_manifest_200_seed20260601/manifest.json"
)
SOURCE_ONLINE_INPUTS = REPO_ROOT / (
    "outputs/p2_phase_c_control_inputs/"
    "musique_calibration_controls_200_seed20260601_v1_strict_wrong/online_inputs.jsonl"
)
SOURCE_PACKETS = REPO_ROOT / (
    "outputs/p3_dream_latent_packets/"
    "dream_textmas_selected_suffix_tensor_smoke1_packets_20260617"
)
BUNDLE_DIR = REPO_ROOT / "migration/post_d55_smoke"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    bundle_packets = BUNDLE_DIR / "packets"
    bundle_hidden = BUNDLE_DIR / "hidden_refs"
    bundle_hidden.mkdir(parents=True, exist_ok=True)
    bundle_packets.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    selected_samples = [row for row in manifest["samples"] if row.get("sample_id") == SAMPLE_ID]
    if len(selected_samples) != 1:
        raise ValueError(f"expected exactly one manifest sample for {SAMPLE_ID}")
    manifest["samples"] = selected_samples
    manifest["archive_smoke_subset"] = True
    (BUNDLE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_online = [
        row
        for row in read_jsonl(SOURCE_ONLINE_INPUTS)
        if row.get("sample_id") == SAMPLE_ID and row.get("condition") == "textmas_matched"
    ]
    if len(selected_online) != 1:
        raise ValueError(f"expected exactly one matched online row for {SAMPLE_ID}")
    write_jsonl(BUNDLE_DIR / "online_inputs.jsonl", selected_online)

    packets = read_jsonl(SOURCE_PACKETS / "packets.jsonl")
    if len(packets) != 2:
        raise ValueError("expected exactly two source packets")
    for packet in packets:
        source_tensor = Path(packet["hidden_ref"])
        target_tensor = bundle_hidden / source_tensor.name
        shutil.copy2(source_tensor, target_tensor)
        packet["hidden_ref"] = target_tensor.relative_to(REPO_ROOT).as_posix()
    write_jsonl(bundle_packets / "packets.jsonl", packets)

    groups = read_jsonl(SOURCE_PACKETS / "packet_groups.jsonl")
    if len(groups) != 1:
        raise ValueError("expected exactly one packet group")
    write_jsonl(bundle_packets / "packet_groups.jsonl", groups)

    summary = {
        "status": "pass",
        "sample_id": SAMPLE_ID,
        "num_manifest_samples": 1,
        "num_online_rows": 1,
        "num_packets": 2,
        "num_packet_groups": 1,
        "hidden_refs_are_repo_relative": True,
        "source_note": "D6 selected-suffix smoke packet; retained only as a D7 interface smoke fixture.",
    }
    (BUNDLE_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

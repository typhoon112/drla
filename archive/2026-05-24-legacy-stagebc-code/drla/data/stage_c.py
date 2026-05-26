"""Stage C latent cache datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class LatentCacheDataset(Dataset[dict[str, Any]]):
    def __init__(self, latent_dir: str | Path, *, max_samples: int | None = None) -> None:
        paths = sorted(Path(latent_dir).glob("*.pt"))
        if max_samples is not None:
            paths = paths[:max_samples]
        if not paths:
            raise ValueError(f"No latent cache files found in {latent_dir}")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = torch.load(self.paths[index], map_location="cpu")
        return {
            "id": item["id"],
            "z_blocks": item["z_blocks"].float(),
            "question_ids": item["question_ids"].long().tolist(),
            "target_ids": item["target_ids"].long().tolist(),
            "answer_norm": str(item["answer_norm"]),
            "b_star": int(item["B_star"]),
            "block_mask": item["block_mask"].long(),
            "noop_mask": item["noop_mask"].long(),
        }


class LatentCacheCollator:
    def __init__(self, *, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        question_ids, question_mask = _pad_sequences(
            [row["question_ids"] for row in rows], pad_id=self.pad_id
        )
        target_ids, target_mask = _pad_sequences(
            [row["target_ids"] for row in rows], pad_id=self.pad_id
        )
        return {
            "ids": [row["id"] for row in rows],
            "z_blocks": torch.stack([row["z_blocks"] for row in rows]),
            "question_ids": question_ids,
            "question_mask": question_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "answer_norms": [row["answer_norm"] for row in rows],
            "b_star": torch.tensor([row["b_star"] for row in rows], dtype=torch.long),
            "block_mask": torch.stack([row["block_mask"] for row in rows]),
            "noop_mask": torch.stack([row["noop_mask"] for row in rows]),
        }


def _pad_sequences(values: Sequence[Sequence[int]], *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(value) for value in values)
    output = torch.full((len(values), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(values), length), dtype=torch.long)
    for i, value in enumerate(values):
        if value:
            output[i, : len(value)] = torch.tensor(value, dtype=torch.long)
            mask[i, : len(value)] = 1
    return output, mask

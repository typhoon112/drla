"""Stage B data utilities for deterministic reasoning latent experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class StageBExample:
    id: str
    question_ids: list[int]
    target_ids: list[int]
    answer_ids: list[int]
    answer_norm: str
    b_star: int
    b_max: int
    block_size: int


class VocabularyMapper:
    """Map original tokenizer ids into a compact local vocabulary for smoke runs."""

    def __init__(self, original_token_ids: Sequence[int], *, add_unk: bool = True) -> None:
        special_count = 2 if add_unk else 1
        self.pad_id = 0
        self.unk_id = 1 if add_unk else 0
        unique_ids = sorted(set(original_token_ids))
        self.local_to_original = [-1] * special_count + unique_ids
        self.original_to_local = {
            token_id: i + special_count for i, token_id in enumerate(unique_ids)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.local_to_original)

    def encode(self, token_ids: Sequence[int]) -> list[int]:
        return [self.original_to_local.get(token_id, self.unk_id) for token_id in token_ids]

    def decode(self, local_ids: Sequence[int]) -> list[int]:
        result: list[int] = []
        for local_id in local_ids:
            if 0 <= local_id < len(self.local_to_original):
                original_id = self.local_to_original[local_id]
                if original_id >= 0:
                    result.append(original_id)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "local_to_original": self.local_to_original,
        }


class StageBDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[StageBExample],
        *,
        vocab_mapper: VocabularyMapper | None = None,
        max_answer_len: int = 16,
    ) -> None:
        self.examples = list(examples)
        self.vocab_mapper = vocab_mapper
        self.max_answer_len = max_answer_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        question_ids = _map_ids(example.question_ids, self.vocab_mapper)
        target_ids = _map_ids(example.target_ids, self.vocab_mapper)
        answer_ids = _map_ids(example.answer_ids[: self.max_answer_len], self.vocab_mapper)
        return {
            "id": example.id,
            "question_ids": question_ids,
            "target_ids": target_ids,
            "answer_ids": answer_ids,
            "answer_norm": example.answer_norm,
            "b_star": min(example.b_star, example.b_max),
            "b_max": example.b_max,
            "block_size": example.block_size,
            "original_question_ids": example.question_ids,
            "original_target_ids": example.target_ids,
            "original_answer_ids": example.answer_ids,
        }


def load_stage_b_examples(
    data_dir: str | Path,
    split: str,
    *,
    tokenizer: Any,
    b_max: int | None = None,
    block_size: int = 16,
    max_samples: int | None = None,
) -> list[StageBExample]:
    """Load Stage A tokenized JSONL rows and attach answer token ids."""
    data_path = Path(data_dir)
    tokenized_rows = _read_jsonl(data_path / f"gsm8k_{split}.tokenized.jsonl")
    raw_rows = {row["id"]: row for row in _read_jsonl(data_path / f"gsm8k_{split}.jsonl")}
    if max_samples is not None:
        tokenized_rows = tokenized_rows[:max_samples]

    examples: list[StageBExample] = []
    for row in tokenized_rows:
        row_b_max = int(b_max or row["B_max"])
        row_block_size = int(block_size)
        answer_norm = str(row["answer_norm"])
        answer_ids = tokenizer.encode(answer_norm, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Empty answer tokenization for {row['id']}: {answer_norm!r}")
        if row["id"] not in raw_rows:
            raise ValueError(f"Missing raw Stage A row for {row['id']}")
        examples.append(
            StageBExample(
                id=row["id"],
                question_ids=list(row["question_ids"]),
                target_ids=list(row["target_ids"]),
                answer_ids=answer_ids,
                answer_norm=answer_norm,
                b_star=int(row["B_star"]),
                b_max=row_b_max,
                block_size=row_block_size,
            )
        )
    return examples


def build_local_vocab(examples: Iterable[StageBExample]) -> VocabularyMapper:
    token_ids: list[int] = []
    for example in examples:
        token_ids.extend(example.question_ids)
        token_ids.extend(example.target_ids)
        token_ids.extend(example.answer_ids)
    return VocabularyMapper(token_ids)


class StageBCollator:
    def __init__(self, *, pad_id: int, b_max: int, block_size: int, max_answer_len: int) -> None:
        self.pad_id = pad_id
        self.b_max = b_max
        self.block_size = block_size
        self.capacity = b_max * block_size
        self.max_answer_len = max_answer_len

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        question_ids, question_mask = _pad_sequences(
            [row["question_ids"] for row in rows], pad_id=self.pad_id
        )
        target_input_ids, target_mask = _pad_sequences(
            [row["target_ids"][: self.capacity] for row in rows],
            pad_id=self.pad_id,
            max_len=self.capacity,
        )
        answer_input_ids, answer_mask = _pad_sequences(
            [row["answer_ids"][: self.max_answer_len] for row in rows],
            pad_id=self.pad_id,
            max_len=self.max_answer_len,
        )

        target_labels = target_input_ids.clone()
        target_labels[target_mask == 0] = -100
        answer_labels = answer_input_ids.clone()

        b_star = torch.tensor(
            [min(int(row["b_star"]), self.b_max) for row in rows], dtype=torch.long
        )
        block_positions = torch.arange(self.b_max).unsqueeze(0)
        block_mask = (block_positions < b_star.unsqueeze(1)).long()
        noop_mask = 1 - block_mask

        return {
            "ids": [row["id"] for row in rows],
            "question_ids": question_ids,
            "question_mask": question_mask,
            "target_input_ids": target_input_ids,
            "target_labels": target_labels,
            "target_mask": target_mask,
            "answer_input_ids": answer_input_ids,
            "answer_labels": answer_labels,
            "answer_mask": answer_mask,
            "answer_norms": [row["answer_norm"] for row in rows],
            "b_star": b_star,
            "block_mask": block_mask,
            "noop_mask": noop_mask,
            "original_question_ids": [row["original_question_ids"] for row in rows],
            "original_target_ids": [row["original_target_ids"] for row in rows],
            "original_answer_ids": [row["original_answer_ids"] for row in rows],
        }


def _map_ids(token_ids: Sequence[int], vocab_mapper: VocabularyMapper | None) -> list[int]:
    if vocab_mapper is None:
        return list(token_ids)
    return vocab_mapper.encode(token_ids)


def _pad_sequences(
    values: Sequence[Sequence[int]], *, pad_id: int, max_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    length = max_len or max(len(value) for value in values)
    output = torch.full((len(values), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(values), length), dtype=torch.long)
    for i, value in enumerate(values):
        clipped = list(value[:length])
        if clipped:
            output[i, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            mask[i, : len(clipped)] = 1
    return output, mask


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


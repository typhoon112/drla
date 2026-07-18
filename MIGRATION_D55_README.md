# DRLA D5.5-preserving migration archive

本归档用于迁移 DRLA 工作区。边界是：

- 完整保留 CoLA/P0/P1/P2 历史线和 P3 Dream D0–D5.5；
- P3 D5.5 之后只保留代码、结论、配置、轻量日志、V7 可运行 checkpoint 和自包含 smoke fixture；
- 不把 D7 任一结果误写成已经通过 held-out gate；
- 打包过程不删除源工作区中的任何文件。

## 归档内容

完整保留：

- `.git` 以及当前工作区全部 tracked/untracked 代码和文档；
- `models/` 中的 Cola、Dream、Qwen 模型权重和 tokenizer/config；
- `.conda/drla-mvp`、环境配置文件和本地 Hugging Face cache；
- `archive/`、CoLA/P0/P1/P2 的输出、权重、config、metrics、manifest 和日志；
- P3 Dream 的 D0–D5.5 protocol、trace summary、frontier、readiness student checkpoint、policy audit；
- D5.5 后所有非大型 tensor 证据、V7 `best_checkpoint.pt` 与 `last_checkpoint.pt`；
- `migration/post_d55_smoke/` 自包含的一行 MuSiQue、A/B packet 与 tensor fixture；
- `outputs/p3_archive_smoke_runs/v7_one_row_20260718/` 已通过的 smoke 输出。

有意不保留：

- D6/D7 full-suffix、heldout800、train2000、validdiag 等大规模 `.pt` tensor；
- V1/V2、soft-prefix、text-interface、D7.16、D7.18、D7.19 等失败分支的大型 checkpoint；
- 这些被省略资产的结论、metrics、summary、generation 和配置仍然保留。

精确文件级清单见 `migration/ARCHIVE_INVENTORY.tsv`，分类汇总见
`migration/ARCHIVE_INVENTORY_SUMMARY.json`。D5.5 后的研究结论见
`docs/current/POST_D55_ARCHIVE_2026-07-18.md`。

## 解压与验证

压缩包内部以 `drla/` 为根目录。在目标父目录运行：

```bash
tar --use-compress-program=unzstd -xf drla_d55_migration_20260718.tar.zst
cd drla
source scripts/activate_conda.sh
python scripts/verify_d55_migration.py
```

完整 GPU smoke：

```bash
source scripts/activate_conda.sh
python scripts/verify_d55_migration.py \
  --full-gpu \
  --output-dir outputs/p3_archive_smoke_runs/restored_v7_one_row
```

验收条件是：

- 所有模型 index 指向的 shard 都存在；
- D5 readiness 和 V7 checkpoint 可被 PyTorch 加载；
- smoke bundle 恰好包含一个样本、A/B 两个 packet，tensor shape 与引用一致；
- packet 不含 gold/scorer/decoder 等禁止在线字段；
- 完整 smoke 对 `no_message/matched/shuffled_row/zero` 产生 4 条成功记录。

Smoke 只证明迁移后的工程链路可运行，不证明 matched 优于 control。D7 held-out gate
仍然失败，不能据此进入 D8。

## 环境迁移说明

`.conda/drla-mvp` 被保留以最大限度保存原环境，但 Conda 环境目录可能含原路径前缀。
如果目标路径不是 `/data1/luyifei/drla`，优先根据
`configs/drla-mvp.environment.yml` 和 requirements 文件重建环境，再运行上述验证。

归档同时包含 SHA-256 校验文件。迁移完成后应先执行：

```bash
sha256sum -c drla_d55_migration_20260718.tar.zst.sha256
```

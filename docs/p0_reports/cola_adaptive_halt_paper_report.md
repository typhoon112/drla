# Safe Adaptive Halting for Block-wise Cola Rollouts

> Status: English historical stub. The Chinese report `/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md` is the canonical P0 paper-style report. The pre-slim English draft is archived at `/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/cola_adaptive_halt_paper_report.md`.

## Abstract

This report summarizes the P0 adaptive-halting evidence for official Cola block-wise rollouts. The core result is not benchmark-accuracy improvement. The core result is that non-gold readiness and prediction-change signals can reduce latent block usage while matching the conservative prediction-stability baseline under the current protocol.

## Main Result

```text
method:
  joint-readiness + prediction-change risk + riskcap04

protocol:
  official Cola 8 tasks
  full prepared split
  b64 / bs12 / t16
  seeds 66, 67, 68

weighted micro accuracy:
  21.596% +/- 0.030%

average blocks:
  2.118 +/- 0.010 / 4

observed losses vs prediction-stability:
  0
```

## Canonical Source

Use the Chinese canonical report for details:

```text
/data1/luyifei/drla/docs/p0_reports/cola_adaptive_halt_paper_report_zh.md
```

Use the archived English draft only if English phrasing is needed:

```text
/data1/luyifei/drla/docs/full_history/2026-05-29_pre_slim/cola_adaptive_halt_paper_report.md
```

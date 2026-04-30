# Scientific Report
## Mamba-360 Ghost Futures (Rebuilt Multi-Match Baseline)

## 1. Objective

Build a reproducible, open-data football world-model MVP that predicts and samples plausible 5-event futures from:
- Event history up to time `t`
- StatsBomb 360 freeze-frame context at events

The system must produce both:
- Quantitative evaluation metrics
- Qualitative ghost-future pitch visualizations

## 2. Data Regime

### Source
- StatsBomb Open Data via `statsbombpy`
- Competition used for this run: Bundesliga 2023/24
- `competition_id=9`, `season_id=281`

### Coverage
- Selected 20 matches with non-empty 360 frames
- Robust filtering by `match_status_360 == available` when available in metadata

### Why This Matters
The previous failure mode (single-match training) invalidates quantitative claims. This rebuild enforces a multi-match training regime.

## 3. Target Formulation

For each target event `i`, input context is events `[i-K, ..., i-1]`:

- `next_action_type_id` (multiclass)
- `next_zone_id` (96 classes; 12x8 pitch grid)
- `next_outcome_binary`
- `shot_next_5_binary`
- `turnover_next_3_binary`

## 4. Feature Construction

### Event Features
- Categorical: event type, team, player, play pattern, body part, outcome
- Continuous: x/y, end_x/end_y, deltas, distance, minute/second, pressure, possession-change indicators

### 360 Features
- Counts: visible players/teammates/opponents/keepers/actor
- Distances to ball: nearest and mean teammate/opponent distances
- Local pressure counts within 5 and 10 units
- Width/depth statistics
- Visible-area polygon size and fraction of pitch

### Robustness Layer
Safe extractors handle:
- Flattened versus nested StatsBomb fields
- Missing optional objects
- Malformed or absent locations
- Alternate 360 event-id column names

## 5. Model

### Architecture
- Event encoder:
  - Embeddings for each categorical field
  - MLP for continuous features
  - Projection to `d_model`
- Sequence backbone:
  - `Mamba` if importable
  - GRU fallback otherwise
- Multi-task heads:
  - Action logits
  - Zone logits
  - Outcome, shot-next-5, turnover-next-3 logits

### Runtime Backend
This run used GRU fallback (`Mamba backend unavailable; falling back to GRU.`).

## 6. Experimental Configuration

Config file: `configs/research.yaml`

Key parameters:
- `max_matches: 20`
- `context_length: 20`
- `epochs: 6`
- `batch_size: 64`
- `lr: 8e-4`

Split policy:
- By `match_id`, not by rows (prevents leakage across train/val/test from same match)

## 7. Quantitative Results

Evaluation file:
- `outputs/metrics/research_eval_metrics.json`

Metrics:
- `action_accuracy = 0.2827`
- `action_top3_accuracy = 0.8037`
- `zone_accuracy = 0.0150`
- `outcome_accuracy = 0.9449`
- `shot_next_5_auc = 0.5020`
- `turnover_next_3_auc = 0.5006`

### Interpretation
1. Action top-3 is substantially above top-1, indicating uncertainty concentration rather than single-mode confidence.
2. Zone accuracy is low (hard 96-way classification under sparse event-level supervision).
3. Future-risk heads (`shot_next_5`, `turnover_next_3`) are near random ranking in this baseline, indicating current architecture/objective is insufficient for those targets without further redesign.

## 8. Qualitative Result

Case study artifact:
- `outputs/case_studies/research_case_study.png`

This visualization remains valuable because it exposes:
- Whether trajectories are spatially coherent
- Whether sampled futures collapse to repetitive low-information patterns
- Whether predicted branches align with tactical context

## 9. Scientific Assessment

### What Is Achieved
- End-to-end reproducible pipeline from open-data ingestion to figure generation
- Multi-match training regime (20 matches) with robust missing-data handling
- Reliable experiment scripts and deterministic split behavior

### What Is Not Yet Achieved
- Strong predictive performance on spatial and binary future-risk heads
- Competitive world-model quality for shot/turnover anticipation

## 10. Immediate Technical Next Steps

1. Replace handcrafted 360 aggregation with learned set encoder (DeepSets/attention over visible players).
2. Introduce action-conditioned zone prediction (factorized or hierarchical heads).
3. Rebalance multitask loss using calibrated uncertainty weighting instead of fixed scalar weights.
4. Add calibration diagnostics (reliability curves, ECE) for binary heads.
5. Expand dataset size and run per-competition generalization checks.
6. Add explicit ablations:
   - Event-only vs Event+360
   - Short vs long context
   - GRU vs Mamba (where installable)

## 11. Reproducibility Commands

```bash
python scripts/01_cache_statsbomb_data.py --config configs/research.yaml --competition-id 9 --season-id 281
python scripts/02_build_dataset.py --config configs/research.yaml
python scripts/03_train_model.py --config configs/research.yaml --processed-path data/processed/dataset.pt --output outputs/checkpoints/research_model.pt
python scripts/04_evaluate_model.py --config configs/research.yaml --checkpoint outputs/checkpoints/research_model_best.pt --output outputs/metrics/research_eval_metrics.json
python scripts/05_make_case_study.py --config configs/research.yaml --checkpoint outputs/checkpoints/research_model_best.pt --output outputs/case_studies/research_case_study.png
```

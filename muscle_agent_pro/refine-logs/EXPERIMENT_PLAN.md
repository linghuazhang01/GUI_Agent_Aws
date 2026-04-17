# Experiment Plan: GAAGA (Grounding-Aware Adaptive GUI Agent)

---

## Experiment Block 1: Grounding Redundancy Characterization

**Goal**: Quantify the problem. How often do grounding calls repeat on visually similar states?

### Run 1.1: Logging baseline
- **What**: Run HIPPO Agent on 50 OSWorld tasks with full logging
- **Metrics**: Per grounding call: (screenshot_pHash, element_description_hash, coords, confidence, step_idx, prev_action_type)
- **Output**: `redundancy_analysis.json`
- **GPU**: 1x A100 (for VLM inference), ~4 hours
- **Success criteria**: Redundancy rate > 15% to justify caching approach

### Run 1.2: Tool usage profiling
- **What**: Same runs, but log tool selections per step
- **Metrics**: Per step: (tool_name, tool_input_keys, mode_label)
- **Output**: `tool_usage_profile.json`
- **GPU**: Included in Run 1.1
- **Success criteria**: > 30% of steps use only a subset of available tools

---

## Experiment Block 2: Step Type Predictor (STP) Training & Evaluation

### Run 2.1: Data labeling
- **What**: Label each step from Block 1 with optimal execution mode
- **Modes**: gui_fresh, gui_cached, code_task, feasibility, done_check
- **No GPU needed**: Post-processing of Block 1 data

### Run 2.2: STP training
- **What**: Train BERT-tiny classifier on labeled trajectories
- **Inputs**: (prev_N_action_names, CLIP_embedding_of_screenshot, task_instruction_embedding)
- **Output**: Mode distribution over K=5 classes
- **GPU**: 1x GPU, ~1 hour
- **Success criteria**: Top-1 accuracy > 80%, Top-2 accuracy > 95%

### Run 2.3: STP ablations
- Ablation A: Remove CLIP embedding (text-only)
- Ablation B: Remove action history (observation-only)
- Ablation C: Random routing (lower bound)
- Ablation D: Oracle routing (upper bound, uses ground truth labels)

---

## Experiment Block 3: Grounding Validity Predictor (GVP) Training

### Run 3.1: Training data generation
- **What**: From Block 1 logs, create positive/negative pairs
- Positive: same description, coords < 5px apart, UI visually similar
- Negative: same description, coords > 10px apart, or UI visually different
- **GPU**: Included in Block 1

### Run 3.2: GVP training
- **What**: 2-layer MLP on (CLIP_emb_cached, CLIP_emb_current, text_emb, confidence, steps_since_cache)
- **GPU**: 1x GPU, ~30 min
- **Success criteria**: AUC-ROC > 0.85 on held-out pairs

### Run 3.3: GVP threshold analysis
- Sweep confidence threshold from 0.5 to 0.95
- Plot: cache_hit_rate vs. grounding_error_rate at each threshold
- Find operating point that maximizes hit rate while keeping error < 2%

---

## Experiment Block 4: End-to-End Evaluation

### Run 4.1: Full OSWorld benchmark — GAAGA vs. Baseline
- **What**: Run both agents on all 369 OSWorld test tasks
- **Configurations**:
  1. HIPPO Agent baseline (current SOTA)
  2. GAAGA (STP + GVP + adaptive tools)
- **Runs per config**: 3 (for confidence intervals)
- **GPU**: 2x A100, ~20 hours per run × 3 = 60 GPU-hours
- **Success criteria**: Task completion rate >= baseline (74.5%), with 20%+ latency reduction

### Run 4.2: Ablation study
- **Configurations**:
  1. Baseline (no cache, no STP)
  2. STP only (adaptive tools, no grounding cache)
  3. GVP only (heuristic tool selection, learned cache)
  4. Heuristic cache (SSIM threshold, no GVP)
  5. Full GAAGA (STP + GVP)
- **GPU**: ~100 GPU-hours total
- **Success criteria**: Full GAAGA > any single component

### Run 4.3: Cross-benchmark evaluation
- **What**: Evaluate on 1 additional benchmark (AndroidWorld or WebArena)
- **GPU**: ~20 GPU-hours
- **Success criteria**: Consistent improvements across benchmarks

---

## Experiment Block 5: Analysis

### Run 5.1: Failure case analysis
- **What**: Manually analyze all cases where GAAGA performs worse than baseline
- **Categories**: cache-induced errors, STP misrouting, cascading failures
- **No GPU**: Manual analysis

### Run 5.2: Efficiency analysis
- **Metrics**: Total VLM tokens, total grounding calls, wall-clock time, per-episode cost
- **Breakdown**: Per application category (LibreOffice, Chrome, GIMP, etc.)

### Run 5.3: Cache behavior analysis
- **Metrics**: Hit rate by application, hit rate by step position, hit rate by grounding confidence
- **Visualization**: Heatmap of cache utilization across task episodes

---

## Run Order & Dependencies

```
Block 1 (Week 1)
  ├── Run 1.1 (logging) ──→ Run 1.2 (profiling)
  │
  ↓
Block 2 (Week 2)          Block 3 (Week 2, parallel)
  ├── Run 2.1 (label)       ├── Run 3.1 (pair generation)
  ├── Run 2.2 (train STP)   ├── Run 3.2 (train GVP)
  └── Run 2.3 (ablate STP)  └── Run 3.3 (threshold sweep)
  │
  ↓
Block 4 (Weeks 3-4)
  ├── Run 4.1 (full benchmark)
  ├── Run 4.2 (ablations) ──→ depends on 4.1
  └── Run 4.3 (cross-benchmark) ──→ depends on 4.1
  │
  ↓
Block 5 (Week 4)
  └── Runs 5.1-5.3 (analysis, no GPU)
```

## Total GPU Budget

| Block | GPU Hours |
|-------|-----------|
| Block 1 | 4 |
| Block 2 | 1 |
| Block 3 | 1 |
| Block 4 | ~140 |
| Block 5 | 0 |
| **Total** | **~146 GPU-hours** |

## First 3 Runs to Launch

1. **Run 1.1**: Logging baseline on 50 OSWorld tasks (validates problem existence)
2. **Run 2.2**: STP training on collected data (first ML component)
3. **Run 3.2**: GVP training on paired grounding data (second ML component)

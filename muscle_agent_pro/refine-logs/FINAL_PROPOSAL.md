# Final Proposal: Grounding-Aware Adaptive GUI Agent (GAAGA)

**Problem Anchor**: Multi-step GUI agents waste 30-60% of VLM inference on redundant grounding calls and irrelevant tool descriptions, yet no prior work systematically characterizes or exploits this redundancy with a learned adaptive policy.

---

## 1. Problem Statement

GUI agents like HIPPO Agent (OSWorld SOTA 74.5%) face two systematic inefficiencies at every execution step:

1. **Grounding Redundancy**: The VLM grounding model is invoked to convert `(screenshot, text_description)` → `(x, y)` coordinates. In multi-step tasks within the same application (e.g., editing a spreadsheet), many grounding calls target the same visual element across steps that don't change that element's position. Yet every call runs full VLM inference.

2. **Tool Description Overhead**: All tool schemas (CodeAgent, SubAgent, InfeasibleAgent, UI actions, web tools) are exposed to the VLM at every step, even when only 2-3 tools are relevant. For a simple "click the Save button" step, exposing 15+ tool descriptions wastes ~1000 tokens and confuses selection.

**Key insight**: These are not independent problems — they share a common structure: **each execution step has a latent "step type" that determines both which tools are needed AND whether grounding can be reused**. If we can predict the step type from cheap signals (previous actions, current observation), we can simultaneously reduce tool overhead and skip redundant grounding.

---

## 2. Method Thesis

**One sentence**: Train a lightweight Step Type Predictor (STP) that, given cheap trajectory signals, routes each step to one of K adaptive execution modes — each mode carries a different tool subset and grounding strategy (fresh VLM call vs. cached coordinates vs. OCR fallback).

### 2.1 Step Type Predictor (STP)

A small classifier (BERT-tiny or distilled from VLM trajectory data) that takes:
- Previous N actions (tool names + abbreviated inputs)
- Current screenshot embedding (from a frozen CLIP encoder, NOT the full VLM)
- Task instruction embedding

And outputs a distribution over K execution modes:

| Mode | Tools Exposed | Grounding Strategy | When Activated |
|------|--------------|-------------------|----------------|
| `gui_fresh` | UI actions only | Full VLM grounding | Need to locate a new UI element |
| `gui_cached` | UI actions only | Cache lookup + uncertainty check | Repeating action on unchanged UI |
| `code_task` | CodeAgent + bash | N/A (no GUI grounding) | File/data/code operations |
| `feasibility` | Infeasible tools + web | Full VLM grounding | Checking task feasibility |
| `done_check` | done, fail, VerificationAgent | Cache + screenshot diff | Verifying task completion |

### 2.2 Learned Cache Policy (Not Heuristic)

**This is the core ML contribution** — replacing reviewer-criticized SSIM thresholds with a learned policy.

Instead of `hash(screenshot_crop + app_state)` with arbitrary SSIM thresholds, we train a **Grounding Validity Predictor (GVP)**:
- Input: `(cached_screenshot_embedding, current_screenshot_embedding, cached_text_description, current_text_description, cached_confidence, steps_since_cache)`
- Output: P(valid | cached_coords_still_correct)
- Architecture: 2-layer MLP on concatenated embeddings
- Training signal: When VLM grounding is actually called, compare returned coords with any cached coords for the same element description. If distance < threshold (e.g., 5px), label as valid; otherwise invalid.

**Key advantage over heuristic cache**: The GVP can predict validity even when the UI has partially changed (e.g., dialog moved but button still in same relative position) — something SSIM cannot do.

### 2.3 Adaptive Tool Routing

When STP predicts a step type, the Worker dynamically filters its tool list:
- `gui_fresh` mode: only `click, click_image, type, hotkey, scroll, drag_and_drop, done, fail`
- `code_task` mode: only `call_code_agent, bash, done, fail`
- `gui_cached` mode: UI actions + `read_scratchpad` (to recall cached element descriptions)

This reduces token overhead by ~40-60% per step (from ~15 tool schemas to ~5-8).

### 2.4 Uncertainty-Gated Caching

Only cache high-confidence grounding results (VLM confidence > 0.7). Low-confidence results are marked volatile and never cached. This addresses the reviewer's concern about systematic cache errors.

---

## 3. Implementation Plan

### Phase A: Empirical Characterization (1 week)
1. Run HIPPO Agent on 50 OSWorld tasks with logging
2. For each grounding call, log: `(screenshot_hash, text_desc, coords, confidence, step_idx, prev_action)`
3. Measure:
   - Grounding call redundancy rate (how often same visual state + similar description appears)
   - Per-step tool usage distribution (what fraction of steps use only GUI tools vs. code tools)
   - Token breakdown: how many tokens are tool descriptions vs. actual reasoning
4. Publish findings as Section 3 (Analysis) of the paper

### Phase B: STP Training (1 week)
1. Use Phase A logs as training data
2. Label each step with its optimal mode (post-hoc: what tools were actually used?)
3. Train BERT-tiny classifier: input = trajectory signals, output = step mode
4. Evaluate prediction accuracy on held-out trajectories

### Phase C: GVP Training (1 week)
1. From Phase A logs, create paired samples:
   - Positive: (screenshot_t, description_d, coords_t) and (screenshot_t+k, description_d, coords_t+k) where distance < 5px
   - Negative: same but distance > 10px
2. Train 2-layer MLP on concatenated CLIP embeddings + metadata
3. Evaluate: precision/recall of validity prediction

### Phase D: Integration (1 week)
1. Modify `OSWorldACI.generate_coords()` to check STP mode and GVP before calling VLM
2. Modify `Worker.generate_next_action()` to use STP-predicted tool subset
3. Add cache layer with GVP-gated storage

### Phase E: Evaluation (2 weeks)
1. Full OSWorld benchmark (369 tasks, 3 runs each)
2. Ablations: STP only, GVP only, STP+GVP, vs. baseline
3. Analysis: per-category breakdown, failure cases, efficiency metrics

---

## 4. Dominant Contribution

**What's new**: A unified learned framework (STP + GVP) that simultaneously addresses grounding redundancy and tool overhead in GUI agents, grounded in an empirical characterization study that itself constitutes a finding.

This is NOT "just a cache" — it's:
1. An empirical discovery paper (grounding redundancy characterization)
2. A learned adaptive execution policy (STP)
3. A learned validity predictor for cached grounding (GVP)
4. An integrated system that combines all three

---

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Redundancy rate too low (< 10%) | Pivot to tool routing only (STP is still valuable); reframe paper as adaptive tool selection |
| GVP accuracy insufficient | Fallback to fresh VLM call when GVP confidence < threshold; cache never hurts accuracy |
| STP prediction errors cause wrong tool subset | Always include `done` and `fail` in all modes; allow VLM to request mode switch |
| OSWorld task completion rate drops | Conservative caching (only cache confidence > 0.8); abort cache if error detected |
| Reviewer still says "engineering not ML" | Lead with empirical findings (Section 3) and learned policy (STP + GVP), not the cache itself |

---

## 6. Target Venue

- **Primary**: ICLR 2027 (deadline ~Sep 2026)
- **Backup**: NeurIPS 2026 (deadline ~May 2026), ACL 2026
- **Positioning**: "Empirical + Systems" paper, not pure ML theory

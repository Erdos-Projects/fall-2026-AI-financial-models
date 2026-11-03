# Findings: LLM-guided improvement search, pilot (one search per method)

Written November 2, 2026, after every search and the confirmation stage had finished. The data are in the [report](report.md) and the [protocol](protocol.json), which was written before any search training. The reserved June–December 2020 test dates have **not** been evaluated.

## Short answer

In this pilot, both LLM searches reached strong configurations within **two proposals**; Bayesian optimization needed all twelve to get there. The search that received diagnostic feedback finished best, and the ranking held on the 150 later validation dates. However:

- Each method ran **one search**, so this cannot tell a better method apart from a lucky search.
- The gaps between the top three methods are small: 0.06–0.11% in paired comparisons.
- The main improvement (switching to L1 loss) was also found by Bayesian optimization, which had no diagnostics.

So the LLM's demonstrated value here is **speed of finding the improvement**, not discovering something conventional tuning missed.

## What was compared

Five searches tuned the unchanged `15→32→16→1` MLP over the same eight training and calibration settings. Each had a budget of 12 proposals; GPT's arm had 3. Each proposal was trained on three forward folds inside the training dates and scored by mean equal-date MAE on the folds' later dates.

| Search | How proposals were made |
|---|---|
| Random | Uniform sampling of the space |
| Bayesian | Optuna Gaussian-process optimizer (3 random starts, then model-guided) |
| LLM, no feedback | Claude Opus 5.5 read a diagnostic brief once and proposed all 12 configurations |
| LLM + feedback | Claude Opus 5.5 proposed one configuration at a time and saw every earlier measured result |
| GPT's 3 hypotheses | The three first experiments from the project's GPT planning notes |

Each search's best configuration was then refit with seeds 17, 42 and 2026 and scored on the 150 original validation dates, which no search had seen. Every LLM prompt was answered by a fresh Claude subagent that saw only its prompt file; see [how the prompts were answered](llm_relay.md).

## Results

| Search | Best search change | Mean best-so-far | First proposal at ≤ −0.45% | Validation vs baseline config [paired 95%] | Validation vs persistence | Validation dates better than persistence |
|---|---|---|---|---|---|---|
| LLM + feedback | −0.603% | **−0.541%** | **2** | **−0.836%** [−1.192, −0.533] | **−0.589%** | 111/150 |
| LLM, no feedback | −0.512% | −0.469% | **2** | −0.787% [−1.118, −0.517] | −0.539% | 106/150 |
| Bayesian | −0.514% | −0.246% | 12 | −0.728% [−1.064, −0.460] | −0.480% | 101/150 |
| Random | −0.184% | −0.095% | never | −0.302% [−0.435, −0.189] | −0.053% | 69/150 |
| GPT's 3 hypotheses | −0.136% | −0.054% | never | −0.264% [−0.447, −0.050] | −0.015% | 19/150 |

How to read the table:

- **Search-stage columns** compare with the baseline configuration on the search folds, where it scored 0.0144484 and persistence 0.0144451.
- **Mean best-so-far** averages the best result found after each of the 12 proposals, so it rewards finding improvements early.
- **The baseline configuration and persistence on validation:** the baseline configuration scored 0.0149357 there, 0.249% worse than persistence (0.0148986). This is consistent with the frozen checkpoint's earlier +0.206%.

Three conclusions:

- **Every search beat the baseline configuration on new dates**, and all five intervals exclude zero.
- **Only the three L1-based selections clearly beat persistence**, on two-thirds or more of validation dates. This is the first time a model in this project has beaten persistence on equal-date MAE.
- **Price errors change much less than return errors.** The feedback selection is the only one that also lowers price MAE (−0.15% vs persistence; price RMSE +0.01%). The other L1 selections keep price MAE 0.31–0.60% worse than persistence.

**Exploratory paired comparison with Bayesian optimization.** This was not a predeclared endpoint. It uses seed-averaged daily errors and the same date-block bootstrap:

| Comparison | Mean difference | 95% interval | Dates where the first is better |
|---|---|---|---|
| LLM + feedback minus Bayesian | −0.110% | [−0.211, −0.024] | 52/150 |
| LLM, no feedback minus Bayesian | −0.060% | [−0.120, −0.012] | 104/150 |
| Random minus Bayesian | +0.429% | [+0.224, +0.669] | 27/150 |

The feedback selection's average advantage comes from fewer, larger daily wins. Bayesian's selection is slightly better on most dates.

These intervals describe variation across dates for **fixed** selected configurations. They say nothing about how much a repeated search would vary. Search-to-search variation is the uncertainty that matters for "which method is better", and one search per method cannot measure it.

## What the searches found

**The shared recipe.** The three strong selections share **L1 loss, date-balanced sample weighting and no dropout**.

- **Why L1 helps.** Under the baseline's Smooth L1 with beta 1, about 98% of training rows fall in the quadratic region. The network therefore learned something close to a conditional mean, whereas MAE is minimized by the conditional median. This explains the earlier pattern: better RMSE, worse MAE.
- **Learning rate.** The feedback search then tuned it: 0.001 → 0.0005 (worse) → 0.002 → 0.004 (better) → 0.008 (much worse) → 0.003 (best).
- **Calibration** changed the search objective by at most about 0.05 percentage points once the loss was L1. Fitted multipliers stayed between 0.85 and 1.0 in every confirmation run.
- **Weight decay's best value depended on the learning rate:** 0.1 helped slightly at a learning rate of 0.001, but hurt at 0.004.

**The no-feedback LLM's opening proposals.** Its first, most conservative proposal was exactly GPT's hypothesis 3, per-price-group calibration; that arm reused the cached result. Its second proposal (L1, date weighting, no dropout, per-group calibration) already reached −0.495%.

**The feedback LLM's procedure.** It changed one or two settings at a time from the best configuration so far. When a change hurt, it reverted it. Its hypotheses cited specific diagnostics, for example learning curves still improving in the last epoch, or gains concentrated in options priced at or below one.

**Quality of the LLM's proposals.**

- **Validity:** all 24 proposals were valid, none needed a repair round, and neither LLM arm repeated a configuration within its own search.
- **Predictions:** the predicted changes ranked the measured changes moderately well (Spearman +0.61 without feedback, +0.41 with). At step 10 it was overconfident: it predicted −0.58% for doubling the learning rate to 0.008, and the measured change was −0.26%.

**GPT's hypotheses.**

- **Date weighting alone:** no effect.
- **One global α:** unstable across folds and ended at persistence's level.
- **Per-price-group α:** helped on the search folds (−0.136%). On validation it was essentially persistence (−0.015%, better on 19/150 dates). Its fitted multiplier was 0 for options priced above one in every seed, and 0 for both groups in two of the three seeds. Shrinking toward persistence was not where the gain lay; changing the training loss was.

## Cost

**Training compute** was similar for the three strong searches: 332–334 epochs for the LLM searches against 299 for Bayesian, because L1 configurations trained longer before early stopping. Wall-clock training times are not comparable: the LLM searches ran alongside each other and alongside confirmation, while random and Bayesian search ran one after the other.

**LLM cost** was not small. The feedback search used 12 subagent runs totalling about **1.26 million tokens and 111 minutes**; the no-feedback search used one run, 0.11 million tokens and 12 minutes. Bayesian optimization's proposal overhead was seconds. Prompts grew with history, from 8.8k to 57k characters, so a longer feedback search would cost more per step.

## Limitations

- **One search per method.** The pilot cannot separate method quality from search luck. The no-feedback/feedback/Bayesian ordering may change with repeats.
- **Single seed during search.** Seed standard deviation of validation equal-date MAE was about 0.03–0.07% here, the same order as the gaps between the top three methods.
- **Validation dates were already used.** They guided the earlier architecture choice, and the LLM brief summarized earlier findings from them. Confirmation there is still a development estimate; only the reserved test dates give an untouched estimate.
- **The LLM was Claude Opus 5.5 running as a Claude Code subagent**, not a chat model with tools disabled. Compliance with "read only this prompt" is self-reported. At step 12 the subagent read its prompt file twice because the first read was truncated. A transcript check confirmed that every subagent saw its complete prompt.
- **Results are specific to this eight-parameter space.** Because L1 loss was one of three loss choices, it was easy for any method to reach.
- **These are forecast errors, not trading results.** The data have no executable quotes, so trading profitability was not evaluated.

## Recommended next steps

1. **Repeat each search** two more times: new random/GP seeds and fresh LLM conversations, same protocol. This would show whether the LLM's early-proposal advantage is consistent.
2. **Score-only ablation.** Run an LLM search whose feedback contains only the objective, no diagnostics, to test whether the diagnostics themselves add value.
3. **Final test.** Once selections are final, evaluate them once on the reserved test dates (`confirm_nifty_llm_search.py --final-test`). This requires the team's approval and should not be repeated.
4. **Feature engineering.** Run it as a separate experiment, now that the loss and weighting choices are settled.

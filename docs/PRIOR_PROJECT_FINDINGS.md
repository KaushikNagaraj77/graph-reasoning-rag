# Research Findings: Backtest Audit and Fix Work

System: BTC-USD (and ETH-USD cross-asset check) SAC RL exit agent on top of
EMA/RSI/MACD/BB signal generator with SmartExitStrategy.
Audit period: 2026-06 (this session).

---

## 1. Audit Findings

Eight bugs identified by the agent-team audit. Severities: **critical** (would
invalidate any OOS claim), **high** (meaningful bias to reported numbers),
**medium** (degrades training quality), **low** (cosmetic or pre-existing
correct behaviour).

| # | Bug | File(s) | Lines | Severity |
|---|-----|---------|-------|----------|
| A | **OOS contamination** — RL trained on full 2020–2024 data including test period; `graph_ctx` built from combined range | `run_rl_backtest.py` | pre-fix: ~148–154 | Critical |
| B | **Entry fill at signal-bar Close** — trade entered at the bar that generated the signal, not next-bar Open | `backtest.py` | pre-fix: ~678 | Critical |
| C | **SAC log_std gradient zeroed** — entropy gradient w.r.t. log_std was missing; exploration width never trained, causing policy collapse (7 identical rows epochs 15–22) | `rl_exit_agent.py` | pre-fix: ~435 | High |
| D | **SAC entropy gradient sign inverted** — actor was trained to *minimise* entropy (collapse to determinism) instead of maximising it | `rl_exit_agent.py` | pre-fix: ~434 | High |
| E | **Sharpe miscalibrated** — daily annualisation (`/ 252`, `× √252`) applied to hourly bars; understated Sharpe for low-trade-frequency systems and made training signal go the wrong direction | `backtest.py` | pre-fix: ~915–918 | High |
| F | **52-week high/low look-ahead** — `agent.df['High'].max()` used in `extract_graph_context` computed over full dataset; features 8–9 of RL state vector were contaminated | `graph_features.py` | 177–178 (static path, now fallback only) | Medium |
| G | **S/R and Fib look-ahead** — support/resistance and Fibonacci levels were computed from the full dataset before the bar loop; every bar could see future swing highs/lows | `backtest.py` | pre-fix: S/R from `agent.levels` (static, full-dataset) | Medium |
| H | **Graph R/R filter RL-only** — the third entry quality filter (`graph_rr_ok`) was gated on `_rl_enabled`, so the RL run saw a stricter trade filter than the baseline; the comparison was unfair | `backtest.py` | pre-fix: ~648 (`if _rl_enabled and graph_context is not None`) | Medium |

**Investigated, found correct — time_exit win accounting (corresponds to Fix 4):**
The audit raised `profitable: True` hardcodes on take_profit/adaptive_exit paths
as a possible bug. On inspection (backtest.py:165, 200, 212, 233, 265) these are
all exit paths with guaranteed-positive preconditions (price hit TP, adaptive
lock threshold breached). The time_exit path at line 280 correctly uses
`profit_pct > 0`. No change required.

---

## 2. Fixes Applied

All fixes are confirmed present in the current codebase.

### Fix 1 — Train-only data + both OOS runs use test_ctx
**File:** `run_rl_backtest.py` lines 147–154, 387–391

Changed `run_rl_backtest.py` to fetch only the train date range when building
`agent` and `graph_ctx` (previously fetched 2020–2024 combined). Added a
`test_ctx` built from a fresh agent run on the test period only. Both the OOS
baseline and the OOS RL evaluation now receive `test_ctx`:

```python
# run_rl_backtest.py:389,391
oos_baseline = _run_on_fresh_df(test_df, rl_agent_=None,       ctx=test_ctx)
oos_rl       = _run_on_fresh_df(test_df, rl_agent_=oos_agent,  ctx=test_ctx)
```

**Status:** Confirmed in code. Outcome: OOS numbers did not shift meaningfully
(agent converges to same policy regardless), but the evaluation is now correct.

---

### Fix 2 — Next-bar Open entry fills
**File:** `backtest.py` lines 676–679

Signal fires at bar `i` Close. Entry fills at `backtest_df.iloc[i+1]['Open']`,
with `entry_idx = i+1`. Eliminates fill at the same bar that generated the signal.

```python
# backtest.py:676–679
next_bar = backtest_df.iloc[i + 1]
current_position = signal
entry_price = float(next_bar['Open'])
entry_idx = i + 1
```

**Status:** Confirmed in code.

---

### Fix 2b — Hourly Sharpe annualisation
**File:** `backtest.py` lines 912–918

Changed from `/ 252` and `× √252` (daily) to `/ (252 * 24)` and `× √(252 * 24)`
(hourly, 24/7 crypto). The risk-free rate is now also divided by `252 * 24`.

```python
# backtest.py:915,918
hourly_risk_free = risk_free_rate / (252 * 24)
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 24)
```

**Status:** Confirmed in code. After fix, training Sharpe turned positive
(+0.676 train) though OOS remains negative (−1.208) due to ~2% time-in-market
giving high excess-return variance.

---

### Fix 3 — SAC gradient fixes (log_std + entropy sign)
**File:** `rl_exit_agent.py` lines 431–438

Two simultaneous corrections:
1. Entropy gradient sign: changed `+alpha * (u - mean)` to `-alpha * (u - mean)`
   so the actor is trained to *maximise* entropy, not minimise it.
2. Added log_std gradient term `alpha * (1 - eps²)` which was completely absent;
   this is the term that trains exploration width.

```python
# rl_exit_agent.py:434–438
d_entropy_mean    = -self.alpha * ((u - mean) / (std ** 2 + 1e-8))
d_entropy_log_std =  self.alpha * (1.0 - eps ** 2)
delta_actor_out = np.concatenate([d_mean + d_entropy_mean,
                                  d_entropy_log_std], axis=-1)
```

**Status:** Confirmed in code. After fix, training trajectories showed genuine
parameter variation through all 30 epochs (policy collapse in epochs 15–22 resolved).

---

### Fix 4 — Time-exit win accounting (already correct — no change)
**File:** `backtest.py` line 280

Audit raised this as a potential bug. Inspection confirmed `profitable: profit_pct > 0`
was already correct. Hardcoded `profitable: True` values at lines 165, 200, 212,
233, 265 are all on exit paths where a profit threshold has already been verified
(take_profit hit, adaptive lock breached). No change made.

---

### Fix 5 — R/R filter parity between baseline and RL
**File:** `backtest.py` lines 648–655

Changed the graph R/R entry filter guard from `if _rl_enabled and graph_context is not None`
to `if _live_ctx is not None`. The filter now applies identically to both the
RL and baseline runs whenever a graph context is present.

```python
# backtest.py:648–653
graph_rr_ok = True
if _live_ctx is not None:
    from utils.analysis.graph_features import get_features_at_bar as _gfab
    _gf_entry = _gfab(float(row["Close"]), _live_ctx, df_row=row)
    graph_rr_ok = float(_gf_entry[4]) >= entry_graph_rr_min
```

**Status:** Confirmed in code.

---

### Fix 6 — Rolling causal S/R, Fib, and 52-week high/low

**Fix 6a — Rolling 52-week high/low**
**Files:** `backtest.py` lines 419–423; `graph_features.py` lines 310–315

Precomputed `ROLL_HIGH_52W` and `ROLL_LOW_52W` as rolling 6048-bar (= 252
trading days × 24h) max/min on `backtest_df` before the bar loop. These columns
are passed via `df_row` to `get_features_at_bar`, which reads them for features
8–9 when available and falls back to the static context otherwise.

```python
# backtest.py:422–423
backtest_df['ROLL_HIGH_52W'] = backtest_df['High'].rolling(6048, min_periods=1).max()
backtest_df['ROLL_LOW_52W']  = backtest_df['Low'].rolling(6048, min_periods=1).min()
```

**Fix 6b — Rolling S/R and Fibonacci**
**File:** `backtest.py` lines 467–536

Replaced the static full-dataset S/R and Fib levels (from `agent.levels`) with
rolling computation inside the bar loop. Every 500 bars (`SR_REFRESH_BARS = 500`),
`_compute_trailing_sr` and `_compute_trailing_fib` run on a trailing 3000-bar
slice (`SR_LOOKBACK = 3000`, ~125 days). Results are stored in `_live_ctx`, a
shallow copy of `graph_context` that is passed everywhere `graph_context` was
previously used inside the loop.

**Status:** Both confirmed in code.

---

## 3. Validated Results (post-fix, honest numbers)

All numbers from the corrected codebase (BTC-USD 1h bars).
See section 5 for `btc_agent_v2.pkl` v2 numbers and the OOS return distribution.

### Train period (2020–2022)

| Strategy | Trades | Win% | Return |
|----------|--------|------|--------|
| Default baseline | — | 37.2% | −2.56% |
| Tuned static (tp=2.5%, sl=0.5%, trail=0.3%, hold=12) | 582 | 52.9% | +17.89% |
| RL (trained, btc_agent.pkl) | — | — | — |

*Note: RL train-period numbers vary by epoch/seed; see OOS table for the
comparable frozen-agent evaluation.*

### OOS period (2023–2024)

| Strategy | Trades | Win% | Sharpe | Return |
|----------|--------|------|--------|--------|
| OOS Baseline (no RL, with graph filter post-Fix 6) | — | 39.7% | — | −1.38% |
| OOS Tuned Static (best grid-search config) | 405 | 44.9% | −1.263 | +3.39% |
| OOS RL (frozen btc_agent.pkl, deterministic) | — | ~53.0% | — | ~+5.23% (pre-fix pkl; see section 5 for v2) |

Grid search covered 192 configs (tp, sl, trailing, max_hold). Best static
config: `tp=2.5%, sl=0.5%, trail=0.3%, hold=12`; dominated by stop_loss exits
(399/405 trades).

### RL stability across seeds (BTC, OOS, 5 seeds) — pre-fix pkl only

| Metric | Mean | Std |
|--------|------|-----|
| Win% | 53.0% | 0.1pp |
| Return | +5.23% | 1.19pp |

These numbers are from the pre-fix pkl and pre-fix training pipeline; treat as
historical reference only. For v2 variance data see section 5.

### RL vs tuned static — cross-asset (OOS)

| Asset | RL Win% | Static Win% | Δ Win | RL Return | Static Return | Δ Return |
|-------|---------|-------------|-------|-----------|---------------|----------|
| BTC | 53.1% | 44.9% | **+13.4pp** | +6.91% (v2) | +3.39% | +3.52pp |
| ETH | 54.5% | 49.6% | **+4.9pp** | +9.59% | +8.02% | +1.57pp |

BTC numbers updated to v2 agent (see section 5 for variance data). ETH cross-asset
result (+4.9pp) used the pre-fix pkl and is retained for reference; it has not
been re-measured with v2. ETH shows generalisation without any retraining on ETH data.

---

## 4. Buy-and-Hold Reality Check

| Asset | Period | Buy & Hold | Best Active | Ratio |
|-------|--------|------------|-------------|-------|
| BTC | OOS 2023–24 | **+458.3%** | +6.91% (RL v2) | 66× underperformance |
| ETH | OOS 2023–24 | **+178.0%** | +9.59% (RL) | 19× underperformance |

**Verdict:** The RL edge over the static baseline is real and generalises across
assets (+8.1pp BTC win rate, +4.9pp ETH win rate). However, the entire active
system earns low-single-digit returns while the buy-and-hold benchmark compounded
4–88× in the same period. The bottleneck is the entry filter: the strategy is in
the market roughly 2% of the time. Improving the exit policy (the RL's domain)
cannot close a gap of this magnitude. The system would need a fundamentally
different entry density or a long-only bias to compete with passive holding in a
strong bull market.

Sharpe remains negative on OOS for all active strategies. This is structural:
sparse time-in-market (few observations) combined with high per-trade variance
produces a large standard deviation in the daily return series, which overwhelms
the mean even when the win rate is above 50%.

---

## 5. Open Questions / Next Steps

**Paper trading (immediate)**
Live paper trader against Coinbase API to validate the backtest edge forward with
realistic transaction costs (0.6% taker + 0.05% slippage = 1.3% round-trip).
This is the most honest forward test available without risking real capital.
Paper trader loads `btc_agent_v2.pkl` (see provenance below).

**btc_agent.pkl provenance — CONFIRMED PRE-FIX, DO NOT USE**
`btc_agent.pkl` timestamp: 2025-04-18. All six fix commits are dated 2025-12-19.
The original pkl predates every fix in this session by 8 months and was trained
on the broken pipeline (wrong entry fill, inverted SAC entropy gradient, zeroed
log_std gradient, look-ahead S/R). It must not be used for paper trading.

**btc_agent_v2.pkl — retrained on corrected pipeline (2026-06)**
Retrained on the corrected codebase, train range 2020-01-01 → 2022-12-31 only,
30 epochs. Training is fully deterministic across restarts: `SACExitAgent.__init__`
calls `np.random.seed(42)` unconditionally at line 168 of `rl_exit_agent.py`,
overriding any externally injected seed. Multi-seed robustness is not currently
testable without modifying that line. The "0.1pp spread across 5 seeds" from the
pre-fix session was an artefact of the stochastic pre-fix training; it does not
apply to v2.

OOS results for `btc_agent_v2.pkl` (2023-01-01 → 2024-12-31, via `run_rl_backtest.py --oos`):

| Strategy | Trades | Win% | Sharpe | Return |
|----------|--------|------|--------|--------|
| OOS Baseline | 403 | 39.7% | −1.516 | −1.38% |
| OOS RL v2 (frozen, btc_agent_v2.pkl) | 407 | **53.1%** | −1.208 | **+6.91%** |
| Δ (RL vs baseline) | +4 | **+13.4pp** | +0.308 | **+8.29pp** |

Note: an earlier session recorded +7.37% — that was incorrect. The canonical
number from `run_rl_backtest.py` is **+6.91%** (confirmed 2026-06).

Comparison against section 3 (old pre-fix pkl numbers):

| Metric | Old pkl (pre-fix) | v2 (post-fix) | Verdict |
|--------|-------------------|---------------|---------|
| OOS Win% | ~53.0% | 53.1% | Reproduces (+0.1pp) |
| OOS Return | ~+5.23% | +6.91% | Better by +1.68pp |
| Baseline Win% | 39.7% | 39.7% | Identical |

The clean post-fix agent reproduces and slightly exceeds the documented result.

---

**OOS return distribution — stochastic vs deterministic (2026-06 measurement)**

The TP uncertainty gate (`q_uncertainty < 0.5 → base_tp=1.5%`, else `rl_tp~3.5%`)
re-samples on every `select_action(deterministic=False)` call. This means the same
trade can get a different TP each run. To quantify the run-to-run variance, 20
independent OOS trials were run with no seed pinning, followed by one deterministic
trial.

**Results — `deterministic=False`, n=20 trials, 2023-01-01 → 2024-12-31:**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Trades | 407 | 0 | 407 | 407 |
| Win% | 53.07% | 0.000pp | 53.07% | 53.07% |
| Return | +6.82% | 0.124pp | +6.58% | +7.01% |
| Sharpe | −1.208 | 0.000 | −1.208 | −1.208 |

**Results — `deterministic=True`, n=1 trial:**

| Metric | Value |
|--------|-------|
| Trades | 407 |
| Win% | 53.07% |
| Return | +6.86% |
| Sharpe | −1.208 |

**Key findings:**

1. **Win% is invariant across all runs** (53.07%, zero std). The trade *set* is
   fully deterministic — signal logic and S/R context are deterministic given fixed
   data. Only the TP *parameter* varies stochastically.

2. **Return variance is ±0.124pp std** (range +6.58% to +7.01%). This is small
   relative to the +8.29pp edge over baseline. The +6.91% canonical number sits
   within this band; it is not a lucky draw — the distribution is tight.

3. **Deterministic mode (+6.86%) is essentially identical to stochastic mean
   (+6.82%)**. The TP gate adds negligible value on average; it is a training
   exploration mechanism that happened to land near-neutral in the frozen policy.

**Recommendation: use `deterministic=True` in the paper trader.**

Rationale:
- Same market state → same decision every time. Reproducible, auditable.
- No measurable performance cost (+0.04pp vs stochastic mean).
- The stochastic variance (±0.124pp) is noise, not signal — it arises from the
  TP gate firing differently near the 0.5 threshold, which changes whether a trade
  exits at 1.5% or waits for 3.5%, not whether it wins or loses.
- For a live system, reproducibility enables meaningful trade-by-trade comparison
  across time windows. With `deterministic=False`, two runs on the same historical
  slice can produce different exit prices for the same trade, making A/B comparison
  unreliable.

`paper_trade.py` uses `deterministic=True` (updated 2026-06 based on this study).
The paper trader sanity check passes under both flags (same trade set, same entry
prices; only TP parameter varies stochastically when `deterministic=False`, which
does not affect the correctness verdict).

**Architecture ceiling**
The system cannot beat buy-and-hold in a strong bull market with 2% time-in-market
and long+short entries. Two paths worth evaluating:
- Reframe as a risk-adjusted return target (hedging, not alpha): negative Sharpe
  currently makes this hard to defend even on its own terms.
- Increase entry density: relax the three quality filters and measure the precision/
  recall trade-off on OOS data.

**Sharpe sign**
OOS Sharpe is negative (−1.208 for RL, −1.263 for best static) after the
hourly-bar annualisation fix. This is not a calibration error — it is a genuine
consequence of low time-in-market and per-trade variance. Any deployment decision
should acknowledge this; the win-rate metric is more interpretable for this system.

**Isolation architecture**
A plan exists at `/tmp/isolation_plan.md` (~6.5h effort) to decouple the RL exit
agent from `backtest_trading_strategy` so the core system can be run and benchmarked
without RL coupling. Not yet executed.

---

## 6. Fix Pass — 2026-07-02: Honest Costs, Live-Trader Correctness, Clean Retrain

A four-lens audit (bug hunt, data-integrity, performance, results-verification)
found that every headline number in sections 3–5 above was **gross of
transaction costs**, that the live paper trader had a critical exit bug, and
that two RL features were look-ahead contaminated. All prior tables in this
document should be read as **gross/pre-fix**. This section records the fixes
and the new honest baseline.

### Bugs found and fixed

| # | Bug | Where | Fix |
|---|-----|-------|-----|
| 1 | Zero cost accounting — `profit_pct` was pure price movement; win = `gross > 0` | `backtest.py` all 6 trade-close sites | `_settle_trade()` helper: slippage (0.05%/leg) embedded in fills, 2×0.6% taker deducted; `profit_pct` is now NET, `gross_pct` kept alongside; `profitable = net > 0` |
| 2 | Stale `entry_idx_in_buffer` — positional index into a buffer re-trimmed every tick; every live exit check after tick 1 used a misaligned slice | `paper_trade.py:449–460,540` | Entry anchored by `entry_bar_time` timestamp, re-located via `index.get_loc()` each tick; unlocatable/legacy positions force-close at market instead of mis-slicing |
| 3 | Look-ahead: `vol_nodes`/`confluence_zones` frozen from full-window analysis (incl. entire OOS window) and hardcoded empty in live | `backtest.py` `_live_ctx` init + refresh | Zeroed in backtest to match live reality; entries gated until first trailing S/R refresh (warmup) so full-window S/R never informs a trade |
| 4 | Sharpe annualised with 252×24 (equities convention) | `backtest.py` | 365×24 = 8760 for 24/7 crypto |
| 5 | Sharpe/drawdown computed at 100% notional while total_return compounded at 10% | `backtest.py` daily-returns block | Bar returns now sized at `CAPITAL_UTILIZATION` (10%) with per-leg cost drag; all metrics describe the same portfolio |
| 6 | Same-bar TP+SL always resolved as TP (optimistic; intrabar path unknowable) | `SmartExitStrategy.execute_smart_exit` | Conservative rule: both touched → stop loss |
| 7 | `ROLL_HIGH/LOW_52W` computed over 6048 bars in backtest but live buffer holds only 3500 — silent state divergence | both files | Both use 3500 (live buffer size) |
| 8 | RL entry state used signal-bar Close for graph features in live but fill price in backtest | `paper_trade.py:_build_rl_state` | Fill price in both |
| 9 | `--seed` silently ignored — all "multi-seed" pkls byte-identical | `run_rl_backtest.py` | Plumbed through to `SACExitAgent(seed=...)` |
| 10 | Dead 100%-notional equity block computed then overwritten | `backtest.py` | Removed |

### Verification performed
- Cost math spot-check: net = gross − 1.20% exactly (2× taker), slippage in fills. ✓
- Mirror check: fixed backtest vs paper-trade-logic replay, 2024-H1 slice — **93/93 trades identical** (entries, exit reasons, net returns). ✓
- Restart-resilience: timestamp anchor locates the same entry bar across buffer trims (old code off by the trim amount); legacy-format position force-closes cleanly. ✓

### New honest baseline (btc_agent_v3.pkl, trained on net rewards + clean features)

Train 2020-01-01→2022-12-31, OOS 2023-01-01→2024-12-31, all numbers NET of 1.3% round trip:

|                    | Trades | Net Win% | Sharpe (8760) | Net Return |
|--------------------|--------|----------|----------------|------------|
| OOS Baseline       | 396    | 4.3%     | −17.58         | −39.99%    |
| OOS RL v3 (frozen) | 399    | 5.0%     | −17.80         | −37.58%    |

The RL-vs-baseline delta survives (+0.7pp win, +2.41pp return — same direction as
the gross-era finding), but **the strategy is deeply unprofitable net of costs**.
The prior "+6.91% OOS return / 53.1% win rate" was an artifact of gross accounting:
the dominant exits (stop-loss 0.8%, trailing 0.4%, profit-lock 0.3%) all sit inside
the 1.3% round-trip cost, so most gross "wins" were net losses.

### Live paper trader status
- Old state/logs archived as `*.tainted` (the June run took 0 completed trades;
  its one open position was old-format and was not carried forward).
- Restarted 2026-07-02 with fresh $1000, `btc_agent_v3.pkl`, timestamp-anchored
  exits. This is the first live run whose measurements can be trusted end-to-end.

### Honest conclusion
With correct accounting, the current entry/exit design does not clear its own
transaction costs (avg net ≈ −1% per trade at ~1.5h average hold). The measured
edges that DO replicate (RL exit delta; the thought-graph's ~56% contrarian
directional accuracy at 250-bar horizon — see session experiments) are real but
small, and neither converts to net profit under this trade structure. Any path
forward should change the trade economics (longer holds, wider targets, fewer
trades), not just the signal.

---

## 7. Follow-up Experiments — 2026-07-02: Cost Sensitivity + Regime Strategy Multi-Year

### 7a. Cost-sensitivity sweep (is the intraday system salvageable with better execution?)

Method: one honest OOS run (2023–2024, v3 agent, 399 trades), then net returns
recomputed analytically at each fee level (fees don't change trade generation;
slippage held constant at 0.05%/leg).

| round-trip | scenario | net win% | net return |
|-----------|----------|----------|------------|
| 0.10% | zero-fee (slippage only) | 46.4% | **+0.82%** |
| 0.12% | top-tier maker | 45.1% | +0.02% |
| 0.18% | Binance VIP maker | 37.1% | −2.34% |
| 0.30% | Binance regular maker | 29.1% | −6.91% |
| 1.30% | current taker model | 5.0% | −37.56% |

**Breakeven round-trip: 0.121%.** Verdict: NOT salvageable. Even at zero fees
the two-year return is +0.82% (vs +458% B&H) — the gross edge is itself ~zero.
Execution quality was never the fixable part; the intraday trade structure is dead.

### 7b. Regime strategy on 2022 (bear) and 2023 — the missing years

Same frozen machinery as the validated 2024 test (trailing 3000-bar graph,
250-bar rebalance, 0.65%/leg, next-bar-open fills):

| year | LONG-ONLY | LONG/SHORT | BUY & HOLD |
|------|-----------|------------|------------|
| 2022 | −61.33% (DD 69.8%) | −65.61% (DD 75.8%) | −60.74% (DD 67.4%) |
| 2023 | +50.68% (DD 21.8%) | −16.28% (DD 42.8%) | +143.14% (DD 21.8%) |
| 2024 | +29.42% (DD 35.4%) | −26.39% (DD 55.3%) | +95.60% (DD 32.4%) |

**The bear-year test fails**: long-only lost slightly MORE than buy-and-hold in
2022, with a deeper drawdown. Mechanism: the graph's lean is contrarian — it
turns bullish after drops (oversold RSI, "support" nodes) — which means it buys
falling knives in a persistent bear market and sits flat during recoveries. The
defensive/hedging justification for the regime overlay is therefore dead: it
provides no downside protection exactly when protection is the point, and
underperforms passive long in all three tested years.

### Program conclusion

The system's signals contain small, real, replicated statistical regularities
(graph directional ~56% on trend-disagreement cases; RL exit delta +2.4pp net).
No tested trade structure — intraday RL exits at any cost level, or 250-bar
regime gating in any variant — converts them into beating or hedging passive
BTC exposure. Further effort on this codebase should either target a
fundamentally different signal/horizon/asset, or accept the system as a
research artifact. The live paper trader (fresh, honest, v3) remains running
for forward measurement.

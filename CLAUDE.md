# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An interactive, browser-based visualization of policy alignment on a toy problem. The "language model" is a 1D Gaussian policy $\pi_\theta = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$, the reward $r(a)$ is a sum of Gaussian bumps, and a reference policy $\pi_{\mathrm{ref}}$ anchors the optimization via a KL penalty. The app contrasts six alignment algorithms on the same objective:

- **RLHF exact** — analytic $\nabla_\theta J$.
- **PG** — REINFORCE with an in-batch mean baseline.
- **PPO** — clipped surrogate with a learned scalar baseline $V$, K inner epochs per batch against a frozen $\pi_{\mathrm{old}}$.
- **GRPO** — clipped, group-normalized surrogate against a frozen $\pi_{\mathrm{old}}$, also with K inner epochs.
- **DPO** — pairwise preference loss with Bradley-Terry noise.
- **DRO** — Direct Reward Optimisation (Richemond et al., 2024) with a learned baseline $V$.

A **Compare** tab runs all six from the same starting $(\mu_\theta, \sigma_\theta)$, overlays their trajectories on the loss landscape, and plots $J(\theta_t)$ over steps (median + min–max band when "Runs per algorithm" > 1).

The original Mathematica reference (`mathematica_example.nb`) and brief (`prompt.md`) live in the sibling directory `/Users/landajuela/Work/GitHub/alignment/`, not in this repo. Pre-approved grep patterns for the notebook are in `.claude/settings.local.json`.

## How to run

There is no build step, no package manager, and no test suite beyond the in-page self-test. Everything is one static HTML file with CDN-loaded dependencies (Plotly, KaTeX, Google Fonts; the Plotly/KaTeX tags carry SRI hashes — update the `integrity` attribute if you bump a version).

- **Recommended**: open in VS Code and use Live Server (configured for port 5501 in `.vscode/settings.json`).
- **Alternative**: `python3 -m http.server 5501` then visit `http://localhost:5501/index.html`, or `open index.html` for a quick look (some browsers restrict CDN/font behavior under `file://`).
- **Self-test**: visit `index.html?selftest` — `runSelfTest()` finite-differences every analytic gradient across all reward presets, sanity-checks $\pi^\star$ (normalization; $J(\pi^\star)$ must dominate the Gaussian landscape grid), and checks each algorithm core against its own surrogate on a frozen, seeded batch (PPO/GRPO vs. the clipped surrogate $-\beta$KL, DPO vs. the logistic loss, DRO vs. the MSE in $(\theta, V)$, PG statistically vs. $\nabla J$ at $N = 60{,}000$). Run this after any change to the math. It can also be run headlessly: extract the inline `<script>`, stub the DOM (see the memory note on JXA), and call `runSelfTest()` via `osascript -l JavaScript`.

## Code architecture

Everything lives in `index.html` (~3160 lines): HTML structure, inline CSS in `:root`-themed custom properties, and a single inline `<script>` block starting at line ~1030. The script is structured as labeled sections (see the `/* === */` banners):

- **`C` (palette) and RNG (top of script)** — `C` holds the semantic colors read once from the CSS custom properties (`--pi`, `--star`, …); use `C.*` (and `hexToRgba`) in Plotly traces, never hard-coded hexes. `setSeed`/`rand`/`randn` form a seedable RNG stream (mulberry32); `S.seed > 0` makes runs reproducible and `compareRunAll` re-seeds at the start of each Run all.
- **`S` (global state, ~line 1076)** — single source of truth for the page. Holds reference/policy params (`muRef/sigRef`, `muTh/sigTh`, `beta`, `rewardPreset`, `seed`), per-algorithm hyperparameters and learning rates (`lrExact`; `NPG/lrPG`; `NPPO/epsPPO/lrPPO/lrPPOv/Vppo/Kppo`; `G/eps/lrG/Kgrpo`; `betaDPO/N/lrDPO/prefNoise`; `betaDRO/NDRO/lrDROth/lrDROv/Vdro`), per-algorithm sample snapshots (`pg`, `ppo`, `grpo`, `dpo`, `dro`), trajectory arrays (`trajExact/trajPG/trajPPO/trajGRPO/trajDPO/trajDRO`), Compare J-curves (`cmpJ`), play-loop handles, and the `ppoTick/grpoTick` counters that pace π_old refreshes during Play. Mutate `S` then call `renderAll()` (or `requestRender()` from high-frequency handlers); do not keep parallel state in the DOM.
- **Reward model** — `REWARD_PRESETS` defines named lists of bumps `{w, c, s}`; `reward(a)` evaluates the sum. Adding a preset means adding an entry here and an `<option>` in the `rewardPreset` `<select>`.
- **Analytic objective** — because both the policy and each reward bump are Gaussian, $\mathbb{E}_{\pi_\theta}[r]$, $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$, and their $\mu, \sigma$ gradients have closed forms. These live in `expectedReward`, `klGaussian`, `objective`, `gradExpected`, `gradKL`, `gradJ`. Edit these together if you change the model — they must stay consistent, and `?selftest` will catch drift.
- **`π*` (optimal policy)** — `computeStar()` evaluates $\pi^\star \propto \pi_{\mathrm{ref}} \exp(r/\beta)$ numerically on a fixed action grid (`A_GRID`, `A_MIN/A_MAX/A_N`). Both it and `computeLandscape()` are cached on the key `(rewardPreset, β, muRef, sigRef)` — keep the cache keys honest if you add a dependency.
- **Algorithm cores (~line 1266)** — the single home of each algorithm's math: `exactStepCore`, `pgStepCore`, `ppoStepCore`, `grpoStepCore`, `dpoStepCore`, `droStepCore`, plus shared helpers (`dlogPi`, `sampleBatch`, `groupAdvantages`, `clippedSurrogateGrad`, `dpoMarginAt`, `dpoMakePairs`, `droResidualAt`). Cores are pure in the policy: they take `(mu, sig, batch, params)` and return the updated values; they read only the shared environment (β, π_ref, reward preset) from `S` and never touch `S.muTh/S.sigTh` or the DOM. **Both the interactive tabs and the Compare runner call these cores — never re-implement an update rule in either place.**
- **Interactive tab wrappers** — each algorithm keeps the consistent triple around its core: a `*Resample()` that draws a fresh sample snapshot into `S.{pg,ppo,grpo,dpo,dro}`, a `recompute*Quantities()` that recomputes derived per-sample display quantities at the current $\theta$ (advantages, ratios, surrogate, DPO margins, DRO residuals), and a `*Step()` that calls the core and applies the result. PPO/GRPO snapshot $\pi_{\mathrm{old}}$ on each resample; their Play loops resample every `Kppo`/`Kgrpo` ticks so ratios drift from 1 and clipping engages between refreshes. Every step pushes onto its trajectory array so the landscape panel can draw the path.
- **Compare runner** — `compareRunAll()` (~line 1708) re-runs all six algorithms from the user's current $(\mu_\theta, \sigma_\theta)$ using `COMPARE_DEFAULTS` hyperparameters scaled by `cmpScale`, honoring the same K-epoch cadence, optionally repeating each stochastic algorithm `cmpRuns` times. It fills the six `traj*` arrays (first run) and `S.cmpJ` (J-curve per run), then restores the policy. Sample counts are clamped (`G≥2` for the $(G-1)$ std denominator, `N≥1` for DPO pairs, etc.). Keep `COMPARE_DEFAULTS` in sync with the slider defaults if you change them.
- **Rendering** — `renderAll()` reads `data-active-tab` off `<body>` and renders only the active tab's panels (rendering into a `display:none` container makes Plotly mis-size the SVG). Shared panels: `renderDensities` (policies + reward), `renderLandscape` (2D $\mu,\sigma$ heatmap with optional per-algorithm trajectory overlays; on the Compare tab the paths do *not* connect to the live dot), `renderMetrics`. Tab-specific panels: `renderPGPanels`, `renderPPOPanels`, `renderGRPOPanels`, `renderDPOPanels`, `renderDROPanels`, and `renderCompareJ` (~line 2713, the J-vs-step chart with median/min–max bands). The Compare tab uses `renderLandscape` with all six path flags and a `cornerLegend`.
- **Tabs & play loops** — `setTab(name)` toggles `data-active-tab` on `<body>`, maintains `aria-selected`/roving `tabIndex` (arrow keys work via a keydown handler), and triggers `renderAll()`. Each algorithm has its own `togglePlay*()` that drives the corresponding `*Step()` via `setInterval`.
- **URL hash state** — `writeHash`/`readHash` serialize every slider that differs from its HTML default, plus `rewardPreset`, `seed`, and the active tab, into `location.hash` (debounced via `scheduleHashWrite`). New sliders must be added to `HASH_SLIDER_IDS`; `readHash` restores them by dispatching synthetic `input` events, so the regular `bindSlider` handlers do the work.
- **UI binding** — `bindSlider(id, key, decimals, after)` near the bottom wires each `<input type="range">` to a key in `S`, schedules a hash write, and re-renders through `requestRender()` (rAF-coalesced). New controls follow the same pattern. `resetAll()` is the inverse — keep it in sync when you add a slider, play loop, or Compare-tab control (it also resets `compareVisible`, the Compare sliders, the seed, and clears the hash).

When changing the math, the invariant to preserve is: `objective`, the analytic gradients (`gradJ`), the PG/PPO/GRPO/DPO/DRO cores, the trajectory updates, and the optimum (`computeStar`) all refer to the same definition of $J(\theta) = \mathbb{E}_{\pi_\theta}[r] - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})$. The six `traj*` arrays must remain comparable on the same landscape. Verify with `?selftest`.

## Conventions

- Vanilla JS only — no bundler, no modules, no framework. Keep everything in `index.html` unless the user explicitly asks to split it.
- Color tokens (`--ref`, `--pi`, `--star`, `--reward`, `--accent`, plus per-algorithm accents) are the semantic palette; in JS use the `C` object (populated from those tokens at load) rather than hard-coding hex values.
- KaTeX renders math at page load via the auto-render `onload` hook; any math added to dynamically-rendered content must be re-rendered explicitly or kept in static HTML.

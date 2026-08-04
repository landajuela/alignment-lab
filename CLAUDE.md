# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An interactive, browser-based visualization of policy alignment on a toy problem. The "language model" is a 1D Gaussian policy $\pi_\theta = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$, the reward $r(a)$ is a sum of Gaussian bumps, and a reference policy $\pi_{\mathrm{ref}}$ anchors the optimization via a KL penalty. The app contrasts six alignment algorithms on the same objective:

- **RLHF exact** — analytic $\nabla_\theta J$.
- **PG** — REINFORCE with an in-batch mean baseline.
- **PPO** — clipped surrogate with a learned scalar baseline $V$, K inner epochs per batch against a frozen $\pi_{\mathrm{old}}$.
- **GRPO** — clipped, group-normalized surrogate against a frozen $\pi_{\mathrm{old}}$, also with K inner epochs.
- **DPO** — pairwise preference loss with Bradley-Terry noise.
- **DRO** — Direct Reward Optimisation (Richemond et al., 2024) with a learned value $V$, practical $1/\beta$ policy-gradient conditioning, and optional exact profiling of $V$.

A **Compare** tab runs all six from the same starting $(\mu_\theta, \sigma_\theta)$, overlays their trajectories on the loss landscape, and plots $J(\theta_t)$ over steps (median + min–max band when "Runs per algorithm" > 1).

The original Mathematica reference (`mathematica_example.nb`) and brief (`prompt.md`) live in the sibling directory `/Users/landajuela/Work/GitHub/alignment/`, not in this repo. Pre-approved grep patterns for the notebook are in `.claude/settings.local.json`.

## How to run

There is no build step, no package manager, and no test suite beyond the in-page self-test. Everything is one static HTML file with CDN-loaded dependencies (Plotly, KaTeX, Google Fonts; the Plotly/KaTeX tags carry SRI hashes — update the `integrity` attribute if you bump a version).

- **Recommended**: open in VS Code and use Live Server (configured for port 5501 in `.vscode/settings.json`).
- **Alternative**: `python3 -m http.server 5501` then visit `http://localhost:5501/index.html`, or `open index.html` for a quick look (some browsers restrict CDN/font behavior under `file://`).
- **Self-test**: visit `index.html?selftest` — `runSelfTest()` finite-differences every analytic gradient across all reward presets, sanity-checks $\pi^\star$ (normalization; $J(\pi^\star)$ must dominate the Gaussian landscape grid), and checks each algorithm core against its own surrogate on a frozen, seeded batch. DRO tests additionally cover practical log-$\sigma$ updates, exact value profiling, population profiled gradients, a realizable quadratic target, and reward-offset invariance. Run this after any change to the math. It can also be run headlessly: extract the inline `<script>`, stub the DOM (see the memory note on JXA), and call `runSelfTest()` via `osascript -l JavaScript`.

## Code architecture

Everything lives in `index.html`: HTML structure, inline CSS in `:root`-themed custom properties, and a single inline `<script>` block. The script is structured as labeled sections (see the `/* === */` banners):

- **`C` (palette) and RNG (top of script)** — `C` holds the semantic colors read once from the CSS custom properties (`--pi`, `--star`, …); use `C.*` (and `hexToRgba`) in Plotly traces, never hard-coded hexes. `setSeed`/`rand`/`randn` form a seedable RNG stream (mulberry32); `S.seed > 0` makes runs reproducible and `compareRunAll` re-seeds at the start of each Run all.
- **`S` (global state)** — single source of truth for the page. Holds reference/policy params (`muRef/sigRef`, `muTh/sigTh`, `beta`, `rewardPreset`, `seed`), per-algorithm hyperparameters and learning rates, DRO's `droMode`/`droSurface`/`droDatasetId`, per-algorithm sample snapshots, trajectory and aligned diagnostic-history arrays, Compare J-curves, play-loop handles, and the PPO/GRPO refresh counters. Mutate `S` then call `renderAll()` (or `requestRender()` from high-frequency handlers); do not keep parallel state in the DOM.
- **Reward model** — `REWARD_PRESETS` defines named lists of bumps `{w, c, s}`; `reward(a)` evaluates the sum. Adding a preset means adding an entry here and an `<option>` in the `rewardPreset` `<select>`.
- **Analytic objective** — because both the policy and each reward bump are Gaussian, $\mathbb{E}_{\pi_\theta}[r]$, $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$, and their $\mu, \sigma$ gradients have closed forms. These live in `expectedReward`, `klGaussian`, `objective`, `gradExpected`, `gradKL`, `gradJ`. Edit these together if you change the model — they must stay consistent, and `?selftest` will catch drift.
- **Targets and exact oracles** — `computeStar()` evaluates the unrestricted $\pi^\star \propto \pi_{\mathrm{ref}}\exp(r/\beta)$ numerically; `computeGaussianJOracle()` finds the best Gaussian for $J$. These are distinct whenever the unrestricted target is not Gaussian. DRO has analytic Gaussian-moment helpers plus `droPopulationOracle()` and `droEmpiricalOracle()`, which solve the profiled quadratic log-ratio fit in Gaussian natural parameters. Cache keys must include every environment, temperature, and dataset dependency.
- **Algorithm cores** — the single home of each algorithm's math: `exactStepCore`, `pgStepCore`, `ppoStepCore`, `grpoStepCore`, `dpoStepCore`, `droStepCore`, plus their shared helpers. `droStepCore` supports `raw` (legacy direct-$\sigma$ MSE descent), `practical` ($1/\beta$ policy rescaling and log-$\sigma$), and `profile` (practical update with exact batch-$V$ profiling). Cores are pure in the policy: they take state, batch, and params and return updated values; they never touch `S.muTh/S.sigTh` or the DOM. **Both the interactive tabs and the Compare runner call these cores — never re-implement an update rule in either place.**
- **Interactive tab wrappers** — each algorithm keeps the consistent triple around its core: a `*Resample()` that draws a fresh sample snapshot into `S.{pg,ppo,grpo,dpo,dro}`, a `recompute*Quantities()` that recomputes derived per-sample display quantities at the current $\theta$ (advantages, ratios, surrogate, DPO margins, DRO residuals), and a `*Step()` that calls the core and applies the result. PPO/GRPO snapshot $\pi_{\mathrm{old}}$ on each resample; their Play loops resample every `Kppo`/`Kgrpo` ticks so ratios drift from 1 and clipping engages between refreshes. Every step pushes onto its trajectory array so the landscape panel can draw the path.
- **Compare runner** — `compareRunAll()` (~line 1708) re-runs all six algorithms from the user's current $(\mu_\theta, \sigma_\theta)$ using `COMPARE_DEFAULTS` hyperparameters scaled by `cmpScale`, honoring the same K-epoch cadence, optionally repeating each stochastic algorithm `cmpRuns` times. It fills the six `traj*` arrays (first run) and `S.cmpJ` (J-curve per run), then restores the policy. Sample counts are clamped (`G≥2` for the $(G-1)$ std denominator, `N≥1` for DPO pairs, etc.). Keep `COMPARE_DEFAULTS` in sync with the slider defaults if you change them.
- **Rendering** — `renderAll()` reads `data-active-tab` off `<body>` and renders only the active tab's panels (rendering into a `display:none` container makes Plotly mis-size the SVG). Shared panels are `renderDensities`, `renderLandscape`, and `renderMetrics`. The DRO tab uses `renderDROLandscape` to switch among $J$, profiled population loss, and profiled current-batch loss while retaining one trajectory; `droLossColorEncoding` applies a monotone asinh/q90 color encoding to the two loss surfaces while preserving raw values in `customdata`. Its other plots show target-versus-fit residual geometry and aligned batch/population value/loss histories. `renderCompareJ` shows both the unrestricted and best-Gaussian $J$ ceilings.
- **Tabs & play loops** — `setTab(name)` toggles `data-active-tab` on `<body>`, maintains `aria-selected`/roving `tabIndex` (arrow keys work via a keydown handler), and triggers `renderAll()`. Each algorithm has its own `togglePlay*()` that drives the corresponding `*Step()` via `setInterval`.
- **URL hash state** — `writeHash`/`readHash` serialize every slider that differs from its HTML default, plus `rewardPreset`, `droMode`, `droSurface`, `seed`, and the active tab, into `location.hash` (debounced via `scheduleHashWrite`). New sliders must be added to `HASH_SLIDER_IDS`; non-slider controls need explicit serialization/restoration.
- **UI binding** — `bindSlider(id, key, decimals, after)` near the bottom wires each `<input type="range">` to a key in `S`, schedules a hash write, and re-renders through `requestRender()` (rAF-coalesced). New controls follow the same pattern. `resetAll()` is the inverse — keep it in sync when you add a slider, play loop, or Compare-tab control (it also resets `compareVisible`, the Compare sliders, the seed, and clears the hash).

When changing the math, keep each objective internally consistent and keep its target explicit. `computeStar()` is the unrestricted RLHF optimum, `computeGaussianJOracle()` is the restricted Gaussian maximizer of $J$, and the empirical/population DRO oracles minimize their own profiled losses; these points need not coincide. The six `traj*` arrays remain comparable because `renderCompareJ` scores every resulting policy with the same experiment $J$. Verify math changes with `?selftest`.

## Conventions

- Vanilla JS only — no bundler, no modules, no framework. Keep everything in `index.html` unless the user explicitly asks to split it.
- Color tokens (`--ref`, `--pi`, `--star`, `--reward`, `--accent`, plus per-algorithm accents) are the semantic palette; in JS use the `C` object (populated from those tokens at load) rather than hard-coding hex values.
- KaTeX renders math at page load via the auto-render `onload` hook; any math added to dynamically-rendered content must be re-rendered explicitly or kept in static HTML.

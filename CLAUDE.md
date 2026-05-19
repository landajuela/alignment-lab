# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An interactive, browser-based visualization of policy alignment on a toy problem. The "language model" is a 1D Gaussian policy $\pi_\theta = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$, the reward $r(a)$ is a sum of Gaussian bumps, and a reference policy $\pi_{\mathrm{ref}}$ anchors the optimization via a KL penalty. The app contrasts five alignment algorithms on the same objective:

- **RLHF exact** — analytic $\nabla_\theta J$.
- **PG** — REINFORCE with an in-batch mean baseline.
- **GRPO** — clipped, group-normalized surrogate against a frozen $\pi_{\mathrm{old}}$.
- **DPO** — pairwise preference loss with Bradley-Terry noise.
- **DRO** — Direct Reward Optimisation (Richemond et al., 2024) with a learned baseline $V$.

A **Compare** tab runs all five from the same starting $(\mu_\theta, \sigma_\theta)$ and overlays their trajectories on the loss landscape.

The original Mathematica reference (`mathematica_example.nb`) and brief (`prompt.md`) live in the sibling directory `/Users/landajuela/Work/GitHub/alignment/`, not in this repo. Pre-approved grep patterns for the notebook are in `.claude/settings.local.json`.

## How to run

There is no build step, no package manager, and no test suite. Everything is one static HTML file with CDN-loaded dependencies (Plotly, KaTeX, Google Fonts).

- **Recommended**: open in VS Code and use Live Server (configured for port 5501 in `.vscode/settings.json`).
- **Alternative**: `python3 -m http.server 5501` then visit `http://localhost:5501/index.html`, or `open index.html` for a quick look (some browsers restrict CDN/font behavior under `file://`).

## Code architecture

Everything lives in `index.html` (~2250 lines): HTML structure, inline CSS in `:root`-themed custom properties, and a single inline `<script>` block starting at line 862. The script is structured as labeled sections (see the `/* === */` banners):

- **`S` (global state, ~line 868)** — single source of truth for the page. Holds reference/policy params (`muRef/sigRef`, `muTh/sigTh`, `beta`, `rewardPreset`), per-algorithm hyperparameters and learning rates (`lrExact`; `NPG/lrPG`; `G/eps/lrG`; `betaDPO/N/lrDPO/prefNoise`; `betaDRO/NDRO/lrDROth/lrDROv/Vdro`), per-algorithm sample snapshots (`pg`, `grpo`, `dpo`, `dro`), trajectory arrays (`trajExact/trajPG/trajGRPO/trajDPO/trajDRO`), and play-loop handles. Mutate `S` then call `renderAll()`; do not keep parallel state in the DOM.
- **Reward model** — `REWARD_PRESETS` defines named lists of bumps `{w, c, s}`; `reward(a)` evaluates the sum. Adding a preset means adding an entry here and an `<option>` in the `rewardPreset` `<select>`.
- **Analytic objective** — because both the policy and each reward bump are Gaussian, $\mathbb{E}_{\pi_\theta}[r]$, $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$, and their $\mu, \sigma$ gradients have closed forms. These live in `expectedReward`, `klGaussian`, `objective`, `gradExpected`, `gradKL`, `gradJ`. Edit these together if you change the model — they must stay consistent.
- **`π*` (optimal policy)** — `computeStar()` evaluates $\pi^\star \propto \pi_{\mathrm{ref}} \exp(r/\beta)$ numerically on a fixed action grid (`A_GRID`, `A_MIN/A_MAX/A_N`).
- **Algorithm steps** — each algorithm follows a consistent triple: a `*Resample()` that draws a fresh sample snapshot into `S.{pg,grpo,dpo,dro}`, a `recompute*Quantities()` that recomputes derived per-sample quantities at the current $\theta$ (advantages, ratios, surrogate, DPO margins, DRO residuals), and a `*Step()` that applies the parameter update. GRPO additionally snapshots $\pi_{\mathrm{old}}$ on each resample. `stepExact()` is the analytic counterpart with no sampling. Every step pushes onto its trajectory array so the landscape panel can draw the path.
- **Compare runner** — `compareRunAll()` (~line 1320) re-runs all five algorithms from the user's current $(\mu_\theta, \sigma_\theta)$ using `COMPARE_DEFAULTS` hyperparameters scaled by `cmpScale`, fills the five `traj*` arrays, and restores the policy. Sample counts are clamped (`G≥2` for the $(G-1)$ std denominator, `N≥1` for DPO pairs, etc.). Keep `COMPARE_DEFAULTS` in sync with the slider defaults if you change them.
- **Rendering** — `renderAll()` reads `data-active-tab` off `<body>` and renders only the active tab's panels (rendering into a `display:none` container makes Plotly mis-size the SVG). Shared panels: `renderDensities` (policies + reward), `renderLandscape` (2D $\mu,\sigma$ heatmap with optional per-algorithm trajectory overlays), `renderMetrics`. Tab-specific panels: `renderPGPanels`, `renderGRPOPanels`, `renderDPOPanels`, `renderDROPanels`. The Compare tab uses `renderLandscape` with all five path flags and a `cornerLegend`.
- **Tabs & play loops** — `setTab(name)` toggles `data-active-tab` on `<body>` and triggers `renderAll()`. Each algorithm has its own `togglePlay*()` that drives the corresponding `*Step()` via `setInterval`.
- **UI binding** — `bindSlider(id, key, decimals, after)` near the bottom wires each `<input type="range">` to a key in `S` and re-renders. New controls follow the same pattern. `resetAll()` is the inverse — keep it in sync when you add a slider or play loop.

When changing the math, the invariant to preserve is: `objective`, the analytic gradients (`gradJ`), the GRPO/PG/DPO/DRO surrogates, the trajectory updates, and the optimum (`computeStar`) all refer to the same definition of $J(\theta) = \mathbb{E}_{\pi_\theta}[r] - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})$. The five `traj*` arrays must remain comparable on the same landscape.

## Conventions

- Vanilla JS only — no bundler, no modules, no framework. Keep everything in `index.html` unless the user explicitly asks to split it.
- Color tokens (`--ref`, `--pi`, `--star`, `--reward`, `--accent`, plus per-algorithm accents) are the semantic palette; reuse them rather than hard-coding hex values.
- KaTeX renders math at page load via the auto-render `onload` hook; any math added to dynamically-rendered content must be re-rendered explicitly or kept in static HTML.

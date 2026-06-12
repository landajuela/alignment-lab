# alignment-lab

Interactive browser visualization comparing six alignment algorithms — **RLHF exact**, **PG** (REINFORCE), **PPO**, **GRPO**, **DPO**, and **DRO** — on a 1D Gaussian policy $\pi_\theta = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$ with a KL-anchored reference $\pi_{\mathrm{ref}}$ and a sum-of-Gaussian-bumps reward.

Each algorithm has its own tab (samples, advantages/ratios/margins/residuals as appropriate) plus a shared loss landscape over $(\mu_\theta, \sigma_\theta)$. The **Compare** tab runs all six from the same starting point, overlays their trajectories, and charts $J(\theta_t)$ over steps — optionally as a median + min–max band over multiple runs per algorithm.

## Run

No build, no package manager. Just open `index.html`.

- VS Code Live Server (port 5501 is configured in `.vscode/settings.json`), or
- `python3 -m http.server 5501` then visit http://localhost:5501/index.html

Dependencies (Plotly, KaTeX, Google Fonts) load from CDNs (with subresource-integrity pins).

## Features

- **Seeded runs** — set a nonzero seed in the sidebar to make all sampling reproducible; *Run all* on the Compare tab then replays bit-identically.
- **Shareable URLs** — every control that differs from its default is serialized into `location.hash`, so a specific configuration (reward preset, β, starting point, active tab, …) can be sent as a link.
- **Inner epochs (K)** — PPO and GRPO take K gradient steps per batch before refreshing $\pi_{\mathrm{old}}$, so the clipped surrogate actually engages (ratios drift from 1 between refreshes).
- **Self-test** — open `index.html?selftest` to run finite-difference checks of every analytic gradient across all reward presets, plus sanity checks on $\pi^\star$; results land in the console and a banner.

## Files

- `index.html` — everything: markup, CSS, and the single inline script.
- `CLAUDE.md` — architecture notes for working in this repo.

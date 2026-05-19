# alignment-lab

Interactive browser visualization comparing five alignment algorithms — **RLHF exact**, **PG** (REINFORCE), **GRPO**, **DPO**, and **DRO** — on a 1D Gaussian policy $\pi_\theta = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$ with a KL-anchored reference $\pi_{\mathrm{ref}}$ and a sum-of-Gaussian-bumps reward.

Each algorithm has its own tab (samples, advantages/ratios/margins/residuals as appropriate) plus a shared loss landscape over $(\mu_\theta, \sigma_\theta)$. The **Compare** tab runs all five from the same starting point and overlays their trajectories.

## Run

No build, no package manager. Just open `index.html`.

- VS Code Live Server (port 5501 is configured in `.vscode/settings.json`), or
- `python3 -m http.server 5501` then visit http://localhost:5501/index.html

Dependencies (Plotly, KaTeX, Google Fonts) load from CDNs.

## Files

- `index.html` — everything: markup, CSS, and the single inline script.
- `CLAUDE.md` — architecture notes for working in this repo.

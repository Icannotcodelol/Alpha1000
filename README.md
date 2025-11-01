<div align="center">

# Alpha‑1000

**Building the strongest possible two‑player _Tysiąc_ (1000) agent.**

</div>

---

## Current Status

- ✅ **Rule-accurate engine** with deterministic tests (bidding proof, trump override, 800-lock scoring, etc.).
- ✅ **Self-play infrastructure** (FastAPI match service, React “Alpha1000 Arena” web UI, CLI tools).  
  Includes Cloudflare Quick Tunnel instructions for ad‑hoc online play.
- ✅ **Baseline bots** and analytical scripts (`bots/bot_arena.py`, evaluation CSV logging).
- ✅ **PPO‑LSTM training pipeline** with larger 1024×512 policy/value net, richer observations (meld flags, trump counts, contract margin, trick/card counters).
- ✅ **League automation** (`scripts/overnight_league.py`) with diversified self-play roster (mutated champions) and adaptive checkpoint loading for expanded feature sets.
- ⏳ **Current champion** remains `ppo_update_3485.pt` and its variants — new checkpoints dominate scripted bots but still lose all matches to the champion family. We’re in an optimization plateau.

---

## Goal

> **Deliver the world’s best 1000 (Tysiąc) player** — a model that consistently beats all scripted baselines, historical champions, and strong human opposition, and can be deployed to a public web interface for anyone to challenge.

Everything in this repository pushes toward that single objective.

---

## Repository Overview

```
alpha-1000/
├── engine/                 # Core game logic & encoders
├── bots/                   # Greedy, trump-manager, counter, self-play wrappers
├── rl/                     # PPO-LSTM implementation (AlphaZero WIP)
├── ui/                     # Streamlit & React (Vite) front-ends
├── scripts/                # Training/eval automation, checkpoint tools
└── tests/                  # Pytest suite validating every rule edge case
```

Key scripts:

- `scripts/overnight_league.py` – cyclic self-play league with evaluation/promotions.
- `scripts/evaluate_checkpoints.py` – batch evaluation vs scripted & self opponents.
- `scripts/mutate_checkpoint.py` – create noisy champion variants for league diversity.
- `scripts/play_vs_agent.py` – CLI match vs champion.
- `server/play_service.py` + `ui/web/` – FastAPI + React visual arena.

---

## How to Reproduce the Current Setup

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Tests
pytest -q

# 3. FastAPI service (arena backend)
./.venv/bin/uvicorn server.play_service:app --host 0.0.0.0 --port 8000 --reload

# 4. React arena (frontend)
cd ui/web
VITE_API_BASE="http://localhost:8000" npm install
VITE_API_BASE="http://localhost:8000" npm run dev

# 5. PPO League run (example)
nohup ./scripts/overnight_league.py \
  --cycles 6 \
  --resume data/checkpoints/ppo_update_6055.pt \
  --selfplay data/checkpoints/ppo_update_3185.pt \
  --selfplay data/checkpoints/ppo_update_3485.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_11.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_23.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_42.pt \
  --schedule 'self0:0.30,self1:0.30,self2:0.15,self3:0.15,self4:0.10@300;self0:0.25,self1:0.25,self2:0.20,self3:0.20,self4:0.10@300;self0:0.25,self1:0.25,self2:0.20,self3:0.20,self4:0.05,trump:0.05@200' \
  --model-hidden 1024 --model-lstm 512 \
  --learning-rate 5e-5 --entropy-coef 0.005 \
  --log-csv data/training_log_trump_v3.csv \
  --eval-episodes 200 --self-threshold 10 --max-selfplay 6 \
  > data/run_logs/league_master.log 2>&1 &

# 6. Evaluate latest checkpoints
./scripts/evaluate_checkpoints.py \
  --checkpoints 'data/checkpoints/ppo_update_*.pt' \
  --limit 10 \
  --episodes 200 \
  --selfplay data/checkpoints/ppo_update_3185.pt \
  --selfplay data/checkpoints/ppo_update_3485.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_11.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_23.pt \
  --selfplay data/checkpoints/ppo_update_3485_mut_42.pt \
  --device cpu \
  --json data/eval_latest.json \
  --csv data/eval_history.csv
```

To share the arena quickly without a custom domain, run Cloudflare Quick Tunnels:

```bash
# Backend tunnel
cloudflared tunnel --url http://localhost:8000

# Frontend tunnel (after running npm preview on port 4173)
cloudflared tunnel --url http://localhost:4173
```

Set `VITE_API_BASE` to the backend tunnel URL before launching the frontend preview.

---

## Play Against the Champion

1. Launch the FastAPI backend (`uvicorn server.play_service:app ...`) so the current champion roster is available. The `/play/session` endpoint autoloads the latest promoted model specified in `data/current_champion.json`.
2. Start the React arena (`npm run dev` or `npm run preview` in `ui/web`) with `VITE_API_BASE` pointing at the backend URL (local or tunneled). The lobby dropdown lists all packaged checkpoints; select `Champion` to face the reigning agent.
3. Open the provided URL in a browser. Click a card to select, then confirm actions using the button prompts—no drag-and-drop required.
4. For quick terminal matches, run `./scripts/play_vs_agent.py --agent data/checkpoints/ppo_update_6055.pt` (or any checkpoint). The script prints hand-by-hand summaries so you can review bidding decisions without the UI.

These steps mirror the workflow we use internally before pushing checkpoints to remote evaluators or sharing a Cloudflare tunnel with external testers.

---

## Recent Findings

- The expanded observation vector (meld flags, trump counts, score margin, tricks remaining) was essential to keep training stable after increasing model capacity.
- Even with those features and a diverse self-play roster, the league’s newest checkpoints (up to `ppo_update_12655.pt`) still record ~0 % wins against the champion lineage (`ppo_update_3485` and its mutants) while crushing scripted opponents.
- This plateau indicates the agent either still lacks key information or needs a different training objective (e.g., match-level reward, search guidance) to surpass the champions.

---

## Roadmap

### 1. Enhance Observations & Architecture
- Add higher-order features (opponent inferred hand strength, history of trump changes, contract streaks).
- Experiment with deeper/residual policy heads or transformer encoders.

### 2. Match-Level Training
- Wrap environment to reward full 1000-point match outcomes (win/loss, contract streak bonuses).
- Blend hand-level shaping with match-level rewards to encourage strategic risk management.

### 3. League & Evaluation Upgrades
- Track Elo/Glicko ratings for checkpoints and champion variants.
- Automate best-of-N full matches (`scripts/eval_matches.py`) for top candidates before promotion.
- Consider self-play curriculum that mixes “last N champions” instead of a fixed roster.

### 4. Search-Augmented Agents (Longer-term)
- Implement AlphaZero-style determinization (ISMCTS) as an expert that guides PPO or serves as a stronger champion.
- Explore model distillation from search to policy-only networks.

### 5. Deployment
- Quantize/distill the champion for CPU inference.
- Deploy the FastAPI service to a managed platform (Railway/Fly.io/Replicate).
- Host the React arena on Netlify/Vercel and gate the API (rate limiting, optional auth).

---

## Contributing / Using AI Coding Assistants

When using GPT-Codecs/Cursor, provide concrete tasks and refer to this README as the source of truth. After every generated change, run the test suite and the evaluation scripts, and only accept modifications that keep the rules intact.

---

## License

This repository is intended for research, experimentation, and non-commercial deployment of 1000/Tysiąc agents. License will be finalized once the project reaches champion-level performance.

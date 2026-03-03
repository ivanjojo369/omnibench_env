@'
# Training / Baselines (OpenEnv OmniBench)

This folder contains **reproducible baseline scripts** to interact with the OmniBench OpenEnv environment server.
Goal: provide a simple, public, runnable reference that can be extended into real training (RL / imitation / LLM-based).

## Baseline included
- `baseline_solver.py`: a minimal **rule-based** solver that uses the environment API:
  - `POST /reset`
  - `POST /step`
  - reads observations and issues actions (tool-calls or final responses)

It produces a JSONL log with episode traces.

## Run (local)
1) Run the env server (Docker or local):
- Docker (example): `docker run --rm -p 8003:8000 <image>`
2) Run baseline:
```bash
uv run --project . python training/baseline_solver.py --base-url http://127.0.0.1:8003 --out training/results/local_baseline.jsonl
